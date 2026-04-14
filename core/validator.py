# -*- coding: utf-8 -*-
"""
validator.py – Road network validation for SkogVegPlanering (STEG 2)

Validates forest road segments against Norwegian forest road standards:
  - Maximum slope (default ≤ 12 %)
  - Minimum curve radius (default ≥ 20 m)
  - Road-class specific width and bearing-capacity requirements

The computation functions work with plain (x, y) / (x, y, z) tuples so
they can be unit-tested independently of QGIS.  The public method
``validate_layer`` accepts a QgsVectorLayer and adapts its geometry.
"""

import math
import sys
import logging

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

class ValidationResult:
    """Holds the outcome of a single validation check."""

    def __init__(self, feature_id, check_type, passed, message, value=None):
        self.feature_id = feature_id
        self.check_type = check_type
        self.passed = passed
        self.message = message
        self.value = value

    def __repr__(self):
        status = "✓" if self.passed else "✗"
        return f"[{status}] Feature {self.feature_id} – {self.check_type}: {self.message}"


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class RoadValidator:
    """
    Validates a forest road network against Norwegian standards.

    Norwegian forest road classes (Skogsvegklasse):
      Class 1 – main forest road: width ≥ 3.5 m, slope ≤ 8 %, bearing ≥ 10 t
      Class 2 – secondary road : width ≥ 3.0 m, slope ≤ 10 %, bearing ≥  8 t
      Class 3 – branch road    : width ≥ 2.5 m, slope ≤ 12 %, bearing ≥  6 t
    """

    ROAD_CLASSES = {
        1: {"min_width_m": 3.5, "max_slope_pct": 8.0,  "min_bearing_tons": 10},
        2: {"min_width_m": 3.0, "max_slope_pct": 10.0, "min_bearing_tons":  8},
        3: {"min_width_m": 2.5, "max_slope_pct": 12.0, "min_bearing_tons":  6},
    }

    def __init__(self, max_slope_percent=12.0, min_curve_radius=20.0):
        self.max_slope_percent = max_slope_percent
        self.min_curve_radius = min_curve_radius

    # ------------------------------------------------------------------
    # Public: QGIS layer entry point
    # ------------------------------------------------------------------

    def validate_layer(self, layer):
        """
        Validate every feature in a QgsVectorLayer.

        Parameters
        ----------
        layer : QgsVectorLayer
            A line/multiline layer with road segments.

        Returns
        -------
        list[ValidationResult]
        """
        results = []
        field_names = [f.name() for f in layer.fields()]

        for feature in layer.getFeatures():
            fid = feature.id()
            geom = feature.geometry()

            if geom is None or geom.isEmpty():
                log.warning("Feature %s has empty geometry – skipped", fid)
                continue

            # Collect polyline vertices as plain tuples
            polylines = self._extract_polylines(geom)

            for pts in polylines:
                results.extend(self.check_curve_radius_points(fid, pts))

                # Slope requires Z coordinates
                if pts and len(pts[0]) == 3:
                    results.extend(self.check_slope_points(fid, pts))

            # Attribute-based checks
            road_class = feature["road_class"] if "road_class" in field_names else None
            width_m    = feature["width_m"]    if "width_m"    in field_names else None
            bearing    = feature["bearing_tons"] if "bearing_tons" in field_names else None

            if road_class is not None:
                results.extend(
                    self.check_road_class(fid, int(road_class), width_m, bearing)
                )

        return results

    # ------------------------------------------------------------------
    # Public: pure-Python checks (unit-testable without QGIS)
    # ------------------------------------------------------------------

    def check_slope_simple(self, feature_id, dz, dx):
        """
        Check slope from elevation difference *dz* and horizontal distance *dx*.

        Returns a single ValidationResult.
        """
        if dx == 0:
            return ValidationResult(
                feature_id, "slope", False, "Null horisontal avstand", None
            )
        slope_pct = (abs(dz) / dx) * 100
        passed = slope_pct <= self.max_slope_percent
        return ValidationResult(
            feature_id=feature_id,
            check_type="slope",
            passed=passed,
            message=f"Stigning {slope_pct:.1f}% (maks {self.max_slope_percent:.0f}%)",
            value=slope_pct,
        )

    def check_slope_points(self, feature_id, points):
        """
        Check slope for each segment of a 3-D polyline.

        Parameters
        ----------
        points : list[tuple]
            Each element is ``(x, y, z)``.

        Returns
        -------
        list[ValidationResult]
        """
        results = []
        for i in range(len(points) - 1):
            x1, y1, z1 = points[i]
            x2, y2, z2 = points[i + 1]
            dx = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if dx == 0:
                continue
            results.append(self.check_slope_simple(feature_id, z2 - z1, dx))
        return results

    def check_curve_radius_points(self, feature_id, points):
        """
        Check minimum curve radius at every interior vertex of a polyline.

        Parameters
        ----------
        points : list[tuple]
            Each element is ``(x, y)`` or ``(x, y, z)``.

        Returns
        -------
        list[ValidationResult]
        """
        results = []
        for i in range(1, len(points) - 1):
            p1, p2, p3 = points[i - 1], points[i], points[i + 1]
            radius = self._circumradius(p1[0], p1[1], p2[0], p2[1], p3[0], p3[1])
            if radius is None:
                continue
            passed = radius >= self.min_curve_radius
            results.append(ValidationResult(
                feature_id=feature_id,
                check_type="curve_radius",
                passed=passed,
                message=(
                    f"Kurvaturradius {radius:.1f} m "
                    f"(min {self.min_curve_radius:.0f} m)"
                ),
                value=radius,
            ))
        return results

    def check_road_class(self, feature_id, road_class, width_m=None, bearing_tons=None):
        """
        Check road-class-specific attribute requirements.

        Parameters
        ----------
        feature_id : int
        road_class : int
            1, 2, or 3 (Norwegian forest road classes).
        width_m : float | None
        bearing_tons : float | None

        Returns
        -------
        list[ValidationResult]
        """
        results = []

        if road_class not in self.ROAD_CLASSES:
            results.append(ValidationResult(
                feature_id=feature_id,
                check_type="road_class",
                passed=False,
                message=f"Ukjent vegklasse: {road_class}",
            ))
            return results

        std = self.ROAD_CLASSES[road_class]

        if width_m is not None:
            passed = float(width_m) >= std["min_width_m"]
            results.append(ValidationResult(
                feature_id=feature_id,
                check_type="road_width",
                passed=passed,
                message=(
                    f"Vegbredde {width_m} m "
                    f"(min {std['min_width_m']} m for klasse {road_class})"
                ),
                value=float(width_m),
            ))

        if bearing_tons is not None:
            passed = float(bearing_tons) >= std["min_bearing_tons"]
            results.append(ValidationResult(
                feature_id=feature_id,
                check_type="bearing_capacity",
                passed=passed,
                message=(
                    f"Bæreevne {bearing_tons} t "
                    f"(min {std['min_bearing_tons']} t for klasse {road_class})"
                ),
                value=float(bearing_tons),
            ))

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_polylines(geom):
        """Convert a QgsGeometry line/multiline to list of point-tuple lists."""
        polylines = []
        wkb_type = geom.wkbType()

        # WKB type constants: LineString=2, MultiLineString=5 (and 3D variants)
        if geom.isMultipart():
            parts = geom.asMultiPolyline()
        else:
            parts = [geom.asPolyline()]

        for part in parts:
            pts = []
            for qpt in part:
                try:
                    pts.append((qpt.x(), qpt.y(), qpt.z()))
                except Exception:
                    pts.append((qpt.x(), qpt.y()))
            if len(pts) >= 2:
                polylines.append(pts)

        return polylines

    @staticmethod
    def _circumradius(ax, ay, bx, by, cx, cy):
        """Return circumscribed circle radius for three 2-D points, or None."""
        a = math.sqrt((bx - ax) ** 2 + (by - ay) ** 2)
        b = math.sqrt((cx - bx) ** 2 + (cy - by) ** 2)
        c = math.sqrt((ax - cx) ** 2 + (ay - cy) ** 2)

        s = (a + b + c) / 2
        area_sq = s * (s - a) * (s - b) * (s - c)
        if area_sq <= 0:
            return None
        area = math.sqrt(area_sq)
        if area == 0:
            return None
        return (a * b * c) / (4 * area)
