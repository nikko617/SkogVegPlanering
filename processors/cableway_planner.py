# -*- coding: utf-8 -*-
"""
cableway_planner.py – Cable Way Planning for SkogVegPlanering (STEG 5)

Norwegian cable way (taubane) planning model
--------------------------------------------
In Norwegian forestry, timber from steep terrain is extracted using cable
way systems.  The key planning task is choosing *standplass* positions –
the anchor / landing points where logging machinery stands – along a
forest road, so that every part of the harvest area is reachable within
the maximum winch cable length (vinsjelengde).

This module computes standplass positions along a road polyline and derives
the coverage radius at each station.  All heavy geometry is pure Python /
math – no QGIS runtime required.  QGIS layer integration is in
``ui/cableway_dialog.py``.

Key concepts
------------
station_interval_m : float
    Spacing between consecutive standplass along the road (default 200 m).
    Chosen to overlap coverage zones so no gap is left unharvested.
winch_length_m : float
    Maximum reach of the cable drum from each standplass (default 400 m).
    Determines the coverage radius of each station.
max_span_m : float
    Maximum allowable span between two cable towers / anchor anchors.
    Used to flag segments that are too long and need an intermediate tower.
min_slope_pct : float
    Minimum terrain slope (%) to justify a cable way.  Below this value the
    terrain is gentle enough for machinery access – a warning is emitted.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

Point2D = Tuple[float, float]   # (x, y)
Point3D = Tuple[float, float, float]   # (x, y, z)


@dataclass
class CablewayStation:
    """
    A single standplass (cable way station / anchor point).

    Attributes
    ----------
    station_id : int
        Sequential index starting at 1.
    x, y : float
        Map coordinates of the standplass.
    z : float
        Elevation in metres.  0.0 if no Z data.
    distance_along_road_m : float
        Cumulative road distance from the start of the polyline to this
        station (metres).
    coverage_radius_m : float
        The winch reach at this station (= winch_length_m parameter).
    slope_pct : float
        Terrain slope at this station derived from the two neighbouring
        road segments (%). 0.0 if no Z data.
    """
    station_id: int
    x: float
    y: float
    z: float
    distance_along_road_m: float
    coverage_radius_m: float
    slope_pct: float = 0.0

    @property
    def xy(self) -> Point2D:
        return (self.x, self.y)

    def __repr__(self):
        return (
            f"CablewayStation({self.station_id}, "
            f"d={self.distance_along_road_m:.0f}m, "
            f"r={self.coverage_radius_m:.0f}m, "
            f"slope={self.slope_pct:.1f}%)"
        )


@dataclass
class CablewaySegment:
    """
    The cable span between two consecutive standplass stations.

    Attributes
    ----------
    from_station : CablewayStation
    to_station   : CablewayStation
    span_m : float
        Horizontal distance between the two stations (metres).
    slope_pct : float
        Slope of the segment (%).  Requires Z data.
    exceeds_max_span : bool
        True when ``span_m > max_span_m``.
    """
    from_station: CablewayStation
    to_station:   CablewayStation
    span_m: float
    slope_pct: float
    exceeds_max_span: bool

    @property
    def midpoint(self) -> Point2D:
        return (
            (self.from_station.x + self.to_station.x) / 2,
            (self.from_station.y + self.to_station.y) / 2,
        )

    def __repr__(self):
        flag = " [SPAN EXCEEDED]" if self.exceeds_max_span else ""
        return (
            f"CablewaySegment("
            f"{self.from_station.station_id}->{self.to_station.station_id}, "
            f"{self.span_m:.0f}m, {self.slope_pct:.1f}%){flag}"
        )


@dataclass
class CablewayPlan:
    """Result of a full cable way planning run for one polyline."""

    polyline_id: object
    station_interval_m: float
    winch_length_m: float
    max_span_m: float
    stations: List[CablewayStation] = field(default_factory=list)
    segments: List[CablewaySegment] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def total_road_length_m(self) -> float:
        """Total length of the source polyline."""
        if not self.stations:
            return 0.0
        return self.stations[-1].distance_along_road_m

    @property
    def coverage_overlap_m(self) -> float:
        """
        Theoretical overlap between adjacent coverage zones.

        A positive value means zones overlap (good – no gap).
        A negative value means there is a gap (stations too far apart).
        """
        return 2 * self.winch_length_m - self.station_interval_m

    def __repr__(self):
        return (
            f"CablewayPlan(polyline={self.polyline_id!r}, "
            f"{len(self.stations)} stations, "
            f"{len(self.segments)} segments, "
            f"road={self.total_road_length_m:.0f}m)"
        )


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class CablewayPlanner:
    """
    Auto-generate standplass (cable way stations) along a road polyline.

    Parameters
    ----------
    station_interval_m : float
        Spacing between stations along the road (default 200 m).
    winch_length_m : float
        Coverage radius at each station – the maximum reach of the winch
        cable (default 400 m).
    max_span_m : float
        Maximum cable span allowed between two towers/stations.  Segments
        exceeding this are flagged (default 500 m).
    min_slope_pct : float
        Minimum terrain slope (%) to justify a cable way; stations on
        gentler terrain produce a warning (default 15 %).

    Examples
    --------
    >>> pts = [(0, 0, 100), (200, 0, 90), (400, 0, 75), (600, 0, 55)]
    >>> planner = CablewayPlanner(station_interval_m=200, winch_length_m=300)
    >>> plan = planner.plan(pts)
    >>> len(plan.stations)
    4
    """

    def __init__(
        self,
        station_interval_m: float = 200.0,
        winch_length_m: float = 400.0,
        max_span_m: float = 500.0,
        min_slope_pct: float = 15.0,
    ):
        if station_interval_m <= 0:
            raise ValueError(f"station_interval_m must be > 0 (got {station_interval_m})")
        if winch_length_m <= 0:
            raise ValueError(f"winch_length_m must be > 0 (got {winch_length_m})")
        if max_span_m <= 0:
            raise ValueError(f"max_span_m must be > 0 (got {max_span_m})")
        if min_slope_pct < 0:
            raise ValueError(f"min_slope_pct must be >= 0 (got {min_slope_pct})")

        self.station_interval_m  = station_interval_m
        self.winch_length_m      = winch_length_m
        self.max_span_m          = max_span_m
        self.min_slope_pct       = min_slope_pct

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(
        self,
        points: Sequence,
        polyline_id: object = 0,
    ) -> CablewayPlan:
        """
        Compute standplass positions along a polyline.

        Parameters
        ----------
        points : sequence of (x, y) or (x, y, z) tuples
            Vertices of the road centre-line.
        polyline_id : any
            Identifier carried through to the result (feature ID, etc.).

        Returns
        -------
        CablewayPlan
        """
        result = CablewayPlan(
            polyline_id=polyline_id,
            station_interval_m=self.station_interval_m,
            winch_length_m=self.winch_length_m,
            max_span_m=self.max_span_m,
        )

        if len(points) < 2:
            result.warnings.append(
                f"Polyline {polyline_id}: for fa punkter ({len(points)}) – minst 2 kreves"
            )
            return result

        has_z = len(points[0]) >= 3

        # Build a densely-sampled parametric version of the polyline so we
        # can place stations at exact road-distance intervals.
        samples = self._sample_polyline(points, has_z)
        total_length = samples[-1][0]   # last cumulative distance

        if total_length == 0:
            result.warnings.append(f"Polyline {polyline_id}: total lengde er 0")
            return result

        # Place first station at distance 0, then every station_interval_m
        station_distances = []
        d = 0.0
        while d <= total_length + 1e-9:
            station_distances.append(min(d, total_length))
            d += self.station_interval_m

        # Ensure the last station is at the road end (avoids stranded section)
        if station_distances[-1] < total_length - 1.0:
            station_distances.append(total_length)

        # Interpolate coordinates for each station distance
        for sid, dist in enumerate(station_distances, start=1):
            x, y, z = self._interpolate(samples, dist, has_z)
            slope = self._slope_at(samples, dist) if has_z else 0.0

            station = CablewayStation(
                station_id=sid,
                x=x,
                y=y,
                z=z,
                distance_along_road_m=dist,
                coverage_radius_m=self.winch_length_m,
                slope_pct=slope,
            )
            result.stations.append(station)

            if has_z and slope < self.min_slope_pct and slope >= 0:
                result.warnings.append(
                    f"Standplass {sid} (d={dist:.0f}m): stigning {slope:.1f}% er "
                    f"under minimum {self.min_slope_pct:.0f}% – taubane kanskje "
                    f"ikke nodvendig her"
                )

        # Build segments between consecutive stations
        for i in range(len(result.stations) - 1):
            a = result.stations[i]
            b = result.stations[i + 1]
            span = math.sqrt((b.x - a.x) ** 2 + (b.y - a.y) ** 2)
            dz   = b.z - a.z
            seg_slope = (abs(dz) / span * 100) if span > 0 else 0.0
            exceeds = span > self.max_span_m
            seg = CablewaySegment(
                from_station=a,
                to_station=b,
                span_m=span,
                slope_pct=seg_slope,
                exceeds_max_span=exceeds,
            )
            result.segments.append(seg)
            if exceeds:
                result.warnings.append(
                    f"Spenn {a.station_id}->{b.station_id}: {span:.0f}m "
                    f"> maks {self.max_span_m:.0f}m – vurder ekstra tarn"
                )

        # Coverage overlap check
        if result.coverage_overlap_m < 0:
            result.warnings.append(
                f"Stasjonssavstand ({self.station_interval_m:.0f}m) overstiger "
                f"2 x vinsjelengde ({2 * self.winch_length_m:.0f}m) – "
                f"udekket omrade mulig"
            )

        return result

    def plan_layer(self, layer) -> List[CablewayPlan]:
        """
        Run ``plan()`` for every feature in a QgsVectorLayer.

        Parameters
        ----------
        layer : QgsVectorLayer
            A line/multiline layer (road centre-lines).

        Returns
        -------
        list[CablewayPlan]
        """
        plans = []
        for feature in layer.getFeatures():
            fid = feature.id()
            geom = feature.geometry()
            if geom is None or geom.isEmpty():
                continue
            for pts in self._extract_polylines(geom):
                plans.append(self.plan(pts, polyline_id=fid))
        return plans

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_polylines(geom) -> List[List]:
        """Convert QgsGeometry line/multiline to plain list-of-tuples lists."""
        polylines = []
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
    def _segment_length(p1, p2) -> float:
        """2-D Euclidean distance between two (x,y[,z]) points."""
        return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

    @classmethod
    def _sample_polyline(cls, points, has_z: bool) -> List[Tuple]:
        """
        Build a list of (cumulative_distance, x, y, z) tuples – one per
        input vertex.  z is 0.0 when has_z is False.
        """
        samples = []
        cumulative = 0.0
        for i, p in enumerate(points):
            x, y = p[0], p[1]
            z = float(p[2]) if has_z else 0.0
            if i > 0:
                cumulative += cls._segment_length(points[i - 1], p)
            samples.append((cumulative, x, y, z))
        return samples

    @staticmethod
    def _interpolate(samples: List[Tuple], target_d: float, has_z: bool):
        """
        Linear interpolation: return (x, y, z) at distance ``target_d``
        along the polyline described by ``samples``.
        """
        # Clamp to ends
        if target_d <= samples[0][0]:
            return samples[0][1], samples[0][2], samples[0][3]
        if target_d >= samples[-1][0]:
            return samples[-1][1], samples[-1][2], samples[-1][3]

        for i in range(1, len(samples)):
            d0, x0, y0, z0 = samples[i - 1]
            d1, x1, y1, z1 = samples[i]
            if d0 <= target_d <= d1:
                if d1 == d0:
                    t = 0.0
                else:
                    t = (target_d - d0) / (d1 - d0)
                x = x0 + t * (x1 - x0)
                y = y0 + t * (y1 - y0)
                z = (z0 + t * (z1 - z0)) if has_z else 0.0
                return x, y, z

        return samples[-1][1], samples[-1][2], samples[-1][3]

    @staticmethod
    def _slope_at(samples: List[Tuple], target_d: float) -> float:
        """
        Return terrain slope (%) at position ``target_d`` – derived from
        the polyline segment that contains that position.
        """
        if len(samples) < 2:
            return 0.0

        for i in range(1, len(samples)):
            d0, x0, y0, z0 = samples[i - 1]
            d1, x1, y1, z1 = samples[i]
            if d0 <= target_d <= d1:
                horiz = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
                if horiz == 0:
                    return 0.0
                return abs(z1 - z0) / horiz * 100
        return 0.0
