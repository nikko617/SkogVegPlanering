# -*- coding: utf-8 -*-
"""
cost_calculator.py – Property-based cost distribution for SkogVegPlanering (STEG 4)

Norwegian forest road cost distribution (kostnadsfordeling) model
-----------------------------------------------------------------
Each property (eiendom) that uses a road contributes to its construction and
maintenance costs in proportion to its expected timber transport volume.

The weight assigned to each property can be based on three models:

  * ``"area"``   – weight ∝ property area (enkleste modell)
  * ``"volume"`` – weight ∝ explicit timber volume attribute
  * ``"distance_area"`` – weight ∝ area / sqrt(haul_distance)
    (longer haul distance → heavier wear, but area still dominates)

The class is free of any QGIS imports and can be unit-tested with plain
Python dicts/lists.

QGIS integration is handled in ``ui/cost_dialog.py``.
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

class CostBasis(str, Enum):
    """How property weights are computed."""
    AREA            = "area"
    VOLUME          = "volume"
    DISTANCE_AREA   = "distance_area"


@dataclass
class PropertyRecord:
    """
    Data for a single property (eiendom).

    Parameters
    ----------
    property_id : str | int
        Unique identifier (e.g. feature ID or cadastral number).
    area_m2 : float
        Property area in square metres.  Required for AREA and DISTANCE_AREA bases.
    volume_m3 : float, optional
        Estimated timber volume in cubic metres.  Required for VOLUME basis.
    haul_distance_m : float, optional
        Distance from property centroid to the nearest public road (metres).
        Used by DISTANCE_AREA basis.  Defaults to 1.0 so it never causes
        division by zero.
    name : str, optional
        Human-readable label (e.g. cadastral name or owner).
    """
    property_id: object
    area_m2: float
    volume_m3: float = 0.0
    haul_distance_m: float = 1.0
    name: str = ""

    def __post_init__(self):
        if self.area_m2 < 0:
            raise ValueError(f"area_m2 must be ≥ 0 (got {self.area_m2})")
        if self.volume_m3 < 0:
            raise ValueError(f"volume_m3 must be ≥ 0 (got {self.volume_m3})")
        if self.haul_distance_m <= 0:
            raise ValueError(f"haul_distance_m must be > 0 (got {self.haul_distance_m})")


@dataclass
class CostShare:
    """Cost share result for one property."""

    property_id: object
    name: str
    weight: float           # raw weight value
    share_fraction: float   # 0.0 – 1.0
    share_percent: float    # 0.0 – 100.0
    cost_nok: float         # allocated cost in NOK (or any currency unit)

    def __repr__(self):
        return (
            f"CostShare({self.property_id!r}, "
            f"{self.share_percent:.2f}%, "
            f"{self.cost_nok:,.0f} kr)"
        )


@dataclass
class CostDistributionResult:
    """Result of a full cost distribution run."""

    total_cost_nok: float
    basis: CostBasis
    shares: List[CostShare] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def total_shares_percent(self) -> float:
        return sum(s.share_percent for s in self.shares)

    @property
    def largest_share(self) -> Optional[CostShare]:
        return max(self.shares, key=lambda s: s.share_fraction) if self.shares else None

    @property
    def smallest_share(self) -> Optional[CostShare]:
        return min(self.shares, key=lambda s: s.share_fraction) if self.shares else None


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------

class CostCalculator:
    """
    Distribute road construction/maintenance costs across properties.

    Parameters
    ----------
    basis : CostBasis
        Which weighting model to use (default: ``CostBasis.AREA``).
    min_weight_threshold : float
        Properties with a weight below this fraction of the total weight are
        still included but flagged with a warning.  Default: 0.0 (no filter).

    Examples
    --------
    >>> props = [
    ...     PropertyRecord(1, area_m2=50_000, name="Eiendom A"),
    ...     PropertyRecord(2, area_m2=30_000, name="Eiendom B"),
    ...     PropertyRecord(3, area_m2=20_000, name="Eiendom C"),
    ... ]
    >>> calc = CostCalculator(basis=CostBasis.AREA)
    >>> result = calc.distribute(props, total_cost_nok=1_000_000)
    >>> for s in result.shares:
    ...     print(s)
    CostShare(1, 50.00%, 500 000 kr)
    CostShare(2, 30.00%, 300 000 kr)
    CostShare(3, 20.00%, 200 000 kr)
    """

    def __init__(
        self,
        basis: CostBasis = CostBasis.AREA,
        min_weight_threshold: float = 0.0,
    ):
        self.basis = CostBasis(basis)
        self.min_weight_threshold = min_weight_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def distribute(
        self,
        properties: Sequence[PropertyRecord],
        total_cost_nok: float,
    ) -> CostDistributionResult:
        """
        Calculate cost shares for a list of properties.

        Parameters
        ----------
        properties : sequence of PropertyRecord
        total_cost_nok : float
            Total cost to be distributed (any currency; variable name says NOK
            for Norwegian context but the maths is currency-agnostic).

        Returns
        -------
        CostDistributionResult
        """
        result = CostDistributionResult(
            total_cost_nok=total_cost_nok,
            basis=self.basis,
        )

        if total_cost_nok < 0:
            result.warnings.append("total_cost_nok er negativ – kan gi uventede resultater")

        if not properties:
            result.warnings.append("Ingen eiendommer oppgitt; ingen kostnadsfordeling mulig")
            return result

        # Compute raw weights
        weights: List[Tuple[PropertyRecord, float]] = []
        for prop in properties:
            w = self._weight(prop)
            weights.append((prop, w))

        total_weight = sum(w for _, w in weights)

        if total_weight <= 0:
            result.warnings.append(
                "Total vekt er null – kontroller at eiendommene har gyldige attributter "
                f"for basis='{self.basis.value}'"
            )
            # Give equal share to everyone as fallback
            equal_share = 1.0 / len(weights) if weights else 0.0
            for prop, _ in weights:
                result.shares.append(CostShare(
                    property_id=prop.property_id,
                    name=prop.name,
                    weight=0.0,
                    share_fraction=equal_share,
                    share_percent=equal_share * 100,
                    cost_nok=equal_share * total_cost_nok,
                ))
            return result

        for prop, w in weights:
            frac = w / total_weight
            result.shares.append(CostShare(
                property_id=prop.property_id,
                name=prop.name,
                weight=w,
                share_fraction=frac,
                share_percent=frac * 100,
                cost_nok=frac * total_cost_nok,
            ))

        # Optional: flag very small shares
        if self.min_weight_threshold > 0:
            for share in result.shares:
                if share.share_fraction < self.min_weight_threshold:
                    result.warnings.append(
                        f"Eiendom {share.property_id!r} har svært liten andel "
                        f"({share.share_percent:.3f}%)"
                    )

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _weight(self, prop: PropertyRecord) -> float:
        """Return the raw weight for one property under ``self.basis``."""
        if self.basis == CostBasis.AREA:
            return prop.area_m2

        if self.basis == CostBasis.VOLUME:
            return prop.volume_m3

        if self.basis == CostBasis.DISTANCE_AREA:
            # Weight ∝ area / sqrt(haul_distance) so that properties with
            # longer hauls pay a bit less per m² (they wear the road more but
            # also transport further, so cost attribution is tempered).
            return prop.area_m2 / math.sqrt(prop.haul_distance_m)

        raise ValueError(f"Unknown cost basis: {self.basis}")  # pragma: no cover

    # ------------------------------------------------------------------
    # Convenience: build from a QGIS layer (imported at call time)
    # ------------------------------------------------------------------

    @staticmethod
    def records_from_layer(
        layer,
        area_field: str = "area_m2",
        volume_field: Optional[str] = None,
        distance_field: Optional[str] = None,
        name_field: Optional[str] = None,
    ) -> Tuple[List[PropertyRecord], List[str]]:
        """
        Build ``PropertyRecord`` instances from a QgsVectorLayer.

        Parameters
        ----------
        layer : QgsVectorLayer
            Polygon or any feature layer representing properties.
        area_field : str
            Attribute name for area in m².  If the field does not exist,
            the geometry area is used instead (requires a projected CRS).
        volume_field : str, optional
            Attribute name for timber volume.  Omit or pass None to default to 0.
        distance_field : str, optional
            Attribute name for haul distance.  Omit to default to 1.0.
        name_field : str, optional
            Attribute name for a human-readable property label.

        Returns
        -------
        records : list[PropertyRecord]
        warnings : list[str]
            Non-fatal issues encountered during extraction.
        """
        records: List[PropertyRecord] = []
        warnings: List[str] = []
        field_names = {f.name() for f in layer.fields()}

        for feat in layer.getFeatures():
            fid = feat.id()

            # Area
            if area_field in field_names:
                try:
                    area = float(feat[area_field])
                except (TypeError, ValueError):
                    area = 0.0
                    warnings.append(
                        f"Feature {fid}: ugyldig verdi i '{area_field}'; bruker 0"
                    )
            else:
                geom = feat.geometry()
                area = geom.area() if geom and not geom.isEmpty() else 0.0
                if area == 0.0:
                    warnings.append(f"Feature {fid}: areal er 0 – kontroller geometri/CRS")

            # Volume
            volume = 0.0
            if volume_field and volume_field in field_names:
                try:
                    volume = float(feat[volume_field])
                except (TypeError, ValueError):
                    warnings.append(
                        f"Feature {fid}: ugyldig verdi i '{volume_field}'; bruker 0"
                    )

            # Haul distance
            haul = 1.0
            if distance_field and distance_field in field_names:
                try:
                    raw = float(feat[distance_field])
                    haul = max(raw, 0.001)   # guard against zero/negative
                except (TypeError, ValueError):
                    warnings.append(
                        f"Feature {fid}: ugyldig verdi i '{distance_field}'; bruker 1.0"
                    )

            # Name
            name = ""
            if name_field and name_field in field_names:
                name = str(feat[name_field] or "")

            records.append(PropertyRecord(
                property_id=fid,
                area_m2=max(area, 0.0),
                volume_m3=max(volume, 0.0),
                haul_distance_m=haul,
                name=name,
            ))

        return records, warnings
