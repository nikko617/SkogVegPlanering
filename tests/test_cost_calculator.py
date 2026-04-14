# -*- coding: utf-8 -*-
"""
test_cost_calculator.py – Unit tests for CostCalculator (STEG 4)

All tests use plain Python – no QGIS runtime required.
"""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from processors.cost_calculator import (
    CostBasis,
    CostCalculator,
    CostDistributionResult,
    CostShare,
    PropertyRecord,
)


# ---------------------------------------------------------------------------
# PropertyRecord validation
# ---------------------------------------------------------------------------

class TestPropertyRecord:

    def test_valid_record_created(self):
        r = PropertyRecord(1, area_m2=10_000, volume_m3=500, haul_distance_m=200, name="Skog A")
        assert r.property_id == 1
        assert r.area_m2 == 10_000
        assert r.volume_m3 == 500
        assert r.haul_distance_m == 200
        assert r.name == "Skog A"

    def test_defaults(self):
        r = PropertyRecord(99, area_m2=5_000)
        assert r.volume_m3 == 0.0
        assert r.haul_distance_m == 1.0
        assert r.name == ""

    def test_negative_area_raises(self):
        with pytest.raises(ValueError, match="area_m2"):
            PropertyRecord(1, area_m2=-1)

    def test_negative_volume_raises(self):
        with pytest.raises(ValueError, match="volume_m3"):
            PropertyRecord(1, area_m2=100, volume_m3=-5)

    def test_zero_haul_distance_raises(self):
        with pytest.raises(ValueError, match="haul_distance_m"):
            PropertyRecord(1, area_m2=100, haul_distance_m=0)

    def test_negative_haul_distance_raises(self):
        with pytest.raises(ValueError, match="haul_distance_m"):
            PropertyRecord(1, area_m2=100, haul_distance_m=-10)

    def test_zero_area_is_allowed(self):
        r = PropertyRecord(1, area_m2=0)
        assert r.area_m2 == 0.0


# ---------------------------------------------------------------------------
# CostCalculator – area basis
# ---------------------------------------------------------------------------

class TestAreaBasis:

    def _make_props(self, areas):
        return [PropertyRecord(i + 1, area_m2=a) for i, a in enumerate(areas)]

    def test_equal_areas_give_equal_shares(self):
        props = self._make_props([10_000, 10_000, 10_000])
        result = CostCalculator(CostBasis.AREA).distribute(props, 300_000)
        shares = result.shares
        assert len(shares) == 3
        for s in shares:
            assert s.share_percent == pytest.approx(100 / 3, rel=1e-6)
            assert s.cost_nok == pytest.approx(100_000, rel=1e-6)

    def test_proportional_shares(self):
        props = self._make_props([50_000, 30_000, 20_000])
        result = CostCalculator(CostBasis.AREA).distribute(props, 1_000_000)
        shares = {s.property_id: s for s in result.shares}
        assert shares[1].share_percent == pytest.approx(50.0)
        assert shares[2].share_percent == pytest.approx(30.0)
        assert shares[3].share_percent == pytest.approx(20.0)

    def test_total_cost_allocated_fully(self):
        props = self._make_props([40_000, 35_000, 25_000])
        total = 750_000
        result = CostCalculator(CostBasis.AREA).distribute(props, total)
        assert sum(s.cost_nok for s in result.shares) == pytest.approx(total, rel=1e-9)

    def test_shares_sum_to_100_percent(self):
        props = self._make_props([12_345, 67_890, 54_321])
        result = CostCalculator(CostBasis.AREA).distribute(props, 500_000)
        assert sum(s.share_percent for s in result.shares) == pytest.approx(100.0, rel=1e-9)

    def test_single_property_gets_full_cost(self):
        props = self._make_props([20_000])
        result = CostCalculator(CostBasis.AREA).distribute(props, 800_000)
        assert result.shares[0].share_percent == pytest.approx(100.0)
        assert result.shares[0].cost_nok == pytest.approx(800_000)

    def test_zero_cost_gives_zero_allocations(self):
        props = self._make_props([10_000, 20_000])
        result = CostCalculator(CostBasis.AREA).distribute(props, 0)
        for s in result.shares:
            assert s.cost_nok == pytest.approx(0.0)

    def test_negative_cost_propagates(self):
        props = self._make_props([50_000, 50_000])
        result = CostCalculator(CostBasis.AREA).distribute(props, -100_000)
        assert sum(s.cost_nok for s in result.shares) == pytest.approx(-100_000, rel=1e-9)
        assert result.warnings  # should warn about negative cost


# ---------------------------------------------------------------------------
# CostCalculator – volume basis
# ---------------------------------------------------------------------------

class TestVolumeBasis:

    def test_volume_basis_proportional(self):
        props = [
            PropertyRecord(1, area_m2=1000, volume_m3=600),
            PropertyRecord(2, area_m2=1000, volume_m3=400),
        ]
        result = CostCalculator(CostBasis.VOLUME).distribute(props, 1_000_000)
        shares = {s.property_id: s for s in result.shares}
        assert shares[1].share_percent == pytest.approx(60.0)
        assert shares[2].share_percent == pytest.approx(40.0)

    def test_zero_volumes_trigger_warning_and_equal_fallback(self):
        props = [
            PropertyRecord(1, area_m2=1000, volume_m3=0),
            PropertyRecord(2, area_m2=2000, volume_m3=0),
        ]
        result = CostCalculator(CostBasis.VOLUME).distribute(props, 200_000)
        assert result.warnings
        # equal fallback
        for s in result.shares:
            assert s.share_percent == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# CostCalculator – distance_area basis
# ---------------------------------------------------------------------------

class TestDistanceAreaBasis:

    def test_same_area_closer_property_gets_higher_share(self):
        props = [
            PropertyRecord(1, area_m2=10_000, haul_distance_m=100),
            PropertyRecord(2, area_m2=10_000, haul_distance_m=400),
        ]
        result = CostCalculator(CostBasis.DISTANCE_AREA).distribute(props, 1_000_000)
        shares = {s.property_id: s for s in result.shares}
        assert shares[1].share_fraction > shares[2].share_fraction

    def test_weights_use_sqrt_denominator(self):
        props = [
            PropertyRecord(1, area_m2=10_000, haul_distance_m=100),
            PropertyRecord(2, area_m2=10_000, haul_distance_m=400),
        ]
        # weight1 = 10000/sqrt(100) = 1000; weight2 = 10000/sqrt(400) = 500
        # share1 = 1000/1500 ≈ 66.67 %
        result = CostCalculator(CostBasis.DISTANCE_AREA).distribute(props, 1_000_000)
        shares = {s.property_id: s for s in result.shares}
        assert shares[1].share_percent == pytest.approx(200 / 3, rel=1e-4)
        assert shares[2].share_percent == pytest.approx(100 / 3, rel=1e-4)

    def test_total_cost_still_fully_allocated(self):
        props = [
            PropertyRecord(i, area_m2=10_000 * (i + 1), haul_distance_m=50 * (i + 1))
            for i in range(5)
        ]
        total = 2_500_000
        result = CostCalculator(CostBasis.DISTANCE_AREA).distribute(props, total)
        assert sum(s.cost_nok for s in result.shares) == pytest.approx(total, rel=1e-9)


# ---------------------------------------------------------------------------
# CostCalculator – edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_empty_properties_returns_warning(self):
        result = CostCalculator().distribute([], 500_000)
        assert result.shares == []
        assert result.warnings

    def test_result_has_correct_basis(self):
        props = [PropertyRecord(1, area_m2=1000)]
        result = CostCalculator(CostBasis.VOLUME).distribute(props, 100_000)
        assert result.basis == CostBasis.VOLUME

    def test_largest_share_identified(self):
        props = [
            PropertyRecord(1, area_m2=70_000),
            PropertyRecord(2, area_m2=30_000),
        ]
        result = CostCalculator(CostBasis.AREA).distribute(props, 100_000)
        assert result.largest_share.property_id == 1

    def test_smallest_share_identified(self):
        props = [
            PropertyRecord(1, area_m2=70_000),
            PropertyRecord(2, area_m2=30_000),
        ]
        result = CostCalculator(CostBasis.AREA).distribute(props, 100_000)
        assert result.smallest_share.property_id == 2

    def test_largest_and_smallest_none_for_empty(self):
        result = CostDistributionResult(total_cost_nok=0, basis=CostBasis.AREA)
        assert result.largest_share is None
        assert result.smallest_share is None

    def test_min_weight_threshold_adds_warnings(self):
        props = [
            PropertyRecord(1, area_m2=99_000),
            PropertyRecord(2, area_m2=1_000),
        ]
        calc = CostCalculator(CostBasis.AREA, min_weight_threshold=0.05)
        result = calc.distribute(props, 1_000_000)
        # Property 2 has 1% share, below 5% threshold
        assert any("2" in w for w in result.warnings)

    def test_string_property_ids_supported(self):
        props = [
            PropertyRecord("parcel-A", area_m2=50_000),
            PropertyRecord("parcel-B", area_m2=50_000),
        ]
        result = CostCalculator().distribute(props, 200_000)
        ids = {s.property_id for s in result.shares}
        assert ids == {"parcel-A", "parcel-B"}

    def test_cost_share_repr_contains_percent(self):
        s = CostShare(1, "Test", 100.0, 0.5, 50.0, 500_000)
        assert "50.00%" in repr(s)

    def test_total_shares_percent_sums_to_100(self):
        props = [PropertyRecord(i, area_m2=1000 * (i + 1)) for i in range(7)]
        result = CostCalculator(CostBasis.AREA).distribute(props, 1_000_000)
        assert result.total_shares_percent == pytest.approx(100.0, rel=1e-9)

    def test_cost_basis_enum_from_string(self):
        assert CostBasis("area") == CostBasis.AREA
        assert CostBasis("volume") == CostBasis.VOLUME
        assert CostBasis("distance_area") == CostBasis.DISTANCE_AREA
