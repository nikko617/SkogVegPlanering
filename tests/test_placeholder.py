# -*- coding: utf-8 -*-
"""
test_placeholder.py – Unit tests for SkogVegPlanering

STEG 6: Real tests for core validation logic.
All tests use plain Python (no QGIS runtime required).
"""

import math
import sys
import os

# Allow imports from the repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from utils.logger import setup_logger


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

def test_logger_import():
    """Test that logger can be imported without errors."""
    logger = setup_logger("test")
    assert logger is not None
    assert logger.name == "test"


# ---------------------------------------------------------------------------
# RoadValidator – slope checks
# ---------------------------------------------------------------------------

class TestSlopeCheck:

    def setup_method(self):
        from core.validator import RoadValidator
        self.v = RoadValidator(max_slope_percent=12.0, min_curve_radius=20.0)

    def test_flat_road_passes(self):
        r = self.v.check_slope_simple(1, dz=0, dx=100)
        assert r.passed
        assert r.value == pytest.approx(0.0)

    def test_acceptable_slope_passes(self):
        # 10 m rise over 100 m horizontal = 10 % → passes (≤ 12 %)
        r = self.v.check_slope_simple(1, dz=10, dx=100)
        assert r.passed
        assert r.value == pytest.approx(10.0)

    def test_exact_limit_passes(self):
        # Exactly 12 % → should pass
        r = self.v.check_slope_simple(1, dz=12, dx=100)
        assert r.passed

    def test_steep_slope_fails(self):
        # 15 % → fails
        r = self.v.check_slope_simple(1, dz=15, dx=100)
        assert not r.passed
        assert r.value == pytest.approx(15.0)

    def test_negative_dz_uses_abs(self):
        # Downhill -10 m over 100 m = 10 % → passes
        r = self.v.check_slope_simple(1, dz=-10, dx=100)
        assert r.passed
        assert r.value == pytest.approx(10.0)

    def test_zero_dx_fails_gracefully(self):
        r = self.v.check_slope_simple(1, dz=5, dx=0)
        assert not r.passed
        assert r.value is None

    def test_3d_polyline_slope(self):
        # Three points: flat then steep segment
        points = [(0, 0, 0), (100, 0, 10), (200, 0, 25)]  # 10 %, 15 %
        results = self.v.check_slope_points(99, points)
        assert len(results) == 2
        assert results[0].passed    # 10 %
        assert not results[1].passed  # 15 %

    def test_custom_max_slope(self):
        v = type(self.v).__new__(type(self.v))
        from core.validator import RoadValidator
        v = RoadValidator(max_slope_percent=8.0, min_curve_radius=20.0)
        assert not v.check_slope_simple(1, dz=9, dx=100).passed
        assert v.check_slope_simple(1, dz=7, dx=100).passed


# ---------------------------------------------------------------------------
# RoadValidator – curve-radius checks
# ---------------------------------------------------------------------------

class TestCurveRadiusCheck:

    def setup_method(self):
        from core.validator import RoadValidator
        self.v = RoadValidator(max_slope_percent=12.0, min_curve_radius=20.0)

    def test_large_radius_passes(self):
        # Near-straight line → very large radius
        points = [(0, 0), (50, 0.1), (100, 0)]
        results = self.v.check_curve_radius_points(1, points)
        assert results, "Expected at least one result"
        assert all(r.passed for r in results)

    def test_tight_curve_fails(self):
        # Sharp 90-degree turn at (10, 0) → (10, 10) → radius ≈ 7 m
        points = [(0, 0), (10, 0), (10, 10)]
        results = self.v.check_curve_radius_points(1, points)
        assert results
        assert not results[0].passed

    def test_exact_min_radius_passes(self):
        # Build a circular arc slightly above the 20 m minimum
        r = 21.0
        points = [
            (r * math.cos(a), r * math.sin(a))
            for a in [0, math.pi / 6, math.pi / 3]
        ]
        results = self.v.check_curve_radius_points(1, points)
        assert results
        assert results[0].passed

    def test_collinear_points_skipped(self):
        # Collinear → area = 0 → circumradius returns None → skipped
        points = [(0, 0), (5, 0), (10, 0)]
        results = self.v.check_curve_radius_points(1, points)
        assert len(results) == 0

    def test_two_points_no_results(self):
        points = [(0, 0), (100, 0)]
        results = self.v.check_curve_radius_points(1, points)
        assert len(results) == 0


# ---------------------------------------------------------------------------
# RoadValidator – road-class checks
# ---------------------------------------------------------------------------

class TestRoadClassCheck:

    def setup_method(self):
        from core.validator import RoadValidator
        self.v = RoadValidator()

    def test_class1_valid(self):
        results = self.v.check_road_class(1, road_class=1, width_m=4.0, bearing_tons=12)
        assert all(r.passed for r in results)

    def test_class1_narrow_fails(self):
        results = self.v.check_road_class(1, road_class=1, width_m=3.0, bearing_tons=12)
        width_result = next(r for r in results if r.check_type == "road_width")
        assert not width_result.passed

    def test_class3_minimum_passes(self):
        results = self.v.check_road_class(1, road_class=3, width_m=2.5, bearing_tons=6)
        assert all(r.passed for r in results)

    def test_unknown_class_fails(self):
        results = self.v.check_road_class(1, road_class=99)
        assert len(results) == 1
        assert not results[0].passed
        assert "99" in results[0].message

    def test_only_width_checked_if_bearing_none(self):
        results = self.v.check_road_class(1, road_class=2, width_m=2.0)
        types = [r.check_type for r in results]
        assert "road_width" in types
        assert "bearing_capacity" not in types

    def test_only_bearing_checked_if_width_none(self):
        results = self.v.check_road_class(1, road_class=2, bearing_tons=5)
        types = [r.check_type for r in results]
        assert "bearing_capacity" in types
        assert "road_width" not in types


# ---------------------------------------------------------------------------
# ValidationResult repr
# ---------------------------------------------------------------------------

def test_validation_result_repr_passed():
    from core.validator import ValidationResult
    r = ValidationResult(1, "slope", True, "OK", 5.0)
    assert "✓" in repr(r)

def test_validation_result_repr_failed():
    from core.validator import ValidationResult
    r = ValidationResult(2, "curve_radius", False, "For liten radius", 8.0)
    assert "✗" in repr(r)

