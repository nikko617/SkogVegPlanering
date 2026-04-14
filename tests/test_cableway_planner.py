# -*- coding: utf-8 -*-
"""
test_cableway_planner.py – Unit tests for CablewayPlanner (STEG 5)

All tests use plain Python – no QGIS runtime required.
"""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from processors.cableway_planner import (
    CablewayPlanner,
    CablewayPlan,
    CablewayStation,
    CablewaySegment,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def straight_road_2d(length_m, n_vertices=2):
    """Return a horizontal 2-D polyline from (0,0) to (length_m,0)."""
    step = length_m / (n_vertices - 1)
    return [(i * step, 0.0) for i in range(n_vertices)]


def straight_road_3d(length_m, n_vertices=2, dz=0.0):
    """Return a horizontal 3-D polyline with constant slope."""
    step = length_m / (n_vertices - 1)
    dz_step = dz / (n_vertices - 1) if n_vertices > 1 else 0
    return [(i * step, 0.0, 100.0 + i * dz_step) for i in range(n_vertices)]


# ---------------------------------------------------------------------------
# CablewayPlanner – parameter validation
# ---------------------------------------------------------------------------

class TestPlannerParams:

    def test_invalid_station_interval_raises(self):
        with pytest.raises(ValueError, match="station_interval_m"):
            CablewayPlanner(station_interval_m=0)

    def test_negative_interval_raises(self):
        with pytest.raises(ValueError, match="station_interval_m"):
            CablewayPlanner(station_interval_m=-100)

    def test_invalid_winch_length_raises(self):
        with pytest.raises(ValueError, match="winch_length_m"):
            CablewayPlanner(winch_length_m=0)

    def test_invalid_max_span_raises(self):
        with pytest.raises(ValueError, match="max_span_m"):
            CablewayPlanner(max_span_m=-1)

    def test_negative_min_slope_raises(self):
        with pytest.raises(ValueError, match="min_slope_pct"):
            CablewayPlanner(min_slope_pct=-5)

    def test_valid_params_accepted(self):
        p = CablewayPlanner(station_interval_m=150, winch_length_m=350,
                            max_span_m=400, min_slope_pct=10)
        assert p.station_interval_m == 150
        assert p.winch_length_m == 350


# ---------------------------------------------------------------------------
# CablewayPlanner – station placement
# ---------------------------------------------------------------------------

class TestStationPlacement:

    def test_single_interval_two_stations(self):
        """200 m road with 200 m interval -> stations at 0 and 200 m."""
        pts = straight_road_2d(200)
        plan = CablewayPlanner(station_interval_m=200).plan(pts)
        dists = [s.distance_along_road_m for s in plan.stations]
        assert dists[0] == pytest.approx(0.0)
        assert dists[-1] == pytest.approx(200.0)

    def test_correct_number_of_stations(self):
        """600 m road with 200 m interval -> 4 stations (0, 200, 400, 600)."""
        pts = straight_road_2d(600, n_vertices=7)
        plan = CablewayPlanner(station_interval_m=200).plan(pts)
        dists = [s.distance_along_road_m for s in plan.stations]
        assert dists == pytest.approx([0, 200, 400, 600], abs=0.5)

    def test_station_ids_are_sequential(self):
        pts = straight_road_2d(500, n_vertices=6)
        plan = CablewayPlanner(station_interval_m=100).plan(pts)
        ids = [s.station_id for s in plan.stations]
        assert ids == list(range(1, len(ids) + 1))

    def test_last_station_at_road_end(self):
        """When road length is not a multiple of interval, last station is at end."""
        pts = straight_road_2d(550, n_vertices=12)
        plan = CablewayPlanner(station_interval_m=200).plan(pts)
        assert plan.stations[-1].distance_along_road_m == pytest.approx(550, abs=1.0)

    def test_coverage_radius_equals_winch_length(self):
        pts = straight_road_2d(400, n_vertices=5)
        plan = CablewayPlanner(winch_length_m=350).plan(pts)
        for s in plan.stations:
            assert s.coverage_radius_m == pytest.approx(350)

    def test_station_xy_on_straight_road(self):
        """Stations on a straight horizontal road should all have y=0."""
        pts = straight_road_2d(800, n_vertices=9)
        plan = CablewayPlanner(station_interval_m=200).plan(pts)
        for s in plan.stations:
            assert s.y == pytest.approx(0.0, abs=1e-6)

    def test_polyline_id_propagated(self):
        pts = straight_road_2d(200)
        plan = CablewayPlanner().plan(pts, polyline_id=42)
        assert plan.polyline_id == 42

    def test_too_few_points_returns_warning(self):
        plan = CablewayPlanner().plan([(0, 0)], polyline_id=1)
        assert plan.stations == []
        assert plan.warnings


# ---------------------------------------------------------------------------
# CablewayPlanner – 3-D slope
# ---------------------------------------------------------------------------

class TestSlopePlanning:

    def test_z_values_set_on_stations(self):
        pts = straight_road_3d(400, n_vertices=5, dz=-40)  # drops 40 m over 400 m
        plan = CablewayPlanner(station_interval_m=100).plan(pts)
        # First station at z=100, last at z~60
        assert plan.stations[0].z == pytest.approx(100.0, abs=0.5)
        assert plan.stations[-1].z == pytest.approx(60.0, abs=1.0)

    def test_slope_computed_correctly(self):
        """40 m drop over 400 m horizontal -> 10% slope."""
        pts = straight_road_3d(400, n_vertices=5, dz=-40)
        plan = CablewayPlanner(station_interval_m=100, min_slope_pct=0).plan(pts)
        for s in plan.stations[1:]:  # skip first (boundary)
            assert s.slope_pct == pytest.approx(10.0, abs=0.5)

    def test_gentle_slope_triggers_warning(self):
        """5% slope < 15% min -> warning expected."""
        pts = straight_road_3d(200, n_vertices=3, dz=-10)  # 5% slope
        plan = CablewayPlanner(station_interval_m=100, min_slope_pct=15).plan(pts)
        assert any("kanskje" in w or "under minimum" in w for w in plan.warnings)

    def test_steep_slope_no_warning(self):
        """30% slope > 15% min -> no slope warning."""
        pts = straight_road_3d(200, n_vertices=3, dz=-60)  # 30% slope
        plan = CablewayPlanner(station_interval_m=100, min_slope_pct=15).plan(pts)
        slope_warnings = [w for w in plan.warnings if "kanskje" in w or "under minimum" in w]
        assert slope_warnings == []


# ---------------------------------------------------------------------------
# CablewayPlanner – segments
# ---------------------------------------------------------------------------

class TestSegments:

    def test_segment_count(self):
        """n stations -> n-1 segments."""
        pts = straight_road_2d(600, n_vertices=7)
        plan = CablewayPlanner(station_interval_m=200).plan(pts)
        assert len(plan.segments) == len(plan.stations) - 1

    def test_segment_span_equals_interval_on_straight_road(self):
        pts = straight_road_2d(600, n_vertices=7)
        plan = CablewayPlanner(station_interval_m=200).plan(pts)
        for seg in plan.segments[:-1]:   # last may differ at road end
            assert seg.span_m == pytest.approx(200.0, abs=2.0)

    def test_excessive_span_flagged(self):
        """Span > max_span_m should set exceeds_max_span and add warning."""
        pts = straight_road_2d(1000, n_vertices=2)
        plan = CablewayPlanner(
            station_interval_m=1000, max_span_m=500
        ).plan(pts)
        # Only one segment from 0 to 1000 m
        assert plan.segments[0].exceeds_max_span is True
        assert any("tarn" in w or "maks" in w for w in plan.warnings)

    def test_normal_span_not_flagged(self):
        pts = straight_road_2d(400, n_vertices=5)
        plan = CablewayPlanner(station_interval_m=100, max_span_m=500).plan(pts)
        for seg in plan.segments:
            assert seg.exceeds_max_span is False

    def test_segment_midpoint(self):
        pts = [(0, 0), (200, 0)]
        plan = CablewayPlanner(station_interval_m=200).plan(pts)
        mid = plan.segments[0].midpoint
        assert mid == pytest.approx((100.0, 0.0), abs=1.0)

    def test_segment_repr_contains_span(self):
        pts = straight_road_2d(200)
        plan = CablewayPlanner(station_interval_m=200).plan(pts)
        assert "200" in repr(plan.segments[0]) or "m" in repr(plan.segments[0])


# ---------------------------------------------------------------------------
# CablewayPlan properties
# ---------------------------------------------------------------------------

class TestCablewayPlanProperties:

    def test_total_road_length(self):
        pts = straight_road_2d(800, n_vertices=9)
        plan = CablewayPlanner(station_interval_m=200).plan(pts)
        assert plan.total_road_length_m == pytest.approx(800.0, abs=1.0)

    def test_zero_length_road_warning(self):
        pts = [(0, 0), (0, 0)]  # zero length
        plan = CablewayPlanner().plan(pts)
        assert plan.warnings

    def test_coverage_overlap_positive_when_overlap(self):
        p = CablewayPlanner(station_interval_m=200, winch_length_m=400)
        # overlap = 2*400 - 200 = 600 m
        plan = CablewayPlan(
            polyline_id=0,
            station_interval_m=200,
            winch_length_m=400,
            max_span_m=500,
        )
        assert plan.coverage_overlap_m == pytest.approx(600.0)

    def test_coverage_gap_warning_when_interval_exceeds_2x_winch(self):
        pts = straight_road_2d(1000, n_vertices=3)
        # interval 900 > 2*400=800 -> gap
        plan = CablewayPlanner(station_interval_m=900, winch_length_m=400).plan(pts)
        assert any("udekket" in w for w in plan.warnings)

    def test_plan_repr(self):
        pts = straight_road_2d(400)
        plan = CablewayPlanner(station_interval_m=200).plan(pts, polyline_id=7)
        r = repr(plan)
        assert "7" in r
        assert "station" in r

    def test_station_repr(self):
        s = CablewayStation(3, 100.0, 200.0, 50.0, 300.0, 400.0, 12.5)
        r = repr(s)
        assert "3" in r
        assert "300" in r


# ---------------------------------------------------------------------------
# Interpolation edge cases
# ---------------------------------------------------------------------------

class TestInterpolation:

    def test_station_at_exact_vertex(self):
        """Station placed at a vertex should match the vertex coordinates."""
        pts = [(0, 0), (100, 0), (200, 0)]
        plan = CablewayPlanner(station_interval_m=100).plan(pts)
        s1 = plan.stations[1]
        assert s1.x == pytest.approx(100.0, abs=0.01)
        assert s1.y == pytest.approx(0.0, abs=0.01)

    def test_station_between_vertices(self):
        """Station placed halfway between two vertices should be at midpoint."""
        pts = [(0, 0), (200, 0)]
        plan = CablewayPlanner(station_interval_m=100).plan(pts)
        mid_station = plan.stations[1]
        assert mid_station.x == pytest.approx(100.0, abs=0.5)

    def test_diagonal_road_station_position(self):
        """Test interpolation on a diagonal road."""
        pts = [(0, 0), (300, 400)]   # length = 500
        plan = CablewayPlanner(station_interval_m=250).plan(pts)
        # Station at 250 m = halfway -> (150, 200)
        s_mid = plan.stations[1]
        assert s_mid.x == pytest.approx(150.0, abs=1.0)
        assert s_mid.y == pytest.approx(200.0, abs=1.0)
