# -*- coding: utf-8 -*-
"""
test_feature_editor.py – Unit tests for FeatureEditor (STEG 6)

All tests use plain Python – no QGIS runtime required.
"""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from processors.feature_editor import (
    FeatureEditor,
    FeatureType,
    RoadFeature,
    StationFeature,
    DumpSiteFeature,
    _polyline_length,
)


# ---------------------------------------------------------------------------
# Geometry helper
# ---------------------------------------------------------------------------

class TestPolylineLength:

    def test_empty(self):
        assert _polyline_length([]) == 0.0

    def test_single_point(self):
        assert _polyline_length([(0, 0)]) == 0.0

    def test_horizontal(self):
        assert _polyline_length([(0, 0), (100, 0)]) == pytest.approx(100.0)

    def test_right_triangle(self):
        # 3-4-5 triangle
        assert _polyline_length([(0, 0), (3, 0), (3, 4)]) == pytest.approx(7.0)

    def test_diagonal(self):
        assert _polyline_length([(0, 0), (3, 4)]) == pytest.approx(5.0)

    def test_three_d_ignored(self):
        # Z is ignored in length
        assert _polyline_length([(0, 0, 0), (3, 4, 100)]) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# RoadFeature
# ---------------------------------------------------------------------------

class TestRoadFeature:

    def test_length_computed_on_init(self):
        r = RoadFeature(1, "A", [(0, 0), (0, 100)])
        assert r.length_m == pytest.approx(100.0)

    def test_repr(self):
        r = RoadFeature(2, "B", [(0, 0), (50, 0)])
        assert "2" in repr(r) and "B" in repr(r) and "50" in repr(r)

    def test_empty_vertices(self):
        r = RoadFeature(1, "A", [])
        assert r.length_m == 0.0


# ---------------------------------------------------------------------------
# StationFeature
# ---------------------------------------------------------------------------

class TestStationFeature:

    def test_xy_property(self):
        s = StationFeature(1, "S1", 10.0, 20.0, z=5.0)
        assert s.xy == (10.0, 20.0)

    def test_repr(self):
        s = StationFeature(3, "Alpha", 1.0, 2.0)
        assert "Alpha" in repr(s)


# ---------------------------------------------------------------------------
# DumpSiteFeature
# ---------------------------------------------------------------------------

class TestDumpSiteFeature:

    def test_xy_property(self):
        d = DumpSiteFeature(1, "V1", 5.0, 10.0, area_m2=200.0)
        assert d.xy == (5.0, 10.0)

    def test_repr(self):
        d = DumpSiteFeature(2, "Velte B", 0, 0, area_m2=500)
        assert "500" in repr(d)


# ---------------------------------------------------------------------------
# FeatureEditor – Roads
# ---------------------------------------------------------------------------

class TestEditorRoads:

    def test_add_road_returns_feature(self):
        ed = FeatureEditor()
        r = ed.add_road("Veg A", [(0, 0), (100, 0)])
        assert isinstance(r, RoadFeature)
        assert r.name == "Veg A"
        assert r.length_m == pytest.approx(100.0)

    def test_get_road(self):
        ed = FeatureEditor()
        r = ed.add_road("X", [(0, 0), (50, 0)])
        assert ed.get_road(r.fid) is not None

    def test_all_roads_sorted_by_fid(self):
        ed = FeatureEditor()
        ed.add_road("C", [(0, 0), (1, 0)])
        ed.add_road("A", [(0, 0), (1, 0)])
        fids = [r.fid for r in ed.all_roads()]
        assert fids == sorted(fids)

    def test_update_road_name(self):
        ed = FeatureEditor()
        r = ed.add_road("Old", [(0, 0), (1, 0)])
        ed.update_road(r.fid, name="New")
        assert ed.get_road(r.fid).name == "New"

    def test_update_road_vertices_recalculates_length(self):
        ed = FeatureEditor()
        r = ed.add_road("Veg", [(0, 0), (100, 0)])
        ed.update_road(r.fid, vertices=[(0, 0), (0, 200)])
        assert ed.get_road(r.fid).length_m == pytest.approx(200.0)

    def test_delete_road(self):
        ed = FeatureEditor()
        r = ed.add_road("X", [(0, 0), (1, 0)])
        ed.delete_road(r.fid)
        assert ed.get_road(r.fid) is None

    def test_delete_nonexistent_road_no_error(self):
        ed = FeatureEditor()
        ed.delete_road(999)   # should not raise

    def test_update_nonexistent_road_raises(self):
        ed = FeatureEditor()
        with pytest.raises(KeyError):
            ed.update_road(999, name="X")

    def test_update_invalid_field_raises(self):
        ed = FeatureEditor()
        r = ed.add_road("V", [(0, 0), (1, 0)])
        with pytest.raises(AttributeError):
            ed.update_road(r.fid, nonexistent_field="value")

    def test_fids_are_unique_across_adds(self):
        ed = FeatureEditor()
        fids = [ed.add_road(f"R{i}", [(0, 0), (i, 0)]).fid for i in range(1, 6)]
        assert len(set(fids)) == 5


# ---------------------------------------------------------------------------
# FeatureEditor – Stations
# ---------------------------------------------------------------------------

class TestEditorStations:

    def test_add_station(self):
        ed = FeatureEditor()
        s = ed.add_station("S1", 1.0, 2.0, z=50.0, capacity_t=10.0)
        assert s.x == 1.0 and s.y == 2.0 and s.z == 50.0

    def test_update_station_position(self):
        ed = FeatureEditor()
        s = ed.add_station("S", 0, 0)
        ed.update_station(s.fid, x=99.0, y=88.0)
        updated = ed.get_station(s.fid)
        assert updated.x == pytest.approx(99.0)
        assert updated.y == pytest.approx(88.0)

    def test_delete_station(self):
        ed = FeatureEditor()
        s = ed.add_station("S", 0, 0)
        ed.delete_station(s.fid)
        assert ed.get_station(s.fid) is None

    def test_all_stations_empty_initially(self):
        assert FeatureEditor().all_stations() == []


# ---------------------------------------------------------------------------
# FeatureEditor – Dump sites
# ---------------------------------------------------------------------------

class TestEditorDumpSites:

    def test_add_dump_site(self):
        ed = FeatureEditor()
        d = ed.add_dump_site("V1", 5.0, 10.0, area_m2=300.0)
        assert d.area_m2 == pytest.approx(300.0)

    def test_update_dump_area(self):
        ed = FeatureEditor()
        d = ed.add_dump_site("V", 0, 0)
        ed.update_dump_site(d.fid, area_m2=500.0)
        assert ed.get_dump_site(d.fid).area_m2 == pytest.approx(500.0)

    def test_delete_dump_site(self):
        ed = FeatureEditor()
        d = ed.add_dump_site("V", 0, 0)
        ed.delete_dump_site(d.fid)
        assert ed.get_dump_site(d.fid) is None


# ---------------------------------------------------------------------------
# Undo / Redo
# ---------------------------------------------------------------------------

class TestUndoRedo:

    def test_undo_add_road(self):
        ed = FeatureEditor()
        r = ed.add_road("A", [(0, 0), (1, 0)])
        assert ed.can_undo
        ed.undo()
        assert ed.get_road(r.fid) is None

    def test_undo_description_returned(self):
        ed = FeatureEditor()
        ed.add_road("A", [(0, 0), (1, 0)])
        desc = ed.undo()
        assert desc is not None and "road" in desc.lower()

    def test_undo_empty_returns_none(self):
        ed = FeatureEditor()
        assert ed.undo() is None

    def test_redo_after_undo(self):
        ed = FeatureEditor()
        r = ed.add_road("B", [(0, 0), (1, 0)])
        fid = r.fid
        ed.undo()
        assert ed.get_road(fid) is None
        ed.redo()
        assert ed.get_road(fid) is not None

    def test_redo_empty_returns_none(self):
        ed = FeatureEditor()
        assert ed.redo() is None

    def test_new_action_clears_redo(self):
        ed = FeatureEditor()
        r = ed.add_road("A", [(0, 0), (1, 0)])
        ed.undo()
        assert ed.can_redo
        ed.add_road("C", [(0, 0), (1, 0)])
        assert not ed.can_redo

    def test_undo_update(self):
        ed = FeatureEditor()
        r = ed.add_road("Original", [(0, 0), (1, 0)])
        ed.update_road(r.fid, name="Modified")
        assert ed.get_road(r.fid).name == "Modified"
        ed.undo()
        assert ed.get_road(r.fid).name == "Original"   # undo the update
        ed.undo()
        assert ed.get_road(r.fid) is None              # undo the add

    def test_undo_delete(self):
        ed = FeatureEditor()
        r = ed.add_road("D", [(0, 0), (1, 0)])
        fid = r.fid
        ed.delete_road(fid)
        assert ed.get_road(fid) is None
        ed.undo()
        assert ed.get_road(fid) is not None

    def test_max_undo_limit(self):
        ed = FeatureEditor(max_undo=3)
        for i in range(5):
            ed.add_road(f"R{i}", [(0, 0), (1, 0)])
        assert len(ed._undo_stack) <= 3

    def test_undo_redo_descriptions(self):
        ed = FeatureEditor()
        ed.add_station("S", 0, 0)
        assert ed.undo_description is not None
        ed.undo()
        assert ed.redo_description is not None

    def test_can_undo_redo_properties(self):
        ed = FeatureEditor()
        assert not ed.can_undo
        assert not ed.can_redo
        ed.add_dump_site("V", 0, 0)
        assert ed.can_undo
        ed.undo()
        assert ed.can_redo


# ---------------------------------------------------------------------------
# Bulk load / export
# ---------------------------------------------------------------------------

class TestBulkLoadExport:

    def test_load_roads(self):
        ed = FeatureEditor()
        ed.load_roads([
            {"name": "Veg A", "vertices": [(0, 0), (100, 0)]},
            {"name": "Veg B", "vertices": [(0, 0), (0, 200)]},
        ])
        assert len(ed.all_roads()) == 2

    def test_load_roads_clears_existing(self):
        ed = FeatureEditor()
        ed.add_road("Old", [(0, 0), (1, 0)])
        ed.load_roads([{"name": "New", "vertices": [(0, 0), (1, 0)]}])
        names = [r.name for r in ed.all_roads()]
        assert names == ["New"]

    def test_load_stations(self):
        ed = FeatureEditor()
        ed.load_stations([
            {"name": "S1", "x": 1.0, "y": 2.0},
            {"name": "S2", "x": 3.0, "y": 4.0},
        ])
        assert len(ed.all_stations()) == 2

    def test_load_dump_sites(self):
        ed = FeatureEditor()
        ed.load_dump_sites([{"name": "V1", "x": 0, "y": 0, "area_m2": 100}])
        assert ed.all_dump_sites()[0].area_m2 == pytest.approx(100.0)

    def test_export_roads_round_trip(self):
        ed = FeatureEditor()
        ed.add_road("Export Road", [(0, 0), (100, 0)], road_class="kl.4", notes="test")
        exported = ed.export_roads()
        assert len(exported) == 1
        assert exported[0]["name"] == "Export Road"
        assert exported[0]["road_class"] == "kl.4"
        assert exported[0]["length_m"] == pytest.approx(100.0, abs=0.01)

    def test_export_stations(self):
        ed = FeatureEditor()
        ed.add_station("S", 5.0, 10.0, z=20.0, capacity_t=5.0)
        exp = ed.export_stations()
        assert exp[0]["x"] == 5.0 and exp[0]["z"] == 20.0

    def test_export_dump_sites(self):
        ed = FeatureEditor()
        ed.add_dump_site("V", 1.0, 2.0, area_m2=250.0)
        exp = ed.export_dump_sites()
        assert exp[0]["area_m2"] == pytest.approx(250.0)

    def test_load_roads_missing_vertices_defaults_to_empty(self):
        ed = FeatureEditor()
        ed.load_roads([{"name": "No vertices"}])
        road = ed.all_roads()[0]
        assert road.vertices == []
        assert road.length_m == 0.0
