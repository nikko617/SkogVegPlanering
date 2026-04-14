# -*- coding: utf-8 -*-
"""
feature_editor.py – Interactive feature editing model for SkogVegPlanering (STEG 6)

Provides a lightweight, QGIS-free model for editing three feature types used
in Norwegian forest-road planning:

  * **Road** (veg) – a named line feature (centre-line of a forest road).
  * **Station** (standplass) – a named point feature (cable-way anchor/landing).
  * **DumpSite** (velteplass) – a named point feature (timber stacking area).

The editing model supports:
  * Add / update / delete features.
  * Rename features.
  * Full undo / redo via a command stack.
  * Import from / export to plain Python lists of dicts so that QGIS
    layer integration can live exclusively in ``ui/editor_dialog.py``.

All geometry is stored as plain ``(x, y)`` tuples; Z is optional.
No external dependencies beyond the Python standard library.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Feature types
# ---------------------------------------------------------------------------

class FeatureType(str, Enum):
    ROAD      = "road"
    STATION   = "station"
    DUMP_SITE = "dump_site"


@dataclass
class RoadFeature:
    """
    A named forest-road centre-line.

    Attributes
    ----------
    fid : int
        Local feature ID (auto-assigned by the editor).
    name : str
        Human-readable name, e.g. "Skogsveg nord".
    vertices : list of (x, y) or (x, y, z) tuples
        Ordered vertices of the road polyline.
    road_class : str
        Road class string, e.g. "skogsbilveg kl.3".
    length_m : float
        Cached total length (metres).  Updated by the editor on every change.
    notes : str
        Free-text notes.
    """
    fid: int
    name: str
    vertices: List[Tuple]
    road_class: str = "skogsbilveg kl.3"
    length_m: float = 0.0
    notes: str = ""

    def __post_init__(self):
        self.length_m = _polyline_length(self.vertices)

    def __repr__(self):
        return f"RoadFeature({self.fid}, {self.name!r}, {self.length_m:.0f}m)"


@dataclass
class StationFeature:
    """
    A cable-way standplass (anchor / landing point).

    Attributes
    ----------
    fid : int
    name : str
    x, y : float
        Map coordinates.
    z : float
        Elevation (m).  0.0 if unknown.
    capacity_t : float
        Payload capacity in tonnes.
    notes : str
    """
    fid: int
    name: str
    x: float
    y: float
    z: float = 0.0
    capacity_t: float = 0.0
    notes: str = ""

    @property
    def xy(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def __repr__(self):
        return f"StationFeature({self.fid}, {self.name!r}, ({self.x:.1f},{self.y:.1f}))"


@dataclass
class DumpSiteFeature:
    """
    A timber dump / stacking site (velteplass).

    Attributes
    ----------
    fid : int
    name : str
    x, y : float
        Map coordinates.
    area_m2 : float
        Surface area of the dump site in square metres.
    notes : str
    """
    fid: int
    name: str
    x: float
    y: float
    area_m2: float = 0.0
    notes: str = ""

    @property
    def xy(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def __repr__(self):
        return f"DumpSiteFeature({self.fid}, {self.name!r}, area={self.area_m2:.0f}m2)"


# ---------------------------------------------------------------------------
# Undo / Redo command base
# ---------------------------------------------------------------------------

class _EditCommand:
    """Abstract base for reversible editor commands."""

    def apply(self, state: "_EditorState") -> None:
        raise NotImplementedError

    def undo(self, state: "_EditorState") -> None:
        raise NotImplementedError

    def description(self) -> str:
        return self.__class__.__name__


# ---------------------------------------------------------------------------
# Concrete commands
# ---------------------------------------------------------------------------

class _AddFeature(_EditCommand):
    def __init__(self, ftype: FeatureType, feature):
        self._ftype   = ftype
        self._feature = feature

    def apply(self, state):
        state.get_store(self._ftype)[self._feature.fid] = copy.deepcopy(self._feature)

    def undo(self, state):
        state.get_store(self._ftype).pop(self._feature.fid, None)

    def description(self):
        return f"Add {self._ftype.value} '{self._feature.name}' (fid={self._feature.fid})"


class _DeleteFeature(_EditCommand):
    def __init__(self, ftype: FeatureType, fid: int):
        self._ftype   = ftype
        self._fid     = fid
        self._backup  = None

    def apply(self, state):
        self._backup = copy.deepcopy(state.get_store(self._ftype).get(self._fid))
        state.get_store(self._ftype).pop(self._fid, None)

    def undo(self, state):
        if self._backup is not None:
            state.get_store(self._ftype)[self._fid] = copy.deepcopy(self._backup)

    def description(self):
        return f"Delete {self._ftype.value} fid={self._fid}"


class _UpdateFeature(_EditCommand):
    def __init__(self, ftype: FeatureType, fid: int, **kwargs):
        self._ftype  = ftype
        self._fid    = fid
        self._kwargs = kwargs
        self._backup: Optional[Dict] = None

    def apply(self, state):
        feat = state.get_store(self._ftype).get(self._fid)
        if feat is None:
            raise KeyError(f"{self._ftype.value} fid={self._fid} not found")
        self._backup = copy.deepcopy(feat.__dict__)
        for k, v in self._kwargs.items():
            if not hasattr(feat, k):
                raise AttributeError(f"{feat.__class__.__name__} has no field '{k}'")
            setattr(feat, k, v)
        # Recompute cached length for roads
        if isinstance(feat, RoadFeature):
            feat.length_m = _polyline_length(feat.vertices)

    def undo(self, state):
        if self._backup is not None:
            feat = state.get_store(self._ftype).get(self._fid)
            if feat is not None:
                for k, v in self._backup.items():
                    setattr(feat, k, v)

    def description(self):
        keys = ", ".join(self._kwargs)
        return f"Update {self._ftype.value} fid={self._fid} [{keys}]"


# ---------------------------------------------------------------------------
# Editor state (mutable snapshot)
# ---------------------------------------------------------------------------

class _EditorState:
    """
    Holds the three feature stores (dicts keyed by fid).
    Commands mutate this object.
    """

    def __init__(self):
        self.roads:      Dict[int, RoadFeature]     = {}
        self.stations:   Dict[int, StationFeature]  = {}
        self.dump_sites: Dict[int, DumpSiteFeature] = {}

    def get_store(self, ftype: FeatureType) -> Dict:
        if ftype == FeatureType.ROAD:
            return self.roads
        if ftype == FeatureType.STATION:
            return self.stations
        if ftype == FeatureType.DUMP_SITE:
            return self.dump_sites
        raise ValueError(f"Unknown FeatureType: {ftype!r}")

    def clone(self) -> "_EditorState":
        new = _EditorState()
        new.roads      = copy.deepcopy(self.roads)
        new.stations   = copy.deepcopy(self.stations)
        new.dump_sites = copy.deepcopy(self.dump_sites)
        return new


# ---------------------------------------------------------------------------
# Public editor API
# ---------------------------------------------------------------------------

class FeatureEditor:
    """
    Interactive editor for roads, stations, and dump sites.

    All mutating operations go through ``apply(command)`` which also pushes
    onto the undo stack.  ``undo()`` / ``redo()`` walk the stack.

    Parameters
    ----------
    max_undo : int
        Maximum number of undo steps retained (default 50).

    Examples
    --------
    >>> ed = FeatureEditor()
    >>> road = ed.add_road("Skogsveg A", [(0,0),(100,0),(200,50)])
    >>> road.length_m
    223...
    >>> ed.update_road(road.fid, name="Skogsveg B")
    >>> ed.undo()
    >>> ed.get_road(road.fid).name
    'Skogsveg A'
    """

    def __init__(self, max_undo: int = 50):
        self._state     = _EditorState()
        self._undo_stack: List[_EditCommand] = []
        self._redo_stack: List[_EditCommand] = []
        self._max_undo  = max_undo
        self._next_fid: Dict[FeatureType, int] = {
            FeatureType.ROAD:      1,
            FeatureType.STATION:   1,
            FeatureType.DUMP_SITE: 1,
        }

    # ------------------------------------------------------------------
    # Road operations
    # ------------------------------------------------------------------

    def add_road(
        self,
        name: str,
        vertices: Sequence[Tuple],
        road_class: str = "skogsbilveg kl.3",
        notes: str = "",
    ) -> RoadFeature:
        """Add a new road and return the created feature."""
        fid  = self._alloc_fid(FeatureType.ROAD)
        feat = RoadFeature(fid=fid, name=name, vertices=list(vertices),
                           road_class=road_class, notes=notes)
        self._apply(_AddFeature(FeatureType.ROAD, feat))
        return self._state.roads[fid]

    def update_road(self, fid: int, **kwargs) -> None:
        """Update one or more fields of a road feature."""
        self._apply(_UpdateFeature(FeatureType.ROAD, fid, **kwargs))

    def delete_road(self, fid: int) -> None:
        """Delete a road by fid."""
        self._apply(_DeleteFeature(FeatureType.ROAD, fid))

    def get_road(self, fid: int) -> Optional[RoadFeature]:
        return self._state.roads.get(fid)

    def all_roads(self) -> List[RoadFeature]:
        return sorted(self._state.roads.values(), key=lambda r: r.fid)

    # ------------------------------------------------------------------
    # Station operations
    # ------------------------------------------------------------------

    def add_station(
        self,
        name: str,
        x: float,
        y: float,
        z: float = 0.0,
        capacity_t: float = 0.0,
        notes: str = "",
    ) -> StationFeature:
        fid  = self._alloc_fid(FeatureType.STATION)
        feat = StationFeature(fid=fid, name=name, x=x, y=y, z=z,
                              capacity_t=capacity_t, notes=notes)
        self._apply(_AddFeature(FeatureType.STATION, feat))
        return self._state.stations[fid]

    def update_station(self, fid: int, **kwargs) -> None:
        self._apply(_UpdateFeature(FeatureType.STATION, fid, **kwargs))

    def delete_station(self, fid: int) -> None:
        self._apply(_DeleteFeature(FeatureType.STATION, fid))

    def get_station(self, fid: int) -> Optional[StationFeature]:
        return self._state.stations.get(fid)

    def all_stations(self) -> List[StationFeature]:
        return sorted(self._state.stations.values(), key=lambda s: s.fid)

    # ------------------------------------------------------------------
    # Dump site operations
    # ------------------------------------------------------------------

    def add_dump_site(
        self,
        name: str,
        x: float,
        y: float,
        area_m2: float = 0.0,
        notes: str = "",
    ) -> DumpSiteFeature:
        fid  = self._alloc_fid(FeatureType.DUMP_SITE)
        feat = DumpSiteFeature(fid=fid, name=name, x=x, y=y,
                               area_m2=area_m2, notes=notes)
        self._apply(_AddFeature(FeatureType.DUMP_SITE, feat))
        return self._state.dump_sites[fid]

    def update_dump_site(self, fid: int, **kwargs) -> None:
        self._apply(_UpdateFeature(FeatureType.DUMP_SITE, fid, **kwargs))

    def delete_dump_site(self, fid: int) -> None:
        self._apply(_DeleteFeature(FeatureType.DUMP_SITE, fid))

    def get_dump_site(self, fid: int) -> Optional[DumpSiteFeature]:
        return self._state.dump_sites.get(fid)

    def all_dump_sites(self) -> List[DumpSiteFeature]:
        return sorted(self._state.dump_sites.values(), key=lambda d: d.fid)

    # ------------------------------------------------------------------
    # Undo / Redo
    # ------------------------------------------------------------------

    def undo(self) -> Optional[str]:
        """
        Undo the last command.

        Returns the description of the undone command, or None if
        the stack is empty.
        """
        if not self._undo_stack:
            return None
        cmd = self._undo_stack.pop()
        cmd.undo(self._state)
        self._redo_stack.append(cmd)
        return cmd.description()

    def redo(self) -> Optional[str]:
        """
        Redo the last undone command.

        Returns the description of the redone command, or None if
        the stack is empty.
        """
        if not self._redo_stack:
            return None
        cmd = self._redo_stack.pop()
        cmd.apply(self._state)
        self._undo_stack.append(cmd)
        return cmd.description()

    @property
    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo_stack)

    @property
    def undo_description(self) -> Optional[str]:
        return self._undo_stack[-1].description() if self._undo_stack else None

    @property
    def redo_description(self) -> Optional[str]:
        return self._redo_stack[-1].description() if self._redo_stack else None

    # ------------------------------------------------------------------
    # Import / Export (plain dicts – no QGIS)
    # ------------------------------------------------------------------

    def load_roads(self, records: Sequence[Dict]) -> None:
        """
        Bulk-load road records from a list of dicts.

        Each dict must have ``name`` and ``vertices``; optional keys are
        ``road_class``, ``notes``.  Existing roads are cleared first.
        """
        self._state.roads.clear()
        for rec in records:
            fid  = self._alloc_fid(FeatureType.ROAD)
            feat = RoadFeature(
                fid=fid,
                name=rec.get("name", f"Veg {fid}"),
                vertices=rec.get("vertices", []),
                road_class=rec.get("road_class", "skogsbilveg kl.3"),
                notes=rec.get("notes", ""),
            )
            self._state.roads[fid] = feat

    def load_stations(self, records: Sequence[Dict]) -> None:
        """Bulk-load station records.  Each dict must have ``name``, ``x``, ``y``."""
        self._state.stations.clear()
        for rec in records:
            fid  = self._alloc_fid(FeatureType.STATION)
            feat = StationFeature(
                fid=fid,
                name=rec.get("name", f"Standplass {fid}"),
                x=float(rec.get("x", 0.0)),
                y=float(rec.get("y", 0.0)),
                z=float(rec.get("z", 0.0)),
                capacity_t=float(rec.get("capacity_t", 0.0)),
                notes=rec.get("notes", ""),
            )
            self._state.stations[fid] = feat

    def load_dump_sites(self, records: Sequence[Dict]) -> None:
        """Bulk-load dump-site records.  Each dict must have ``name``, ``x``, ``y``."""
        self._state.dump_sites.clear()
        for rec in records:
            fid  = self._alloc_fid(FeatureType.DUMP_SITE)
            feat = DumpSiteFeature(
                fid=fid,
                name=rec.get("name", f"Velteplass {fid}"),
                x=float(rec.get("x", 0.0)),
                y=float(rec.get("y", 0.0)),
                area_m2=float(rec.get("area_m2", 0.0)),
                notes=rec.get("notes", ""),
            )
            self._state.dump_sites[fid] = feat

    def export_roads(self) -> List[Dict]:
        return [
            {
                "fid":        r.fid,
                "name":       r.name,
                "vertices":   list(r.vertices),
                "road_class": r.road_class,
                "length_m":   round(r.length_m, 2),
                "notes":      r.notes,
            }
            for r in self.all_roads()
        ]

    def export_stations(self) -> List[Dict]:
        return [
            {
                "fid":        s.fid,
                "name":       s.name,
                "x":          s.x,
                "y":          s.y,
                "z":          s.z,
                "capacity_t": s.capacity_t,
                "notes":      s.notes,
            }
            for s in self.all_stations()
        ]

    def export_dump_sites(self) -> List[Dict]:
        return [
            {
                "fid":     d.fid,
                "name":    d.name,
                "x":       d.x,
                "y":       d.y,
                "area_m2": d.area_m2,
                "notes":   d.notes,
            }
            for d in self.all_dump_sites()
        ]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _alloc_fid(self, ftype: FeatureType) -> int:
        fid = self._next_fid[ftype]
        self._next_fid[ftype] += 1
        return fid

    def _apply(self, cmd: _EditCommand) -> None:
        cmd.apply(self._state)
        self._undo_stack.append(cmd)
        if len(self._undo_stack) > self._max_undo:
            self._undo_stack.pop(0)
        # New action clears redo stack
        self._redo_stack.clear()


# ---------------------------------------------------------------------------
# Geometry helper
# ---------------------------------------------------------------------------

import math as _math


def _polyline_length(vertices: Sequence[Tuple]) -> float:
    """Return the 2-D cumulative length of a polyline (list of (x,y[,z]))."""
    total = 0.0
    for i in range(1, len(vertices)):
        dx = vertices[i][0] - vertices[i - 1][0]
        dy = vertices[i][1] - vertices[i - 1][1]
        total += _math.sqrt(dx * dx + dy * dy)
    return total
