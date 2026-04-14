# -*- coding: utf-8 -*-
"""
editor_dialog.py – Interactive Editor Dialog for SkogVegPlanering (STEG 6)

Lets the user:
  1. View and edit roads (veger), cable-way stations (standplass), and
     dump sites (velteplass) in three tabs.
  2. Add, delete, and rename features inline via editable table cells.
  3. Undo / redo any change with Ctrl-Z / Ctrl-Y.
  4. Import feature data from existing QGIS layers.
  5. Export the edited data to new QGIS memory layers or CSV.
"""

import csv

from qgis.PyQt.QtCore import Qt, QModelIndex
from qgis.PyQt.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QFileDialog,
    QMessageBox,
    QGroupBox,
    QTabWidget,
    QWidget,
    QSizePolicy,
    QShortcut,
    QAbstractItemView,
    QDoubleSpinBox,
    QLineEdit,
)
from qgis.PyQt.QtGui import QColor, QKeySequence
from qgis.core import (
    QgsProject,
    QgsWkbTypes,
    QgsVectorLayer,
    QgsFeature,
    QgsGeometry,
    QgsPointXY,
    QgsField,
)
try:
    from qgis.PyQt.QtCore import QMetaType
    _QSTRING = QMetaType.Type.QString
    _QINT    = QMetaType.Type.Int
    _QDOUBLE = QMetaType.Type.Double
except (ImportError, AttributeError):
    from qgis.PyQt.QtCore import QVariant
    _QSTRING = QVariant.String
    _QINT    = QVariant.Int
    _QDOUBLE = QVariant.Double

from ..processors.feature_editor import (
    FeatureEditor,
    FeatureType,
    RoadFeature,
    StationFeature,
    DumpSiteFeature,
)
from ..utils.logger import setup_logger

log = setup_logger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Column definitions (index, header, editable)
# ──────────────────────────────────────────────────────────────────────────────

_ROAD_COLS    = ["FID", "Navn", "Vegklasse", "Lengde (m)", "Notat"]
_STATION_COLS = ["FID", "Navn", "X", "Y", "Høyde (m)", "Kapasitet (t)", "Notat"]
_DUMP_COLS    = ["FID", "Navn", "X", "Y", "Areal (m²)", "Notat"]

_ROAD_EDITABLE    = {1, 2, 4}        # Navn, Vegklasse, Notat
_STATION_EDITABLE = {1, 2, 3, 4, 5, 6}  # Navn, X, Y, Høyde, Kapasitet, Notat
_DUMP_EDITABLE    = {1, 2, 3, 4, 5}     # Navn, X, Y, Areal, Notat


class EditorDialog(QDialog):
    """
    Interactive editor for roads, stations, and dump sites.

    Usage::

        dlg = EditorDialog(iface)
        dlg.exec_()
    """

    def __init__(self, iface, parent=None):
        super().__init__(parent or iface.mainWindow())
        self.iface   = iface
        self.editor  = FeatureEditor()

        self.setWindowTitle("SkogVegPlanering – Interaktiv redigering (STEG 6)")
        self.setMinimumSize(1000, 650)
        self._build_ui()
        self._register_shortcuts()
        self._refresh_all()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        main = QVBoxLayout(self)

        # ── Import banner ────────────────────────────────────────────────
        import_group = QGroupBox("Importer fra QGIS-lag")
        ig = QHBoxLayout(import_group)

        ig.addWidget(QLabel("Linjelag (veger):"))
        self.road_layer_combo = QComboBox()
        self.road_layer_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        ig.addWidget(self.road_layer_combo)

        ig.addWidget(QLabel("Punktlag (standplass):"))
        self.station_layer_combo = QComboBox()
        self.station_layer_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        ig.addWidget(self.station_layer_combo)

        ig.addWidget(QLabel("Punktlag (velteplass):"))
        self.dump_layer_combo = QComboBox()
        self.dump_layer_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        ig.addWidget(self.dump_layer_combo)

        import_btn = QPushButton("Importer valgte lag")
        import_btn.clicked.connect(self._on_import)
        ig.addWidget(import_btn)

        main.addWidget(import_group)
        self._populate_layer_combos()

        # ── Tabs ─────────────────────────────────────────────────────────
        self.tabs = QTabWidget()
        self.road_tab    = self._make_feature_tab(
            _ROAD_COLS, _ROAD_EDITABLE, self._on_road_changed
        )
        self.station_tab = self._make_feature_tab(
            _STATION_COLS, _STATION_EDITABLE, self._on_station_changed
        )
        self.dump_tab    = self._make_feature_tab(
            _DUMP_COLS, _DUMP_EDITABLE, self._on_dump_changed
        )

        self.tabs.addTab(self.road_tab,    "Veger")
        self.tabs.addTab(self.station_tab, "Standplass")
        self.tabs.addTab(self.dump_tab,    "Velteplass")
        main.addWidget(self.tabs)

        # ── Status / undo bar ────────────────────────────────────────────
        status_row = QHBoxLayout()
        self.status_label = QLabel("")
        status_row.addWidget(self.status_label)
        status_row.addStretch()
        self.undo_btn = QPushButton("Angre")
        self.undo_btn.setEnabled(False)
        self.undo_btn.clicked.connect(self._on_undo)
        status_row.addWidget(self.undo_btn)
        self.redo_btn = QPushButton("Gjenta")
        self.redo_btn.setEnabled(False)
        self.redo_btn.clicked.connect(self._on_redo)
        status_row.addWidget(self.redo_btn)
        main.addLayout(status_row)

        # ── Action buttons ───────────────────────────────────────────────
        btn_row = QHBoxLayout()

        add_road_btn = QPushButton("+ Legg til veg")
        add_road_btn.clicked.connect(self._on_add_road)
        btn_row.addWidget(add_road_btn)

        add_sta_btn = QPushButton("+ Legg til standplass")
        add_sta_btn.clicked.connect(self._on_add_station)
        btn_row.addWidget(add_sta_btn)

        add_dump_btn = QPushButton("+ Legg til velteplass")
        add_dump_btn.clicked.connect(self._on_add_dump)
        btn_row.addWidget(add_dump_btn)

        del_btn = QPushButton("Slett valgt")
        del_btn.clicked.connect(self._on_delete)
        btn_row.addWidget(del_btn)

        btn_row.addStretch()

        export_qgis_btn = QPushButton("Lagre til QGIS-lag")
        export_qgis_btn.clicked.connect(self._on_export_qgis)
        btn_row.addWidget(export_qgis_btn)

        export_csv_btn = QPushButton("Eksporter CSV...")
        export_csv_btn.clicked.connect(self._on_export_csv)
        btn_row.addWidget(export_csv_btn)

        close_btn = QPushButton("Lukk")
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(close_btn)

        main.addLayout(btn_row)

    @staticmethod
    def _make_feature_tab(headers, editable_cols, change_callback):
        """Create a QTableWidget inside a plain QWidget tab."""
        w = QWidget()
        lay = QVBoxLayout(w)
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        table.horizontalHeader().setStretchLastSection(True)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setAlternatingRowColors(True)

        # Make non-editable columns read-only via a flag helper applied on fill
        table.setProperty("editable_cols", editable_cols)
        table.itemChanged.connect(change_callback)
        lay.addWidget(table)
        w.table = table
        return w

    def _register_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+Z"), self, self._on_undo)
        QShortcut(QKeySequence("Ctrl+Y"), self, self._on_redo)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self, self._on_redo)

    # ------------------------------------------------------------------
    # Layer combo population
    # ------------------------------------------------------------------

    def _populate_layer_combos(self):
        line_layers  = []
        point_layers = []
        none_item    = ("(ingen)", None)

        for layer in QgsProject.instance().mapLayers().values():
            if not hasattr(layer, "wkbType"):
                continue
            gt = QgsWkbTypes.geometryType(layer.wkbType())
            if gt == QgsWkbTypes.LineGeometry:
                line_layers.append((layer.name(), layer.id()))
            elif gt == QgsWkbTypes.PointGeometry:
                point_layers.append((layer.name(), layer.id()))

        for combo, choices in [
            (self.road_layer_combo,    [none_item] + line_layers),
            (self.station_layer_combo, [none_item] + point_layers),
            (self.dump_layer_combo,    [none_item] + point_layers),
        ]:
            combo.clear()
            for name, lid in choices:
                combo.addItem(name, lid)

    # ------------------------------------------------------------------
    # Import from QGIS layers
    # ------------------------------------------------------------------

    def _on_import(self):
        roads_loaded = stations_loaded = dumps_loaded = 0

        road_id = self.road_layer_combo.currentData()
        if road_id:
            layer = QgsProject.instance().mapLayer(road_id)
            if layer:
                records = self._extract_roads_from_layer(layer)
                self.editor.load_roads(records)
                roads_loaded = len(records)

        sta_id = self.station_layer_combo.currentData()
        if sta_id:
            layer = QgsProject.instance().mapLayer(sta_id)
            if layer:
                records = self._extract_points_from_layer(layer)
                self.editor.load_stations(records)
                stations_loaded = len(records)

        dump_id = self.dump_layer_combo.currentData()
        if dump_id:
            layer = QgsProject.instance().mapLayer(dump_id)
            if layer:
                records = self._extract_points_from_layer(layer)
                self.editor.load_dump_sites(records)
                dumps_loaded = len(records)

        self._refresh_all()
        self._set_status(
            f"Importert: {roads_loaded} veg(er), "
            f"{stations_loaded} standplass, "
            f"{dumps_loaded} velteplass"
        )
        log.info("Imported %d roads, %d stations, %d dump sites",
                 roads_loaded, stations_loaded, dumps_loaded)

    @staticmethod
    def _safe_attr(feat, *field_names):
        """Return the first non-empty attribute value from *field_names*.

        Returns ``None`` if none of the fields exist on the feature or all
        values are falsy.  Using ``feat.attribute()`` raises a ``KeyError``
        when the field is absent from the layer schema, so we catch that.
        """
        for field in field_names:
            try:
                val = feat.attribute(field)
                if val:
                    return val
            except KeyError:
                pass
        return None

    @staticmethod
    def _extract_roads_from_layer(layer) -> list:
        records = []
        for feat in layer.getFeatures():
            geom = feat.geometry()
            if geom is None or geom.isEmpty():
                continue
            vertices = []
            try:
                pts = geom.asPolyline() if not geom.isMultipart() else \
                      (geom.asMultiPolyline()[0] if geom.asMultiPolyline() else [])
                vertices = [(p.x(), p.y()) for p in pts]
            except Exception:
                pass
            name = (EditorDialog._safe_attr(feat, "name", "navn", "id")
                    or f"Veg {feat.id()}")
            records.append({"name": str(name), "vertices": vertices})
        return records

    @staticmethod
    def _extract_points_from_layer(layer) -> list:
        records = []
        for feat in layer.getFeatures():
            geom = feat.geometry()
            if geom is None or geom.isEmpty():
                continue
            pt = geom.asPoint()
            name = (EditorDialog._safe_attr(feat, "name", "navn", "id")
                    or f"Punkt {feat.id()}")
            records.append({"name": str(name), "x": pt.x(), "y": pt.y()})
        return records

    # ------------------------------------------------------------------
    # Add / Delete
    # ------------------------------------------------------------------

    def _on_add_road(self):
        road = self.editor.add_road("Ny veg", [(0, 0), (100, 0)])
        self._refresh_roads()
        self._update_undo_redo()
        self._set_status(f"La til veg (fid={road.fid})")

    def _on_add_station(self):
        s = self.editor.add_station("Ny standplass", 0.0, 0.0)
        self._refresh_stations()
        self._update_undo_redo()
        self._set_status(f"La til standplass (fid={s.fid})")

    def _on_add_dump(self):
        d = self.editor.add_dump_site("Ny velteplass", 0.0, 0.0)
        self._refresh_dumps()
        self._update_undo_redo()
        self._set_status(f"La til velteplass (fid={d.fid})")

    def _on_delete(self):
        tab_idx = self.tabs.currentIndex()
        tab     = [self.road_tab, self.station_tab, self.dump_tab][tab_idx]
        table   = tab.table
        rows    = sorted({idx.row() for idx in table.selectedIndexes()}, reverse=True)
        if not rows:
            return

        ftype = [FeatureType.ROAD, FeatureType.STATION, FeatureType.DUMP_SITE][tab_idx]
        delete_fn = [
            self.editor.delete_road,
            self.editor.delete_station,
            self.editor.delete_dump_site,
        ][tab_idx]

        for row in rows:
            fid_item = table.item(row, 0)
            if fid_item:
                try:
                    fid = int(fid_item.text())
                    delete_fn(fid)
                except (ValueError, KeyError):
                    pass

        self._refresh_all()
        self._update_undo_redo()
        self._set_status(f"Slettet {len(rows)} element(er)")

    # ------------------------------------------------------------------
    # Cell editing – roads
    # ------------------------------------------------------------------

    def _on_road_changed(self, item: QTableWidgetItem):
        self._on_cell_changed(
            item, self.road_tab.table, FeatureType.ROAD,
            col_map={1: "name", 2: "road_class", 4: "notes"},
            refresh_fn=self._refresh_roads,
        )

    def _on_station_changed(self, item: QTableWidgetItem):
        self._on_cell_changed(
            item, self.station_tab.table, FeatureType.STATION,
            col_map={1: "name", 2: "x", 3: "y", 4: "z", 5: "capacity_t", 6: "notes"},
            float_cols={2, 3, 4, 5},
            refresh_fn=self._refresh_stations,
        )

    def _on_dump_changed(self, item: QTableWidgetItem):
        self._on_cell_changed(
            item, self.dump_tab.table, FeatureType.DUMP_SITE,
            col_map={1: "name", 2: "x", 3: "y", 4: "area_m2", 5: "notes"},
            float_cols={2, 3, 4},
            refresh_fn=self._refresh_dumps,
        )

    def _on_cell_changed(
        self, item, table, ftype, col_map, float_cols=None, refresh_fn=None
    ):
        if float_cols is None:
            float_cols = set()
        col = item.column()
        if col not in col_map:
            return
        row = item.row()
        fid_item = table.item(row, 0)
        if not fid_item:
            return
        try:
            fid = int(fid_item.text())
        except ValueError:
            return
        field = col_map[col]
        raw   = item.text().strip()
        if col in float_cols:
            try:
                value = float(raw)
            except ValueError:
                return
        else:
            value = raw

        update_fn = {
            FeatureType.ROAD:      self.editor.update_road,
            FeatureType.STATION:   self.editor.update_station,
            FeatureType.DUMP_SITE: self.editor.update_dump_site,
        }[ftype]

        try:
            update_fn(fid, **{field: value})
        except (KeyError, AttributeError):
            pass

        if refresh_fn:
            # Block signals while refreshing to avoid recursive calls
            table.blockSignals(True)
            refresh_fn()
            table.blockSignals(False)
        self._update_undo_redo()

    # ------------------------------------------------------------------
    # Table refresh helpers
    # ------------------------------------------------------------------

    def _refresh_all(self):
        self._refresh_roads()
        self._refresh_stations()
        self._refresh_dumps()
        self._update_undo_redo()

    def _refresh_roads(self):
        table = self.road_tab.table
        table.blockSignals(True)
        roads = self.editor.all_roads()
        table.setRowCount(len(roads))
        for row, r in enumerate(roads):
            values = [str(r.fid), r.name, r.road_class,
                      f"{r.length_m:.1f}", r.notes]
            self._fill_row(table, row, values, _ROAD_EDITABLE)
        table.blockSignals(False)

    def _refresh_stations(self):
        table = self.station_tab.table
        table.blockSignals(True)
        stations = self.editor.all_stations()
        table.setRowCount(len(stations))
        for row, s in enumerate(stations):
            values = [str(s.fid), s.name, f"{s.x:.4f}", f"{s.y:.4f}",
                      f"{s.z:.2f}", f"{s.capacity_t:.1f}", s.notes]
            self._fill_row(table, row, values, _STATION_EDITABLE)
        table.blockSignals(False)

    def _refresh_dumps(self):
        table = self.dump_tab.table
        table.blockSignals(True)
        dumps = self.editor.all_dump_sites()
        table.setRowCount(len(dumps))
        for row, d in enumerate(dumps):
            values = [str(d.fid), d.name, f"{d.x:.4f}", f"{d.y:.4f}",
                      f"{d.area_m2:.1f}", d.notes]
            self._fill_row(table, row, values, _DUMP_EDITABLE)
        table.blockSignals(False)

    @staticmethod
    def _fill_row(table, row, values, editable_cols):
        for col, text in enumerate(values):
            item = QTableWidgetItem(str(text))
            if col not in editable_cols:
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                item.setBackground(QColor("#f0f0f0"))
            table.setItem(row, col, item)

    # ------------------------------------------------------------------
    # Undo / Redo
    # ------------------------------------------------------------------

    def _on_undo(self):
        desc = self.editor.undo()
        if desc:
            self._refresh_all()
            self._set_status(f"Angret: {desc}")

    def _on_redo(self):
        desc = self.editor.redo()
        if desc:
            self._refresh_all()
            self._set_status(f"Gjentatt: {desc}")

    def _update_undo_redo(self):
        self.undo_btn.setEnabled(self.editor.can_undo)
        self.redo_btn.setEnabled(self.editor.can_redo)
        if self.editor.undo_description:
            self.undo_btn.setToolTip(f"Angre: {self.editor.undo_description}")
        if self.editor.redo_description:
            self.redo_btn.setToolTip(f"Gjenta: {self.editor.redo_description}")

    # ------------------------------------------------------------------
    # Export to QGIS layers
    # ------------------------------------------------------------------

    def _on_export_qgis(self):
        n_roads = self._create_road_layer()
        n_sta   = self._create_station_layer()
        n_dump  = self._create_dump_layer()
        QMessageBox.information(
            self, "Lagret til QGIS",
            f"Opprettet lag med:\n"
            f"  {n_roads} veg(er)\n"
            f"  {n_sta} standplass\n"
            f"  {n_dump} velteplass",
        )
        log.info("Exported to QGIS: %d roads, %d stations, %d dump sites",
                 n_roads, n_sta, n_dump)

    def _create_road_layer(self) -> int:
        try:
            vl = QgsVectorLayer("LineString?crs=EPSG:4326", "Veger (redigert)", "memory")
            pr = vl.dataProvider()
            pr.addAttributes([
                QgsField("fid",        _QINT),
                QgsField("name",       _QSTRING),
                QgsField("road_class", _QSTRING),
                QgsField("length_m",   _QDOUBLE),
                QgsField("notes",      _QSTRING),
            ])
            vl.updateFields()
            features = []
            for r in self.editor.all_roads():
                feat = QgsFeature()
                pts  = [QgsPointXY(v[0], v[1]) for v in r.vertices]
                feat.setGeometry(QgsGeometry.fromPolylineXY(pts))
                feat.setAttributes([r.fid, r.name, r.road_class,
                                    round(r.length_m, 2), r.notes])
                features.append(feat)
            pr.addFeatures(features)
            vl.updateExtents()
            QgsProject.instance().addMapLayer(vl)
            return len(features)
        except Exception as exc:
            log.warning("Could not create road layer: %s", exc)
            return 0

    def _create_station_layer(self) -> int:
        try:
            vl = QgsVectorLayer("Point?crs=EPSG:4326", "Standplass (redigert)", "memory")
            pr = vl.dataProvider()
            pr.addAttributes([
                QgsField("fid",        _QINT),
                QgsField("name",       _QSTRING),
                QgsField("z_m",        _QDOUBLE),
                QgsField("capacity_t", _QDOUBLE),
                QgsField("notes",      _QSTRING),
            ])
            vl.updateFields()
            features = []
            for s in self.editor.all_stations():
                feat = QgsFeature()
                feat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(s.x, s.y)))
                feat.setAttributes([s.fid, s.name, round(s.z, 2),
                                    round(s.capacity_t, 1), s.notes])
                features.append(feat)
            pr.addFeatures(features)
            vl.updateExtents()
            QgsProject.instance().addMapLayer(vl)
            return len(features)
        except Exception as exc:
            log.warning("Could not create station layer: %s", exc)
            return 0

    def _create_dump_layer(self) -> int:
        try:
            vl = QgsVectorLayer("Point?crs=EPSG:4326", "Velteplass (redigert)", "memory")
            pr = vl.dataProvider()
            pr.addAttributes([
                QgsField("fid",     _QINT),
                QgsField("name",    _QSTRING),
                QgsField("area_m2", _QDOUBLE),
                QgsField("notes",   _QSTRING),
            ])
            vl.updateFields()
            features = []
            for d in self.editor.all_dump_sites():
                feat = QgsFeature()
                feat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(d.x, d.y)))
                feat.setAttributes([d.fid, d.name, round(d.area_m2, 1), d.notes])
                features.append(feat)
            pr.addFeatures(features)
            vl.updateExtents()
            QgsProject.instance().addMapLayer(vl)
            return len(features)
        except Exception as exc:
            log.warning("Could not create dump site layer: %s", exc)
            return 0

    # ------------------------------------------------------------------
    # Export to CSV
    # ------------------------------------------------------------------

    def _on_export_csv(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Lagre redigerte elementer", "", "CSV-filer (*.csv)"
        )
        if not path:
            return
        if not path.endswith(".csv"):
            path += ".csv"
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["type"] + _ROAD_COLS)
                for r in self.editor.all_roads():
                    writer.writerow(["road", r.fid, r.name, r.road_class,
                                     round(r.length_m, 2), r.notes])
                writer.writerow([])
                writer.writerow(["type"] + _STATION_COLS)
                for s in self.editor.all_stations():
                    writer.writerow(["station", s.fid, s.name,
                                     round(s.x, 4), round(s.y, 4),
                                     round(s.z, 2), round(s.capacity_t, 1), s.notes])
                writer.writerow([])
                writer.writerow(["type"] + _DUMP_COLS)
                for d in self.editor.all_dump_sites():
                    writer.writerow(["dump_site", d.fid, d.name,
                                     round(d.x, 4), round(d.y, 4),
                                     round(d.area_m2, 1), d.notes])
            QMessageBox.information(self, "Eksport fullført", f"Lagret til:\n{path}")
            log.info("Editor CSV exported to %s", path)
        except OSError as exc:
            QMessageBox.critical(self, "Eksportfeil", str(exc))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_status(self, msg: str):
        self.status_label.setText(msg)
