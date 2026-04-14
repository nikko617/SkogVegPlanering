# -*- coding: utf-8 -*-
"""
cableway_dialog.py – Cable Way Planning Dialog for SkogVegPlanering (STEG 5)

Lets the user:
  1. Pick a line layer (road centre-lines).
  2. Configure station interval, winch length, max span, and min slope.
  3. Run the planning algorithm (in a QThread).
  4. Inspect results in a table (one row per standplass station).
  5. Export results to CSV.
  6. Optionally create a QGIS point layer with the computed stations.
"""

import csv
import os

from qgis.PyQt.QtCore import Qt, QThread, pyqtSignal
from qgis.PyQt.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QDoubleSpinBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QFileDialog,
    QMessageBox,
    QGroupBox,
    QProgressBar,
    QSizePolicy,
    QCheckBox,
    QSpinBox,
)
from qgis.PyQt.QtGui import QColor
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
    _QINT    = QMetaType.Type.Int
    _QDOUBLE = QMetaType.Type.Double
except (ImportError, AttributeError):
    from qgis.PyQt.QtCore import QVariant
    _QINT    = QVariant.Int
    _QDOUBLE = QVariant.Double

from ..processors.cableway_planner import CablewayPlanner
from ..utils.logger import setup_logger

log = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

class _CablewayWorker(QThread):
    """Runs cable way planning in a background thread."""

    finished = pyqtSignal(list)   # list[CablewayPlan]
    error    = pyqtSignal(str)

    def __init__(self, layer, station_interval, winch_length, max_span, min_slope):
        super().__init__()
        self.layer            = layer
        self.station_interval = station_interval
        self.winch_length     = winch_length
        self.max_span         = max_span
        self.min_slope        = min_slope

    def run(self):
        try:
            planner = CablewayPlanner(
                station_interval_m=self.station_interval,
                winch_length_m=self.winch_length,
                max_span_m=self.max_span,
                min_slope_pct=self.min_slope,
            )
            plans = planner.plan_layer(self.layer)
            self.finished.emit(plans)
        except Exception as exc:
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------

class CablewayDialog(QDialog):
    """
    Cable way planning dialog.

    Usage::

        dlg = CablewayDialog(iface)
        dlg.exec_()
    """

    _COL_POLY    = 0
    _COL_SID     = 1
    _COL_DIST    = 2
    _COL_X       = 3
    _COL_Y       = 4
    _COL_Z       = 5
    _COL_SLOPE   = 6
    _COL_RADIUS  = 7
    _HEADERS     = [
        "Polyline ID", "Standplass #", "Avstand (m)",
        "X", "Y", "Høy (m)", "Stigning (%)", "Rekkevidde (m)",
    ]

    def __init__(self, iface, parent=None):
        super().__init__(parent or iface.mainWindow())
        self.iface   = iface
        self._worker = None
        self._plans  = []

        self.setWindowTitle("SkogVegPlanering – Taubanplanlegging (STEG 5)")
        self.setMinimumSize(960, 600)
        self._build_ui()
        self._populate_layers()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        main = QVBoxLayout(self)

        # ── Layer selection ──────────────────────────────────────────────
        layer_group = QGroupBox("Veglinje-lag")
        lg = QVBoxLayout(layer_group)
        row = QHBoxLayout()
        row.addWidget(QLabel("Linjelag (vegsenterlinje):"))
        self.layer_combo = QComboBox()
        self.layer_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        row.addWidget(self.layer_combo)
        lg.addLayout(row)
        main.addWidget(layer_group)

        # ── Planning parameters ──────────────────────────────────────────
        param_group = QGroupBox("Planleggingsparametere")
        pg = QVBoxLayout(param_group)

        def spin_row(label, attr, min_val, max_val, step, default, suffix="m"):
            r = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setMinimumWidth(260)
            r.addWidget(lbl)
            sb = QDoubleSpinBox()
            sb.setRange(min_val, max_val)
            sb.setSingleStep(step)
            sb.setValue(default)
            sb.setSuffix(f" {suffix}")
            r.addWidget(sb)
            r.addStretch()
            setattr(self, attr, sb)
            return r

        pg.addLayout(spin_row(
            "Stasjonssavstand (m):", "interval_spin",
            10, 2000, 50, 200,
        ))
        pg.addLayout(spin_row(
            "Vinsjelengde – rekkevidde (m):", "winch_spin",
            10, 2000, 50, 400,
        ))
        pg.addLayout(spin_row(
            "Maks kabelspenn (m):", "span_spin",
            10, 3000, 100, 500,
        ))
        pg.addLayout(spin_row(
            "Min stigning for taubane (%):", "slope_spin",
            0, 100, 1, 15, suffix="%",
        ))

        self.create_layer_chk = QCheckBox(
            "Opprett punktlag med standplass-posisjoner i QGIS"
        )
        self.create_layer_chk.setChecked(True)
        pg.addWidget(self.create_layer_chk)

        main.addWidget(param_group)

        # ── Progress ────────────────────────────────────────────────────
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        main.addWidget(self.progress_bar)

        # ── Results table ────────────────────────────────────────────────
        self.table = QTableWidget(0, len(self._HEADERS))
        self.table.setHorizontalHeaderLabels(self._HEADERS)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)
        main.addWidget(self.table)

        # ── Summary ─────────────────────────────────────────────────────
        self.summary_label = QLabel("")
        main.addWidget(self.summary_label)

        # ── Buttons ──────────────────────────────────────────────────────
        btn_row = QHBoxLayout()

        self.run_btn = QPushButton("Beregn")
        self.run_btn.setDefault(True)
        self.run_btn.clicked.connect(self._on_run)
        btn_row.addWidget(self.run_btn)

        self.export_btn = QPushButton("Eksporter CSV...")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._on_export)
        btn_row.addWidget(self.export_btn)

        btn_row.addStretch()

        close_btn = QPushButton("Lukk")
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(close_btn)

        main.addLayout(btn_row)

    # ------------------------------------------------------------------
    # Layer population
    # ------------------------------------------------------------------

    def _populate_layers(self):
        self.layer_combo.clear()
        for layer in QgsProject.instance().mapLayers().values():
            if hasattr(layer, "wkbType") and QgsWkbTypes.geometryType(
                layer.wkbType()
            ) == QgsWkbTypes.LineGeometry:
                self.layer_combo.addItem(layer.name(), layer.id())

        if self.layer_combo.count() == 0:
            self.layer_combo.addItem("(ingen linjelag funnet)", None)
            self.run_btn.setEnabled(False)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _on_run(self):
        layer_id = self.layer_combo.currentData()
        if not layer_id:
            QMessageBox.warning(self, "Mangler lag", "Velg et veglinje-lag.")
            return

        layer = QgsProject.instance().mapLayer(layer_id)
        if layer is None:
            QMessageBox.critical(self, "Feil", "Laget ble ikke funnet.")
            return

        self._set_running(True)
        self._worker = _CablewayWorker(
            layer=layer,
            station_interval=self.interval_spin.value(),
            winch_length=self.winch_spin.value(),
            max_span=self.span_spin.value(),
            min_slope=self.slope_spin.value(),
        )
        self._worker.finished.connect(self._on_results)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_results(self, plans):
        self._plans = plans
        self._set_running(False)

        all_stations = [
            (plan, s) for plan in plans for s in plan.stations
        ]
        self._populate_table(all_stations)
        self.export_btn.setEnabled(bool(all_stations))

        all_warnings = [w for plan in plans for w in plan.warnings]
        total_stations = len(all_stations)
        total_roads    = len(plans)
        self.summary_label.setText(
            f"{total_roads} polyline(r)  |  "
            f"{total_stations} standplass-posisjoner  |  "
            f"{len(all_warnings)} advarsel(er)"
        )

        if all_warnings:
            QMessageBox.warning(
                self, "Advarsler",
                "\n".join(all_warnings[:10])
                + ("\n..." if len(all_warnings) > 10 else ""),
            )

        if self.create_layer_chk.isChecked() and all_stations:
            self._create_point_layer(all_stations)

        log.info(
            "Cable way planning complete: %d polylines, %d stations",
            total_roads, total_stations,
        )

    def _on_error(self, msg):
        self._set_running(False)
        QMessageBox.critical(self, "Planleggingsfeil", f"En feil oppstod:\n{msg}")
        log.error("Cableway planning error: %s", msg)

    def _on_export(self):
        if not self._plans:
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Lagre tauban-stasjoner", "", "CSV-filer (*.csv)"
        )
        if not path:
            return
        if not path.endswith(".csv"):
            path += ".csv"

        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self._HEADERS)
                for plan in self._plans:
                    for s in plan.stations:
                        writer.writerow([
                            plan.polyline_id,
                            s.station_id,
                            f"{s.distance_along_road_m:.2f}",
                            f"{s.x:.4f}",
                            f"{s.y:.4f}",
                            f"{s.z:.2f}",
                            f"{s.slope_pct:.2f}",
                            f"{s.coverage_radius_m:.0f}",
                        ])
            QMessageBox.information(self, "Eksport fullført", f"Lagret til:\n{path}")
            log.info("Cableway stations exported to %s", path)
        except OSError as exc:
            QMessageBox.critical(self, "Eksportfeil", str(exc))

    # ------------------------------------------------------------------
    # QGIS layer creation
    # ------------------------------------------------------------------

    def _create_point_layer(self, all_stations):
        """Create a temporary memory point layer with standplass positions."""
        try:
            vl = QgsVectorLayer("Point?crs=EPSG:4326", "Tauban-standplass", "memory")
            pr = vl.dataProvider()
            pr.addAttributes([
                QgsField("polyline_id", _QINT),
                QgsField("station_id",  _QINT),
                QgsField("dist_m",      _QDOUBLE),
                QgsField("z_m",         _QDOUBLE),
                QgsField("slope_pct",   _QDOUBLE),
                QgsField("radius_m",    _QDOUBLE),
            ])
            vl.updateFields()

            features = []
            for plan, s in all_stations:
                feat = QgsFeature()
                feat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(s.x, s.y)))
                feat.setAttributes([
                    plan.polyline_id,
                    s.station_id,
                    round(s.distance_along_road_m, 2),
                    round(s.z, 2),
                    round(s.slope_pct, 2),
                    round(s.coverage_radius_m, 0),
                ])
                features.append(feat)

            pr.addFeatures(features)
            vl.updateExtents()

            QgsProject.instance().addMapLayer(vl)
            log.info("Created standplass point layer with %d features", len(features))
        except Exception as exc:
            log.warning("Could not create point layer: %s", exc)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_running(self, running: bool):
        self.run_btn.setEnabled(not running)
        self.progress_bar.setVisible(running)

    def _populate_table(self, all_stations):
        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)
        self.table.setRowCount(len(all_stations))

        colors = [QColor("#cce5ff"), QColor("#d4edda")]  # alternate by polyline

        prev_poly = None
        color_idx = 0
        for row, (plan, s) in enumerate(all_stations):
            if plan.polyline_id != prev_poly:
                prev_poly = plan.polyline_id
                color_idx = (color_idx + 1) % len(colors)
            color = colors[color_idx]

            values = [
                str(plan.polyline_id),
                str(s.station_id),
                f"{s.distance_along_road_m:.1f}",
                f"{s.x:.2f}",
                f"{s.y:.2f}",
                f"{s.z:.2f}",
                f"{s.slope_pct:.1f}",
                f"{s.coverage_radius_m:.0f}",
            ]
            for col, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setBackground(color)
                if col >= self._COL_DIST:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(row, col, item)

        self.table.resizeColumnsToContents()
        self.table.setSortingEnabled(True)
