# -*- coding: utf-8 -*-
"""
cost_dialog.py – Cost Distribution Dialog for SkogVegPlanering (STEG 4)

Lets the user:
  1. Pick a polygon layer representing properties (eiendommer).
  2. Map layer attributes to area, volume, haul-distance, and name fields.
  3. Enter the total project cost and choose a cost basis.
  4. Run the distribution calculation (in a QThread).
  5. Inspect results in a sortable table and export to CSV.
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
)
from qgis.PyQt.QtGui import QColor
from qgis.core import QgsProject, QgsWkbTypes

from ..processors.cost_calculator import CostBasis, CostCalculator
from ..utils.logger import setup_logger

log = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

class _CostWorker(QThread):
    """Runs cost distribution in a background thread."""

    finished = pyqtSignal(object)   # CostDistributionResult
    error    = pyqtSignal(str)

    def __init__(self, layer, basis, total_cost, area_field,
                 volume_field, distance_field, name_field):
        super().__init__()
        self.layer          = layer
        self.basis          = basis
        self.total_cost     = total_cost
        self.area_field     = area_field
        self.volume_field   = volume_field
        self.distance_field = distance_field
        self.name_field     = name_field

    def run(self):
        try:
            records, warnings = CostCalculator.records_from_layer(
                layer=self.layer,
                area_field=self.area_field,
                volume_field=self.volume_field or None,
                distance_field=self.distance_field or None,
                name_field=self.name_field or None,
            )
            calc = CostCalculator(basis=self.basis)
            result = calc.distribute(records, total_cost_nok=self.total_cost)
            result.warnings = warnings + result.warnings
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------

class CostDialog(QDialog):
    """
    Cost distribution dialog.

    Usage::

        dlg = CostDialog(iface)
        dlg.exec_()
    """

    _COL_ID       = 0
    _COL_NAME     = 1
    _COL_WEIGHT   = 2
    _COL_SHARE    = 3
    _COL_COST     = 4
    _HEADERS      = ["Eiendom ID", "Navn", "Vekt", "Andel (%)", "Kostnad (kr)"]

    def __init__(self, iface, parent=None):
        super().__init__(parent or iface.mainWindow())
        self.iface    = iface
        self._worker  = None
        self._results = None

        self.setWindowTitle("SkogVegPlanering – Kostnadsfordeling")
        self.setMinimumSize(860, 560)
        self._build_ui()
        self._populate_layers()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        main = QVBoxLayout(self)

        # ── Layer & field mapping ────────────────────────────────────────
        layer_group = QGroupBox("Eiendomsdata")
        lg = QVBoxLayout(layer_group)

        def combo_row(label, attr):
            row = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setMinimumWidth(200)
            row.addWidget(lbl)
            cb = QComboBox()
            cb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            row.addWidget(cb)
            setattr(self, attr, cb)
            return row

        lg.addLayout(combo_row("Eiendomslag (polygoner):", "layer_combo"))
        self.layer_combo.currentIndexChanged.connect(self._on_layer_changed)

        lg.addLayout(combo_row("Arealattributt (m²):", "area_field_combo"))
        lg.addLayout(combo_row("Volumattributt (m³, valgfri):", "volume_field_combo"))
        lg.addLayout(combo_row("Tømmerkjøringsavstand (m, valgfri):", "distance_field_combo"))
        lg.addLayout(combo_row("Navnattributt (valgfri):", "name_field_combo"))

        main.addWidget(layer_group)

        # ── Cost parameters ──────────────────────────────────────────────
        cost_group = QGroupBox("Kostnadsparametere")
        cg = QVBoxLayout(cost_group)

        cost_row = QHBoxLayout()
        cost_row.addWidget(QLabel("Total prosjektkostnad (kr):"))
        self.cost_spin = QDoubleSpinBox()
        self.cost_spin.setRange(0.0, 1e10)
        self.cost_spin.setSingleStep(10_000)
        self.cost_spin.setValue(1_000_000)
        self.cost_spin.setGroupSeparatorShown(True)
        cost_row.addWidget(self.cost_spin)
        cost_row.addStretch()
        cg.addLayout(cost_row)

        basis_row = QHBoxLayout()
        basis_row.addWidget(QLabel("Fordelingsgrunnlag:"))
        self.basis_combo = QComboBox()
        self.basis_combo.addItem("Areal (m²)",               CostBasis.AREA)
        self.basis_combo.addItem("Tømmervolum (m³)",          CostBasis.VOLUME)
        self.basis_combo.addItem("Areal / √Kjøringsavstand",  CostBasis.DISTANCE_AREA)
        basis_row.addWidget(self.basis_combo)
        basis_row.addStretch()
        cg.addLayout(basis_row)

        main.addWidget(cost_group)

        # ── Progress ────────────────────────────────────────────────────
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        main.addWidget(self.progress_bar)

        # ── Results table ────────────────────────────────────────────────
        self.table = QTableWidget(0, len(self._HEADERS))
        self.table.setHorizontalHeaderLabels(self._HEADERS)
        self.table.horizontalHeader().setSectionResizeMode(
            self._COL_NAME, QHeaderView.Stretch
        )
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)
        main.addWidget(self.table)

        # ── Summary label ────────────────────────────────────────────────
        self.summary_label = QLabel("")
        main.addWidget(self.summary_label)

        # ── Buttons ──────────────────────────────────────────────────────
        btn_row = QHBoxLayout()

        self.run_btn = QPushButton("Beregn")
        self.run_btn.setDefault(True)
        self.run_btn.clicked.connect(self._on_run)
        btn_row.addWidget(self.run_btn)

        self.export_btn = QPushButton("Eksporter CSV…")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._on_export)
        btn_row.addWidget(self.export_btn)

        btn_row.addStretch()

        close_btn = QPushButton("Lukk")
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(close_btn)

        main.addLayout(btn_row)

    # ------------------------------------------------------------------
    # Layer / field population
    # ------------------------------------------------------------------

    def _populate_layers(self):
        """Fill the layer combo with polygon layers from the current project."""
        self.layer_combo.blockSignals(True)
        self.layer_combo.clear()
        for layer in QgsProject.instance().mapLayers().values():
            if hasattr(layer, "wkbType") and QgsWkbTypes.geometryType(
                layer.wkbType()
            ) == QgsWkbTypes.PolygonGeometry:
                self.layer_combo.addItem(layer.name(), layer.id())

        if self.layer_combo.count() == 0:
            self.layer_combo.addItem("(ingen polygonlag funnet)", None)
            self.run_btn.setEnabled(False)

        self.layer_combo.blockSignals(False)
        self._on_layer_changed()

    def _on_layer_changed(self):
        """Refresh field combos when the selected layer changes."""
        layer_id = self.layer_combo.currentData()
        layer = QgsProject.instance().mapLayer(layer_id) if layer_id else None

        field_names = [f.name() for f in layer.fields()] if layer else []
        none_option = "(ingen)"

        for combo, hint_fields in [
            (self.area_field_combo,     ["area_m2", "areal", "area"]),
            (self.volume_field_combo,   ["volume_m3", "volum", "vol"]),
            (self.distance_field_combo, ["haul_distance_m", "avstand", "dist"]),
            (self.name_field_combo,     ["name", "navn", "eiendom"]),
        ]:
            combo.clear()
            combo.addItem(none_option, "")
            for fn in field_names:
                combo.addItem(fn, fn)
            # Pre-select a sensible default
            for hint in hint_fields:
                idx = combo.findData(hint)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
                    break

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _on_run(self):
        layer_id = self.layer_combo.currentData()
        if not layer_id:
            QMessageBox.warning(self, "Mangler lag", "Velg et eiendomslag.")
            return

        layer = QgsProject.instance().mapLayer(layer_id)
        if layer is None:
            QMessageBox.critical(self, "Feil", "Laget ble ikke funnet i prosjektet.")
            return

        basis       = self.basis_combo.currentData()
        total_cost  = self.cost_spin.value()
        area_field  = self.area_field_combo.currentData() or "area_m2"
        vol_field   = self.volume_field_combo.currentData() or ""
        dist_field  = self.distance_field_combo.currentData() or ""
        name_field  = self.name_field_combo.currentData() or ""

        self._set_running(True)
        self._worker = _CostWorker(
            layer=layer,
            basis=basis,
            total_cost=total_cost,
            area_field=area_field,
            volume_field=vol_field,
            distance_field=dist_field,
            name_field=name_field,
        )
        self._worker.finished.connect(self._on_results)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_results(self, result):
        self._results = result
        self._set_running(False)
        self._populate_table(result.shares)
        self.export_btn.setEnabled(bool(result.shares))

        n = len(result.shares)
        self.summary_label.setText(
            f"{n} eiendom(mer)  |  "
            f"Total kostnad: {result.total_cost_nok:,.0f} kr  |  "
            f"Basis: {result.basis.value}"
        )

        if result.warnings:
            QMessageBox.warning(
                self, "Advarsler",
                "\n".join(result.warnings[:10])
                + ("\n…" if len(result.warnings) > 10 else ""),
            )

        log.info(
            "Cost distribution complete: %d properties, basis=%s, total=%.0f",
            n, result.basis.value, result.total_cost_nok,
        )

    def _on_error(self, msg):
        self._set_running(False)
        QMessageBox.critical(self, "Beregningsfeil", f"En feil oppstod:\n{msg}")
        log.error("Cost calculation error: %s", msg)

    def _on_export(self):
        if not self._results or not self._results.shares:
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Lagre kostnadsfordeling", "", "CSV-filer (*.csv)"
        )
        if not path:
            return
        if not path.endswith(".csv"):
            path += ".csv"

        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self._HEADERS)
                for s in self._results.shares:
                    writer.writerow([
                        s.property_id,
                        s.name,
                        f"{s.weight:.4f}",
                        f"{s.share_percent:.4f}",
                        f"{s.cost_nok:.2f}",
                    ])
            QMessageBox.information(self, "Eksport fullført", f"Lagret til:\n{path}")
            log.info("Cost distribution exported to %s", path)
        except OSError as exc:
            QMessageBox.critical(self, "Eksportfeil", str(exc))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_running(self, running: bool):
        self.run_btn.setEnabled(not running)
        self.progress_bar.setVisible(running)

    def _populate_table(self, shares):
        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)
        self.table.setRowCount(len(shares))

        # Colour gradient: highest share → green-ish, lowest → yellow-ish
        max_share = max((s.share_fraction for s in shares), default=1.0)

        for row, s in enumerate(shares):
            intensity = int(220 - 80 * (s.share_fraction / max_share)) if max_share > 0 else 220
            color = QColor(intensity, 255, intensity)

            items = [
                str(s.property_id),
                s.name,
                f"{s.weight:.2f}",
                f"{s.share_percent:.4f}",
                f"{s.cost_nok:,.2f}",
            ]
            for col, text in enumerate(items):
                item = QTableWidgetItem(text)
                item.setBackground(color)
                if col in (self._COL_WEIGHT, self._COL_SHARE, self._COL_COST):
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(row, col, item)

        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(
            self._COL_NAME, QHeaderView.Stretch
        )
        self.table.setSortingEnabled(True)
