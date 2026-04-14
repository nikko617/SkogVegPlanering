# -*- coding: utf-8 -*-
"""
validation_dialog.py – Qt dialog for road network validation (STEG 2)

Lets the user:
  1. Pick a line layer from the current QGIS project.
  2. Configure max slope and min curve radius.
  3. Run validation and inspect results in a table.
  4. Export results to CSV.
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
    QSplitter,
    QSizePolicy,
)
from qgis.PyQt.QtGui import QColor
from qgis.core import QgsProject, QgsWkbTypes

from ..core.validator import RoadValidator
from ..utils.logger import setup_logger

log = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

class _ValidationWorker(QThread):
    """Runs validation in a background thread to keep the UI responsive."""

    finished = pyqtSignal(list)   # list[ValidationResult]
    error    = pyqtSignal(str)

    def __init__(self, layer, max_slope, min_radius):
        super().__init__()
        self.layer = layer
        self.max_slope = max_slope
        self.min_radius = min_radius

    def run(self):
        try:
            validator = RoadValidator(
                max_slope_percent=self.max_slope,
                min_curve_radius=self.min_radius,
            )
            results = validator.validate_layer(self.layer)
            self.finished.emit(results)
        except Exception as exc:
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------

class ValidationDialog(QDialog):
    """
    Main validation dialog.

    Usage::

        dlg = ValidationDialog(iface)
        dlg.exec_()
    """

    # Table column indices
    _COL_FID    = 0
    _COL_CHECK  = 1
    _COL_STATUS = 2
    _COL_VALUE  = 3
    _COL_MSG    = 4
    _HEADERS    = ["Feature ID", "Sjekk", "Status", "Verdi", "Melding"]

    def __init__(self, iface, parent=None):
        super().__init__(parent or iface.mainWindow())
        self.iface = iface
        self._worker = None
        self._results = []

        self.setWindowTitle("SkogVegPlanering – Vegnettvalidering")
        self.setMinimumSize(800, 520)
        self._build_ui()
        self._populate_layers()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        main_layout = QVBoxLayout(self)

        # ── Layer & parameters ──────────────────────────────────────────
        param_group = QGroupBox("Innstillinger")
        param_layout = QVBoxLayout(param_group)

        # Layer selector
        layer_row = QHBoxLayout()
        layer_row.addWidget(QLabel("Veglinje-lag:"))
        self.layer_combo = QComboBox()
        self.layer_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layer_row.addWidget(self.layer_combo)
        param_layout.addLayout(layer_row)

        # Max slope
        slope_row = QHBoxLayout()
        slope_row.addWidget(QLabel("Maks stigning (%):"))
        self.slope_spin = QDoubleSpinBox()
        self.slope_spin.setRange(0.1, 30.0)
        self.slope_spin.setSingleStep(0.5)
        self.slope_spin.setValue(12.0)
        slope_row.addWidget(self.slope_spin)
        slope_row.addStretch()
        param_layout.addLayout(slope_row)

        # Min curve radius
        radius_row = QHBoxLayout()
        radius_row.addWidget(QLabel("Min kurvaturradius (m):"))
        self.radius_spin = QDoubleSpinBox()
        self.radius_spin.setRange(1.0, 500.0)
        self.radius_spin.setSingleStep(5.0)
        self.radius_spin.setValue(20.0)
        radius_row.addWidget(self.radius_spin)
        radius_row.addStretch()
        param_layout.addLayout(radius_row)

        main_layout.addWidget(param_group)

        # ── Progress bar ────────────────────────────────────────────────
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)   # indeterminate
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        # ── Results table ────────────────────────────────────────────────
        self.table = QTableWidget(0, len(self._HEADERS))
        self.table.setHorizontalHeaderLabels(self._HEADERS)
        self.table.horizontalHeader().setSectionResizeMode(
            self._COL_MSG, QHeaderView.Stretch
        )
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setAlternatingRowColors(True)
        main_layout.addWidget(self.table)

        # ── Summary label ────────────────────────────────────────────────
        self.summary_label = QLabel("")
        main_layout.addWidget(self.summary_label)

        # ── Buttons ──────────────────────────────────────────────────────
        btn_row = QHBoxLayout()

        self.run_btn = QPushButton("Valider")
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

        main_layout.addLayout(btn_row)

    # ------------------------------------------------------------------
    # Layer population
    # ------------------------------------------------------------------

    def _populate_layers(self):
        """Fill the combo with line layers from the current project."""
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
            QMessageBox.critical(self, "Feil", "Laget ble ikke funnet i prosjektet.")
            return

        self._set_running(True)

        self._worker = _ValidationWorker(
            layer=layer,
            max_slope=self.slope_spin.value(),
            min_radius=self.radius_spin.value(),
        )
        self._worker.finished.connect(self._on_results)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_results(self, results):
        self._results = results
        self._set_running(False)
        self._populate_table(results)
        self.export_btn.setEnabled(bool(results))

        total  = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed
        self.summary_label.setText(
            f"Totalt: {total}  |  ✓ Godkjent: {passed}  |  ✗ Feilet: {failed}"
        )
        log.info("Validation complete: %d checks, %d passed, %d failed", total, passed, failed)

    def _on_error(self, msg):
        self._set_running(False)
        QMessageBox.critical(self, "Valideringsfeil", f"En feil oppstod:\n{msg}")
        log.error("Validation error: %s", msg)

    def _on_export(self):
        if not self._results:
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Lagre resultater", "", "CSV-filer (*.csv)"
        )
        if not path:
            return
        if not path.endswith(".csv"):
            path += ".csv"

        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self._HEADERS)
                for r in self._results:
                    writer.writerow([
                        r.feature_id,
                        r.check_type,
                        "Godkjent" if r.passed else "Feilet",
                        f"{r.value:.2f}" if r.value is not None else "",
                        r.message,
                    ])
            QMessageBox.information(self, "Eksport fullført", f"Lagret til:\n{path}")
        except OSError as exc:
            QMessageBox.critical(self, "Eksportfeil", str(exc))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_running(self, running):
        self.run_btn.setEnabled(not running)
        self.progress_bar.setVisible(running)

    def _populate_table(self, results):
        self.table.setRowCount(0)
        self.table.setRowCount(len(results))

        green = QColor("#d4edda")
        red   = QColor("#f8d7da")

        for row, r in enumerate(results):
            color = green if r.passed else red

            items = [
                str(r.feature_id),
                r.check_type,
                "✓ Godkjent" if r.passed else "✗ Feilet",
                f"{r.value:.2f}" if r.value is not None else "—",
                r.message,
            ]
            for col, text in enumerate(items):
                item = QTableWidgetItem(text)
                item.setBackground(color)
                if col == self._COL_STATUS:
                    item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(row, col, item)

        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(
            self._COL_MSG, QHeaderView.Stretch
        )
