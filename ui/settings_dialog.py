# -*- coding: utf-8 -*-
"""
settings_dialog.py – Plugin Settings Dialog for SkogVegPlanering

Lets the user configure default values that are shared across all dialogs:
  • Validation thresholds (max slope, min curve radius)
  • Default layer names (road layer, import layer)

Settings are persisted via QgsSettings so they survive QGIS restarts.
"""

from qgis.PyQt.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QDoubleSpinBox,
    QLineEdit,
    QPushButton,
    QGroupBox,
    QDialogButtonBox,
)
from qgis.core import QgsSettings

from ..utils.logger import setup_logger

log = setup_logger(__name__)

# QgsSettings key prefix for this plugin
_PREFIX = "SkogVegPlanering"


def _key(name):
    return f"{_PREFIX}/{name}"


# ---------------------------------------------------------------------------
# Public helpers – read/write individual settings
# ---------------------------------------------------------------------------

def get_max_slope():
    """Return the stored max slope (%) or the default 12.0."""
    return float(QgsSettings().value(_key("max_slope_percent"), 12.0))


def get_min_radius():
    """Return the stored min curve radius (m) or the default 20.0."""
    return float(QgsSettings().value(_key("min_curve_radius"), 20.0))


def get_layer_name():
    """Return the stored default road-layer name."""
    return QgsSettings().value(_key("layer_name"), "")


def get_import_layer_name():
    """Return the stored default import-layer name."""
    return QgsSettings().value(_key("import_layer_name"), "")


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------

class SettingsDialog(QDialog):
    """
    Plugin settings dialog.

    Usage::

        dlg = SettingsDialog(iface)
        dlg.exec_()
    """

    def __init__(self, iface, parent=None):
        super().__init__(parent or iface.mainWindow())
        self.iface = iface
        self._settings = QgsSettings()

        self.setWindowTitle("SkogVegPlanering – Innstillinger")
        self.setMinimumWidth(380)
        self._build_ui()
        self._load()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        main_layout = QVBoxLayout(self)

        # ── Validation parameters ────────────────────────────────────────
        val_group = QGroupBox("Validering")
        val_layout = QVBoxLayout(val_group)

        slope_row = QHBoxLayout()
        slope_row.addWidget(QLabel("Maks stigning (%):"))
        self.slope_spin = QDoubleSpinBox()
        self.slope_spin.setRange(0.1, 30.0)
        self.slope_spin.setSingleStep(0.5)
        self.slope_spin.setDecimals(1)
        self.slope_spin.setToolTip("Maksimal tillatt stigning for skogsveger (standard: 12 %)")
        slope_row.addWidget(self.slope_spin)
        slope_row.addStretch()
        val_layout.addLayout(slope_row)

        radius_row = QHBoxLayout()
        radius_row.addWidget(QLabel("Min kurvaturradius (m):"))
        self.radius_spin = QDoubleSpinBox()
        self.radius_spin.setRange(1.0, 500.0)
        self.radius_spin.setSingleStep(5.0)
        self.radius_spin.setDecimals(1)
        self.radius_spin.setToolTip("Minimum tillatt kurvaturradius (standard: 20 m)")
        radius_row.addWidget(self.radius_spin)
        radius_row.addStretch()
        val_layout.addLayout(radius_row)

        main_layout.addWidget(val_group)

        # ── Default layer names ──────────────────────────────────────────
        layer_group = QGroupBox("Standard lagnavn")
        layer_layout = QVBoxLayout(layer_group)

        road_row = QHBoxLayout()
        road_row.addWidget(QLabel("Veglinje-lag:"))
        self.layer_edit = QLineEdit()
        self.layer_edit.setPlaceholderText("(valgfritt)")
        self.layer_edit.setToolTip("Standardnavn for veglinje-laget som velges automatisk i dialogene")
        road_row.addWidget(self.layer_edit)
        layer_layout.addLayout(road_row)

        import_row = QHBoxLayout()
        import_row.addWidget(QLabel("Import-lag:"))
        self.import_edit = QLineEdit()
        self.import_edit.setPlaceholderText("(valgfritt)")
        self.import_edit.setToolTip("Standardnavn for import-laget")
        import_row.addWidget(self.import_edit)
        layer_layout.addLayout(import_row)

        main_layout.addWidget(layer_group)

        # ── Buttons ──────────────────────────────────────────────────────
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.RestoreDefaults
        )
        button_box.accepted.connect(self._on_ok)
        button_box.rejected.connect(self.reject)
        button_box.button(QDialogButtonBox.RestoreDefaults).clicked.connect(self._on_restore)
        main_layout.addWidget(button_box)

    # ------------------------------------------------------------------
    # Load / save
    # ------------------------------------------------------------------

    def _load(self):
        """Populate widgets from stored settings."""
        self.slope_spin.setValue(get_max_slope())
        self.radius_spin.setValue(get_min_radius())
        self.layer_edit.setText(get_layer_name())
        self.import_edit.setText(get_import_layer_name())

    def _save(self):
        """Write widget values back to QgsSettings."""
        s = self._settings
        s.setValue(_key("max_slope_percent"), self.slope_spin.value())
        s.setValue(_key("min_curve_radius"), self.radius_spin.value())
        s.setValue(_key("layer_name"), self.layer_edit.text().strip())
        s.setValue(_key("import_layer_name"), self.import_edit.text().strip())
        log.info(
            "Settings saved: max_slope=%.1f, min_radius=%.1f, layer=%r, import=%r",
            self.slope_spin.value(),
            self.radius_spin.value(),
            self.layer_edit.text().strip(),
            self.import_edit.text().strip(),
        )

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_ok(self):
        self._save()
        self.accept()

    def _on_restore(self):
        """Reset widgets to factory defaults (does not save immediately)."""
        self.slope_spin.setValue(12.0)
        self.radius_spin.setValue(20.0)
        self.layer_edit.clear()
        self.import_edit.clear()
