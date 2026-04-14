# -*- coding: utf-8 -*-
"""
plugin.py – Main SkogVegPlanering Plugin Class
Refactored from VegValidering for unified forest road planning
"""

import os
from qgis.PyQt.QtWidgets import (
    QAction,
    QMessageBox,
)
from qgis.PyQt.QtGui import QIcon
from qgis.core import QgsProject

from .utils.logger import setup_logger

log = setup_logger(__name__)


class SkogVegPlaneringPlugin:
    """
    SkogVegPlanering – Unified plugin for forest road planning.
    Integrates road validation, analysis, cost distribution, and batch PDF import.
    """

    def __init__(self, iface):
        """Initialize the plugin."""
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)
        self.toolbar = None

        # State variables
        self.layer_name = None
        self.import_layer_name = None
        self.max_slope_percent = 12.0
        self.min_curve_radius = 20.0

        log.info("SkogVegPlanering plugin initialized")

    def initGui(self):
        """Create toolbar and menu items."""
        log.info("Initializing GUI")

        # Create toolbar
        self.toolbar = self.iface.addToolBar("SkogVegPlanering")
        self.toolbar.setObjectName("SkogVegPlanering")

        # Helper function to add toolbar actions
        def add_action(icon_file, text, callback):
            icon_path = os.path.join(self.plugin_dir, "icons", icon_file)
            action = QAction(
                QIcon(icon_path) if os.path.exists(icon_path) else QIcon(),
                text,
                self.iface.mainWindow()
            )
            action.triggered.connect(callback)
            self.toolbar.addAction(action)
            return action

        # Toolbar actions (BASIC FOR NOW)
        add_action("tree.png", "Settings", self.open_settings)
        self.toolbar.addSeparator()
        add_action("tree.png", "Batch Import Wizard", self.open_batch_wizard)
        self.toolbar.addSeparator()
        add_action("tree.png", "Validate Network", self.run_validation)
        self.toolbar.addSeparator()
        add_action("tree.png", "Cost Distribution", self.open_cost_dialog)
        self.toolbar.addSeparator()
        add_action("tree.png", "Cable Way Planning", self.open_cableway_dialog)

        log.info("GUI initialized successfully")

    def unload(self):
        """Unload the plugin."""
        log.info("Unloading SkogVegPlanering plugin")
        if self.toolbar:
            self.toolbar.deleteLater()

    # --- Menu Actions ---

    def open_settings(self):
        """Open settings dialog (placeholder)."""
        QMessageBox.information(
            self.iface.mainWindow(),
            "SkogVegPlanering",
            "Settings dialog - Coming soon!"
        )
        log.info("Settings dialog requested")

    def open_batch_wizard(self):
        """Open batch PDF import wizard (STEG 3)."""
        from .ui.batch_wizard import BatchWizard
        wizard = BatchWizard(self.iface)
        wizard.exec_()
        log.info("Batch wizard closed")

    def run_validation(self):
        """Open road network validation dialog (STEG 2)."""
        from .ui.validation_dialog import ValidationDialog
        dlg = ValidationDialog(self.iface)
        dlg.exec_()
        log.info("Validation dialog closed")

    def open_cost_dialog(self):
        """Open cost distribution dialog (STEG 4)."""
        from .ui.cost_dialog import CostDialog
        dlg = CostDialog(self.iface)
        dlg.exec_()
        log.info("Cost dialog closed")

    def open_cableway_dialog(self):
        """Open cable way planning dialog (STEG 5)."""
        from .ui.cableway_dialog import CablewayDialog
        dlg = CablewayDialog(self.iface)
        dlg.exec_()
        log.info("Cableway dialog closed")
