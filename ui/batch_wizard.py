# -*- coding: utf-8 -*-
"""
batch_wizard.py – Batch PDF Import Wizard for SkogVegPlanering (STEG 3)

Three-page QWizard:
  Page 1 – Select PDF files
  Page 2 – Configure detection parameters and target CRS
  Page 3 – Run import, show progress and log, open result layer

The heavy PDF → polyline work runs in a ``_ImportWorker`` QThread so the
UI stays responsive.
"""

import os

from qgis.PyQt.QtCore import Qt, QThread, pyqtSignal
from qgis.PyQt.QtWidgets import (
    QWizard,
    QWizardPage,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QFileDialog,
    QDoubleSpinBox,
    QSpinBox,
    QGroupBox,
    QProgressBar,
    QTextEdit,
    QSizePolicy,
    QMessageBox,
)
from qgis.core import (
    QgsProject,
    QgsVectorLayer,
    QgsFeature,
    QgsGeometry,
    QgsPointXY,
    QgsFields,
    QgsField,
    QgsCoordinateReferenceSystem,
    QgsWkbTypes,
)
try:
    from qgis.PyQt.QtCore import QMetaType
    _QSTRING = QMetaType.Type.QString
    _QINT    = QMetaType.Type.Int
except (ImportError, AttributeError):
    from qgis.PyQt.QtCore import QVariant
    _QSTRING = QVariant.String
    _QINT    = QVariant.Int

from ..processors.pdf_importer import DetectionParams, PdfImporter, ImportResult
from ..utils.logger import setup_logger

log = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Field ids used by QWizard to share data between pages
# ---------------------------------------------------------------------------

_FIELD_PDF_PATHS    = "pdf_paths"      # list[str]
_FIELD_LAYER_NAME   = "layer_name"     # str
_FIELD_CRS_AUTH_ID  = "crs_auth_id"   # str  e.g. "EPSG:25833"
_FIELD_CANNY_LOW    = "canny_low"      # int
_FIELD_CANNY_HIGH   = "canny_high"     # int
_FIELD_MIN_LINE_LEN = "min_line_len"   # float
_FIELD_MAX_LINE_GAP = "max_line_gap"   # float
_FIELD_HOUGH_THRESH = "hough_thresh"   # int
_FIELD_DPI          = "dpi"            # int


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

class _ImportWorker(QThread):
    """Runs PdfImporter.import_files() in a background thread."""

    file_done  = pyqtSignal(str, int, int)   # path, lines_found, error_count
    all_done   = pyqtSignal(list)             # list[ImportResult]
    log_msg    = pyqtSignal(str)

    def __init__(self, pdf_paths, params: DetectionParams):
        super().__init__()
        self._paths  = pdf_paths
        self._params = params
        self._abort  = False

    def abort(self):
        self._abort = True

    def run(self):
        importer = PdfImporter(self._params)
        results = []

        for path in self._paths:
            if self._abort:
                self.log_msg.emit("Import avbrutt av bruker.")
                break

            self.log_msg.emit(f"Behandler: {os.path.basename(path)} ...")
            result = importer.import_file(path)
            results.append(result)

            for err in result.errors:
                self.log_msg.emit(f"  [ADVARSEL] {err}")

            self.log_msg.emit(
                f"  [OK] {result.line_count} linje(r) funnet på {result.page_count} side(r)"
            )
            self.file_done.emit(path, result.line_count, len(result.errors))

        self.all_done.emit(results)


# ---------------------------------------------------------------------------
# Page 1 – Select files
# ---------------------------------------------------------------------------

class _PageSelectFiles(QWizardPage):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Steg 1 – Velg PDF-filer")
        self.setSubTitle(
            "Legg til én eller flere skannede vegkart-PDF-er som skal importeres."
        )
        self._pdf_paths = []

        layout = QVBoxLayout(self)

        # File list
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.ExtendedSelection)
        layout.addWidget(self.list_widget)

        # Buttons
        btn_row = QHBoxLayout()

        add_btn = QPushButton("Legg til filer…")
        add_btn.clicked.connect(self._add_files)
        btn_row.addWidget(add_btn)

        remove_btn = QPushButton("Fjern valgte")
        remove_btn.clicked.connect(self._remove_selected)
        btn_row.addWidget(remove_btn)

        btn_row.addStretch()
        layout.addLayout(btn_row)

        # Count label
        self._count_label = QLabel("Ingen filer valgt.")
        layout.addWidget(self._count_label)

    # ---- QWizardPage overrides -------------------------------------------

    def isComplete(self):
        return len(self._pdf_paths) > 0

    # ---- Slots -----------------------------------------------------------

    def _add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Velg PDF-filer", "", "PDF-filer (*.pdf);;Alle filer (*)"
        )
        for p in paths:
            if p not in self._pdf_paths:
                self._pdf_paths.append(p)
                self.list_widget.addItem(QListWidgetItem(os.path.basename(p)))
        self._update_count()
        self.completeChanged.emit()

    def _remove_selected(self):
        for item in self.list_widget.selectedItems():
            row = self.list_widget.row(item)
            self.list_widget.takeItem(row)
            del self._pdf_paths[row]
        self._update_count()
        self.completeChanged.emit()

    def _update_count(self):
        n = len(self._pdf_paths)
        self._count_label.setText(
            f"{n} fil(er) valgt." if n else "Ingen filer valgt."
        )

    # ---- Accessor (used by wizard) ---------------------------------------

    def get_pdf_paths(self):
        return list(self._pdf_paths)


# ---------------------------------------------------------------------------
# Page 2 – Settings
# ---------------------------------------------------------------------------

class _PageSettings(QWizardPage):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Steg 2 – Innstillinger")
        self.setSubTitle(
            "Konfigurer målkoordinatsystem, lagnavn og deteksjonsparametere."
        )

        layout = QVBoxLayout(self)

        # ── Layer ──────────────────────────────────────────────────────
        layer_group = QGroupBox("Outputlag")
        layer_layout = QVBoxLayout(layer_group)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Lagnavn:"))
        self.layer_name_edit = QLineEdit("importerte_veglinjer")
        name_row.addWidget(self.layer_name_edit)
        layer_layout.addLayout(name_row)

        crs_row = QHBoxLayout()
        crs_row.addWidget(QLabel("Koordinatsystem (auth-ID):"))
        self.crs_edit = QLineEdit("EPSG:25833")
        self.crs_edit.setToolTip(
            "Oppgi EPSG-kode eller annen gyldig CRS-identifikator, "
            "f.eks. EPSG:25833 (UTM 33N, Norge)"
        )
        crs_row.addWidget(self.crs_edit)
        layer_layout.addLayout(crs_row)

        layout.addWidget(layer_group)

        # ── Detection ──────────────────────────────────────────────────
        det_group = QGroupBox("Linjedeteksjon")
        det_layout = QVBoxLayout(det_group)

        def spin_row(label, widget, tooltip=""):
            row = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setMinimumWidth(220)
            row.addWidget(lbl)
            row.addWidget(widget)
            row.addStretch()
            if tooltip:
                widget.setToolTip(tooltip)
            return row

        # Canny low
        self.canny_low_spin = QSpinBox()
        self.canny_low_spin.setRange(0, 254)
        self.canny_low_spin.setValue(50)
        det_layout.addLayout(spin_row("Canny nedre terskel:", self.canny_low_spin))

        # Canny high
        self.canny_high_spin = QSpinBox()
        self.canny_high_spin.setRange(1, 255)
        self.canny_high_spin.setValue(150)
        det_layout.addLayout(spin_row("Canny øvre terskel:", self.canny_high_spin))

        # Hough threshold
        self.hough_thresh_spin = QSpinBox()
        self.hough_thresh_spin.setRange(1, 500)
        self.hough_thresh_spin.setValue(80)
        det_layout.addLayout(
            spin_row("Hough stemmegrense:", self.hough_thresh_spin,
                     "Minimum antall piksler som støtter en linje")
        )

        # Min line length
        self.min_line_spin = QDoubleSpinBox()
        self.min_line_spin.setRange(1.0, 2000.0)
        self.min_line_spin.setSingleStep(5.0)
        self.min_line_spin.setValue(30.0)
        det_layout.addLayout(
            spin_row("Min. linjelengde (piksler):", self.min_line_spin)
        )

        # Max line gap
        self.max_gap_spin = QDoubleSpinBox()
        self.max_gap_spin.setRange(0.0, 500.0)
        self.max_gap_spin.setSingleStep(2.0)
        self.max_gap_spin.setValue(10.0)
        det_layout.addLayout(
            spin_row("Maks. mellomrom (piksler):", self.max_gap_spin,
                     "Maksimalt gap mellom segmenter for å bli slått sammen")
        )

        # DPI
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(72, 600)
        self.dpi_spin.setSingleStep(25)
        self.dpi_spin.setValue(150)
        det_layout.addLayout(spin_row("Render-DPI:", self.dpi_spin))

        layout.addWidget(det_group)
        layout.addStretch()

    # ---- Accessors -------------------------------------------------------

    def get_layer_name(self) -> str:
        return self.layer_name_edit.text().strip() or "importerte_veglinjer"

    def get_crs_auth_id(self) -> str:
        return self.crs_edit.text().strip() or "EPSG:25833"

    def get_detection_params(self) -> DetectionParams:
        return DetectionParams(
            canny_low=self.canny_low_spin.value(),
            canny_high=self.canny_high_spin.value(),
            hough_threshold=self.hough_thresh_spin.value(),
            hough_min_line_length=self.min_line_spin.value(),
            hough_max_line_gap=self.max_gap_spin.value(),
            dpi=self.dpi_spin.value(),
        )


# ---------------------------------------------------------------------------
# Page 3 – Import and results
# ---------------------------------------------------------------------------

class _PageImport(QWizardPage):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Steg 3 – Import")
        self.setSubTitle("Importerer valgte PDF-filer. Vennligst vent …")
        self.setFinalPage(True)

        self._worker: _ImportWorker = None
        self._results = []
        self._done = False

        layout = QVBoxLayout(self)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMinimumHeight(200)
        layout.addWidget(self.log_view)

        self.open_layer_btn = QPushButton("Åpne importert lag i QGIS")
        self.open_layer_btn.setEnabled(False)
        self.open_layer_btn.clicked.connect(self._open_layer)
        layout.addWidget(self.open_layer_btn)

    # ---- QWizardPage overrides -------------------------------------------

    def isComplete(self):
        return self._done

    def initializePage(self):
        """Called by QWizard when this page becomes visible."""
        wizard = self.wizard()
        pdf_paths = wizard.page(0).get_pdf_paths()
        params    = wizard.page(1).get_detection_params()
        self._layer_name   = wizard.page(1).get_layer_name()
        self._crs_auth_id  = wizard.page(1).get_crs_auth_id()

        total = len(pdf_paths)
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(0)
        self._processed = 0
        self._results = []
        self._done = False

        self._worker = _ImportWorker(pdf_paths, params)
        self._worker.log_msg.connect(self._append_log)
        self._worker.file_done.connect(self._on_file_done)
        self._worker.all_done.connect(self._on_all_done)
        self._worker.start()

    def cleanupPage(self):
        """Stop worker if user navigates back."""
        if self._worker and self._worker.isRunning():
            self._worker.abort()
            self._worker.wait()

    # ---- Slots -----------------------------------------------------------

    def _append_log(self, msg: str):
        self.log_view.append(msg)

    def _on_file_done(self, path: str, line_count: int, error_count: int):
        self._processed += 1
        self.progress_bar.setValue(self._processed)

    def _on_all_done(self, results):
        self._results = results
        self._done = True

        total_lines  = sum(r.line_count for r in results)
        total_errors = sum(len(r.errors) for r in results)

        self._append_log(
            f"\nFerdig! Totalt {total_lines} linje(r) funnet. "
            f"{total_errors} feil."
        )

        # Build QGIS layer from results
        try:
            self._vector_layer = self._build_layer(results)
            self.open_layer_btn.setEnabled(True)
            self._append_log(
                f"Lag '{self._layer_name}' klar med "
                f"{self._vector_layer.featureCount()} feature(r)."
            )
        except Exception as exc:
            self._append_log(f"Kunne ikke opprette lag: {exc}")
            log.error("Layer creation failed: %s", exc)

        self.completeChanged.emit()

    def _build_layer(self, results) -> QgsVectorLayer:
        """Create an in-memory QgsVectorLayer from import results."""
        crs = QgsCoordinateReferenceSystem(self._crs_auth_id)
        layer = QgsVectorLayer(
            f"LineString?crs={crs.authid()}",
            self._layer_name,
            "memory",
        )

        provider = layer.dataProvider()
        fields = QgsFields()
        fields.append(QgsField("source_pdf", _QSTRING))
        fields.append(QgsField("page",       _QINT))
        provider.addAttributes(fields)
        layer.updateFields()

        features = []
        for result in results:
            basename = os.path.basename(result.pdf_path)
            for idx, polyline in enumerate(result.polylines):
                page_num = (
                    result.polyline_pages[idx]
                    if idx < len(result.polyline_pages)
                    else 0
                )
                feat = QgsFeature()
                feat.setFields(layer.fields())
                qgs_points = [QgsPointXY(x, y) for x, y in polyline]
                feat.setGeometry(QgsGeometry.fromPolylineXY(qgs_points))
                feat.setAttribute("source_pdf", basename)
                feat.setAttribute("page", page_num)
                features.append(feat)

        provider.addFeatures(features)
        layer.updateExtents()
        return layer

    def _open_layer(self):
        """Add the built layer to the current QGIS project."""
        try:
            QgsProject.instance().addMapLayer(self._vector_layer)
            self._append_log(f"Laget '{self._layer_name}' er lagt til i prosjektet.")
            self.open_layer_btn.setEnabled(False)
        except Exception as exc:
            QMessageBox.critical(
                self, "Feil", f"Kunne ikke legge til lag:\n{exc}"
            )


# ---------------------------------------------------------------------------
# Wizard
# ---------------------------------------------------------------------------

class BatchWizard(QWizard):
    """
    Batch PDF Import Wizard.

    Usage::

        wizard = BatchWizard(iface)
        wizard.exec_()
    """

    def __init__(self, iface, parent=None):
        super().__init__(parent or iface.mainWindow())
        self.iface = iface

        self.setWindowTitle("SkogVegPlanering – Batch PDF Import")
        self.setMinimumSize(640, 480)
        self.setWizardStyle(QWizard.ModernStyle)

        self.addPage(_PageSelectFiles())
        self.addPage(_PageSettings())
        self.addPage(_PageImport())

        log.info("BatchWizard created")
