# -*- coding: utf-8 -*-
"""
pdf_importer.py – Batch PDF import for SkogVegPlanering (STEG 3)

Converts scanned forest road map PDFs to polyline coordinate lists that can
be loaded into QGIS as a vector layer.

Pipeline per PDF page
---------------------
1. Render page to a Pillow Image (via pypdf + Pillow, or pdf2image if available).
2. Convert to grayscale NumPy array for OpenCV processing.
3. Apply Canny edge detection followed by Probabilistic Hough Line Transform.
4. Return detected segments as ``list[list[tuple[float, float]]]`` –
   each inner list is a two-point polyline ``[(x1, y1), (x2, y2)]``.

The class is intentionally free of any QGIS imports so it can be unit-tested
in a plain Python environment.
"""

import io
import logging
import math
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy imports – gracefully degrade in test environments
# ---------------------------------------------------------------------------

try:
    import cv2 as _cv2
    _HAS_CV2 = True
except ImportError:  # pragma: no cover
    _cv2 = None
    _HAS_CV2 = False

try:
    from PIL import Image as _PILImage
    _HAS_PIL = True
except ImportError:  # pragma: no cover
    _PILImage = None
    _HAS_PIL = False

try:
    import pypdf as _pypdf
    _HAS_PYPDF = True
except ImportError:  # pragma: no cover
    _pypdf = None
    _HAS_PYPDF = False

try:
    import fitz as _fitz  # PyMuPDF – renders vector + raster PDF pages
    _HAS_FITZ = True
except ImportError:  # pragma: no cover
    _fitz = None
    _HAS_FITZ = False


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

Polyline = List[Tuple[float, float]]


@dataclass
class ImportResult:
    """Result for a single PDF file."""

    pdf_path: str
    polylines: List[Polyline] = field(default_factory=list)
    # Page index (0-based) for each polyline in ``polylines``
    polyline_pages: List[int] = field(default_factory=list)
    page_count: int = 0
    errors: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return not self.errors

    @property
    def line_count(self) -> int:
        return len(self.polylines)


# ---------------------------------------------------------------------------
# Detection parameters
# ---------------------------------------------------------------------------

@dataclass
class DetectionParams:
    """Tuning parameters for the Canny + Hough pipeline."""

    # Canny thresholds (intensity values 0-255)
    canny_low: int = 50
    canny_high: int = 150

    # Probabilistic Hough parameters
    hough_rho: float = 1.0          # distance resolution (pixels)
    hough_theta: float = math.pi / 180  # angle resolution (radians)
    hough_threshold: int = 80       # minimum votes
    hough_min_line_length: float = 30.0  # pixels
    hough_max_line_gap: float = 10.0     # pixels

    # Image scaling – render at this DPI (affects detection resolution)
    dpi: int = 150

    def validate(self):
        """Raise ValueError on obviously wrong parameter values."""
        if not (0 <= self.canny_low < self.canny_high <= 255):
            raise ValueError(
                f"Invalid Canny thresholds: low={self.canny_low}, high={self.canny_high}"
            )
        if self.hough_min_line_length <= 0:
            raise ValueError("hough_min_line_length must be positive")
        if self.dpi <= 0:
            raise ValueError("dpi must be positive")


# ---------------------------------------------------------------------------
# Core importer
# ---------------------------------------------------------------------------

class PdfImporter:
    """
    Import scanned forest road maps from PDFs into polyline coordinate lists.

    Parameters
    ----------
    params : DetectionParams, optional
        Detection tuning parameters.  Defaults to ``DetectionParams()``.

    Examples
    --------
    >>> importer = PdfImporter()
    >>> result = importer.import_file("/path/to/map.pdf")
    >>> print(result.line_count, "lines detected")
    """

    def __init__(self, params: Optional[DetectionParams] = None):
        self.params = params or DetectionParams()
        self.params.validate()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def import_file(self, pdf_path: str) -> ImportResult:
        """
        Import a single PDF file.

        Parameters
        ----------
        pdf_path : str
            Absolute or relative path to the PDF.

        Returns
        -------
        ImportResult
        """
        result = ImportResult(pdf_path=pdf_path)

        if not os.path.isfile(pdf_path):
            result.errors.append(f"Filen finnes ikke: {pdf_path}")
            return result

        if not _HAS_CV2:
            result.errors.append("opencv-python-headless er ikke installert")
            return result

        if not _HAS_FITZ and not _HAS_PYPDF:
            result.errors.append("PyMuPDF eller pypdf er ikke installert")
            return result

        if _HAS_FITZ:
            # PyMuPDF renders both vector and raster PDF content correctly
            try:
                doc = _fitz.open(pdf_path)
                result.page_count = doc.page_count
                log.info("Importing %s (%d page(s)) via PyMuPDF", pdf_path, result.page_count)

                for page_num in range(doc.page_count):
                    log.debug("Processing page %d/%d", page_num + 1, result.page_count)
                    try:
                        image = self._render_fitz_page(doc[page_num])
                        polylines = self._detect_lines(image)
                        result.polylines.extend(polylines)
                        result.polyline_pages.extend([page_num] * len(polylines))
                        log.debug(
                            "Page %d: %d line(s) detected", page_num + 1, len(polylines)
                        )
                    except Exception as exc:
                        msg = f"Side {page_num + 1}: {exc}"
                        result.errors.append(msg)
                        log.warning(msg)

                doc.close()
            except Exception as exc:
                result.errors.append(f"Kunne ikke lese PDF: {exc}")
                log.error("Failed to read %s: %s", pdf_path, exc)
        else:
            # Fallback: pypdf can only extract embedded raster XObject images
            if not _HAS_PIL:
                result.errors.append("Pillow er ikke installert")
                return result

            try:
                reader = _pypdf.PdfReader(pdf_path)
                result.page_count = len(reader.pages)
                log.info("Importing %s (%d page(s)) via pypdf", pdf_path, result.page_count)

                for page_num, page in enumerate(reader.pages):
                    log.debug("Processing page %d/%d", page_num + 1, result.page_count)
                    try:
                        image = self._render_page(page)
                        polylines = self._detect_lines(image)
                        result.polylines.extend(polylines)
                        result.polyline_pages.extend([page_num] * len(polylines))
                        log.debug(
                            "Page %d: %d line(s) detected", page_num + 1, len(polylines)
                        )
                    except Exception as exc:
                        msg = f"Side {page_num + 1}: {exc}"
                        result.errors.append(msg)
                        log.warning(msg)

            except Exception as exc:
                result.errors.append(f"Kunne ikke lese PDF: {exc}")
                log.error("Failed to read %s: %s", pdf_path, exc)

        return result

    def import_files(self, pdf_paths: List[str]) -> List[ImportResult]:
        """
        Import multiple PDF files.

        Parameters
        ----------
        pdf_paths : list[str]

        Returns
        -------
        list[ImportResult]
        """
        results = []
        for path in pdf_paths:
            results.append(self.import_file(path))
        return results

    # ------------------------------------------------------------------
    # Image pipeline (testable with synthetic numpy arrays)
    # ------------------------------------------------------------------

    def detect_lines_from_array(
        self, gray_array: "np.ndarray"
    ) -> List[Polyline]:
        """
        Run line detection on a pre-existing grayscale NumPy array.

        This entry point is used by unit tests to inject synthetic images
        without requiring a real PDF file.

        Parameters
        ----------
        gray_array : np.ndarray
            2-D uint8 array (grayscale image).

        Returns
        -------
        list[Polyline]
            Each polyline is ``[(x1, y1), (x2, y2)]``.
        """
        if not _HAS_CV2:
            raise RuntimeError("opencv-python-headless is required")
        return self._detect_lines_from_gray(gray_array)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _render_fitz_page(self, fitz_page) -> "np.ndarray":
        """
        Render a PyMuPDF page to a grayscale NumPy array.

        PyMuPDF (fitz) fully rasterises all PDF content – vector paths,
        text, and embedded images – so this method works for both
        scanned-image PDFs and vector-drawn road-map PDFs.
        """
        scale = self.params.dpi / 72.0
        mat = _fitz.Matrix(scale, scale)
        pixmap = fitz_page.get_pixmap(matrix=mat, colorspace=_fitz.csGRAY)
        # samples is a bytes object: height × width × n_components
        arr = np.frombuffer(pixmap.samples, dtype=np.uint8)
        return arr.reshape(pixmap.height, pixmap.width).copy()

    def _render_page(self, page) -> "np.ndarray":
        """
        Render a pypdf page to a grayscale NumPy array.

        pypdf itself cannot render raster images; we extract the page's
        raw PDF bytes and re-open them via Pillow.  For scanned PDFs
        the page usually has exactly one embedded XObject image which
        we extract directly.  If that fails we fall back to a blank
        placeholder so the pipeline can continue.
        """
        # Try to extract embedded raster images from the page
        images = []
        if hasattr(page, "images"):
            try:
                images = list(page.images)
            except Exception:
                pass

        if images:
            # Use the first embedded image
            img_data = images[0].data
            pil_img = _PILImage.open(io.BytesIO(img_data)).convert("L")
            return np.array(pil_img, dtype=np.uint8)

        # Fallback: create a blank white image at the page's reported size
        width_pt  = float(page.mediabox.width)
        height_pt = float(page.mediabox.height)
        scale = self.params.dpi / 72.0  # 72 pt/inch
        w = max(1, int(width_pt * scale))
        h = max(1, int(height_pt * scale))
        log.debug("No embedded image found; using blank %dx%d canvas", w, h)
        return np.ones((h, w), dtype=np.uint8) * 255

    def _detect_lines(self, gray_array: "np.ndarray") -> List[Polyline]:
        """Run Canny + Hough on a grayscale uint8 array."""
        return self._detect_lines_from_gray(gray_array)

    def _detect_lines_from_gray(self, gray: "np.ndarray") -> List[Polyline]:
        """Core detection logic shared by ``_detect_lines`` and ``detect_lines_from_array``."""
        p = self.params

        edges = _cv2.Canny(gray, p.canny_low, p.canny_high)

        segments = _cv2.HoughLinesP(
            edges,
            rho=p.hough_rho,
            theta=p.hough_theta,
            threshold=p.hough_threshold,
            minLineLength=p.hough_min_line_length,
            maxLineGap=p.hough_max_line_gap,
        )

        polylines: List[Polyline] = []
        if segments is not None:
            for seg in segments:
                x1, y1, x2, y2 = seg[0]
                polylines.append([(float(x1), float(y1)), (float(x2), float(y2))])

        return polylines

    # ------------------------------------------------------------------
    # Coordinate helpers (pixel → geographic)
    # ------------------------------------------------------------------

    @staticmethod
    def pixel_to_geo(
        px: float,
        py: float,
        image_width: int,
        image_height: int,
        bbox: Tuple[float, float, float, float],
    ) -> Tuple[float, float]:
        """
        Map pixel coordinates to geographic coordinates via a bounding box.

        Parameters
        ----------
        px, py : float
            Pixel column and row (origin top-left).
        image_width, image_height : int
            Image dimensions in pixels.
        bbox : tuple[float, float, float, float]
            ``(min_x, min_y, max_x, max_y)`` in the target CRS.

        Returns
        -------
        tuple[float, float]
            ``(geo_x, geo_y)`` in the target CRS.
        """
        min_x, min_y, max_x, max_y = bbox
        geo_x = min_x + (px / image_width) * (max_x - min_x)
        # Image y-axis is top-to-bottom; geographic y-axis is bottom-to-top
        geo_y = max_y - (py / image_height) * (max_y - min_y)
        return geo_x, geo_y

    @staticmethod
    def polylines_to_geo(
        polylines: List[Polyline],
        image_width: int,
        image_height: int,
        bbox: Tuple[float, float, float, float],
    ) -> List[Polyline]:
        """
        Convert a list of pixel-space polylines to geographic coordinates.

        Parameters
        ----------
        polylines : list[Polyline]
        image_width, image_height : int
        bbox : tuple[float, float, float, float]
            ``(min_x, min_y, max_x, max_y)``

        Returns
        -------
        list[Polyline]
        """
        result = []
        for pl in polylines:
            geo_pl = [
                PdfImporter.pixel_to_geo(x, y, image_width, image_height, bbox)
                for x, y in pl
            ]
            result.append(geo_pl)
        return result
