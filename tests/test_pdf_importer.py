# -*- coding: utf-8 -*-
"""
test_pdf_importer.py – Unit tests for PdfImporter (STEG 3)

All tests use plain Python / NumPy / OpenCV – no QGIS runtime required.
Synthetic images are constructed with NumPy so real PDF files are not needed.
"""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Skip the entire module when OpenCV / NumPy are not available
# ---------------------------------------------------------------------------

numpy = pytest.importorskip("numpy")
cv2   = pytest.importorskip("cv2")

from processors.pdf_importer import (
    DetectionParams,
    ImportResult,
    PdfImporter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _white_image(h: int = 200, w: int = 200) -> "numpy.ndarray":
    """Return a blank white uint8 grayscale image."""
    return numpy.ones((h, w), dtype=numpy.uint8) * 255


def _draw_line(img, x1, y1, x2, y2, thickness=2):
    """Draw a black line on the image in-place and return it."""
    cv2.line(img, (x1, y1), (x2, y2), 0, thickness)
    return img


# ---------------------------------------------------------------------------
# DetectionParams
# ---------------------------------------------------------------------------

class TestDetectionParams:

    def test_defaults_are_valid(self):
        p = DetectionParams()
        p.validate()  # must not raise

    def test_invalid_canny_order_raises(self):
        p = DetectionParams(canny_low=150, canny_high=50)
        with pytest.raises(ValueError, match="Canny"):
            p.validate()

    def test_zero_min_line_length_raises(self):
        p = DetectionParams(hough_min_line_length=0)
        with pytest.raises(ValueError, match="hough_min_line_length"):
            p.validate()

    def test_zero_dpi_raises(self):
        p = DetectionParams(dpi=0)
        with pytest.raises(ValueError, match="dpi"):
            p.validate()

    def test_equal_canny_thresholds_raise(self):
        p = DetectionParams(canny_low=100, canny_high=100)
        with pytest.raises(ValueError):
            p.validate()


# ---------------------------------------------------------------------------
# ImportResult
# ---------------------------------------------------------------------------

class TestImportResult:

    def test_success_when_no_errors(self):
        r = ImportResult(pdf_path="/some/file.pdf", polylines=[[(0, 0), (1, 1)]])
        assert r.success
        assert r.line_count == 1

    def test_failure_when_errors_present(self):
        r = ImportResult(pdf_path="/some/file.pdf", errors=["noe gikk galt"])
        assert not r.success
        assert r.line_count == 0

    def test_line_count_matches_polylines(self):
        r = ImportResult(
            pdf_path="x.pdf",
            polylines=[[(0, 0), (1, 1)], [(2, 2), (3, 3)], [(4, 4), (5, 5)]],
        )
        assert r.line_count == 3


# ---------------------------------------------------------------------------
# PdfImporter – detect_lines_from_array
# ---------------------------------------------------------------------------

class TestDetectLinesFromArray:

    def setup_method(self):
        # Sensitive params to reliably detect thick lines in synthetic images
        params = DetectionParams(
            canny_low=30,
            canny_high=100,
            hough_threshold=40,
            hough_min_line_length=20.0,
            hough_max_line_gap=5.0,
        )
        self.importer = PdfImporter(params)

    def test_blank_image_yields_no_lines(self):
        img = _white_image()
        lines = self.importer.detect_lines_from_array(img)
        assert lines == []

    def test_single_horizontal_line_detected(self):
        img = _white_image(200, 200)
        _draw_line(img, 10, 100, 190, 100, thickness=3)
        lines = self.importer.detect_lines_from_array(img)
        assert len(lines) >= 1

    def test_single_vertical_line_detected(self):
        img = _white_image(200, 200)
        _draw_line(img, 100, 10, 100, 190, thickness=3)
        lines = self.importer.detect_lines_from_array(img)
        assert len(lines) >= 1

    def test_diagonal_line_detected(self):
        img = _white_image(200, 200)
        _draw_line(img, 10, 10, 190, 190, thickness=3)
        lines = self.importer.detect_lines_from_array(img)
        assert len(lines) >= 1

    def test_multiple_lines_detected(self):
        img = _white_image(300, 300)
        _draw_line(img, 10, 50, 290, 50, thickness=3)
        _draw_line(img, 10, 150, 290, 150, thickness=3)
        _draw_line(img, 10, 250, 290, 250, thickness=3)
        lines = self.importer.detect_lines_from_array(img)
        assert len(lines) >= 2

    def test_output_is_list_of_two_point_polylines(self):
        img = _white_image(200, 200)
        _draw_line(img, 10, 100, 190, 100, thickness=3)
        lines = self.importer.detect_lines_from_array(img)
        for pl in lines:
            assert len(pl) == 2
            for pt in pl:
                assert len(pt) == 2
                assert isinstance(pt[0], float)
                assert isinstance(pt[1], float)

    def test_short_segment_below_min_length_not_detected(self):
        """A 5-pixel line should not be detected with min_line_length=20."""
        img = _white_image(200, 200)
        _draw_line(img, 100, 100, 105, 100, thickness=2)
        lines = self.importer.detect_lines_from_array(img)
        assert lines == []


# ---------------------------------------------------------------------------
# PdfImporter – import_file error handling (no real PDF needed)
# ---------------------------------------------------------------------------

class TestImportFileErrors:

    def setup_method(self):
        self.importer = PdfImporter()

    def test_missing_file_returns_error(self):
        result = self.importer.import_file("/nonexistent/path/map.pdf")
        assert not result.success
        assert any("finnes ikke" in e for e in result.errors)

    def test_missing_file_has_zero_polylines(self):
        result = self.importer.import_file("/nonexistent/path/map.pdf")
        assert result.line_count == 0

    def test_import_files_returns_one_result_per_path(self):
        paths = ["/no/file1.pdf", "/no/file2.pdf", "/no/file3.pdf"]
        results = self.importer.import_files(paths)
        assert len(results) == len(paths)
        for r in results:
            assert not r.success


# ---------------------------------------------------------------------------
# PdfImporter – coordinate conversion
# ---------------------------------------------------------------------------

class TestPixelToGeo:

    def test_top_left_maps_to_min_x_max_y(self):
        geo = PdfImporter.pixel_to_geo(0, 0, 100, 100, (0.0, 0.0, 1.0, 1.0))
        assert geo == pytest.approx((0.0, 1.0))

    def test_bottom_right_maps_to_max_x_min_y(self):
        geo = PdfImporter.pixel_to_geo(100, 100, 100, 100, (0.0, 0.0, 1.0, 1.0))
        assert geo == pytest.approx((1.0, 0.0))

    def test_center_maps_correctly(self):
        geo = PdfImporter.pixel_to_geo(50, 50, 100, 100, (0.0, 0.0, 100.0, 100.0))
        assert geo == pytest.approx((50.0, 50.0))

    def test_real_world_bbox(self):
        # A 1000×800 image covering UTM bbox
        bbox = (570000.0, 6750000.0, 571000.0, 6751000.0)
        geo = PdfImporter.pixel_to_geo(0, 0, 1000, 800, bbox)
        assert geo == pytest.approx((570000.0, 6751000.0))
        geo2 = PdfImporter.pixel_to_geo(1000, 800, 1000, 800, bbox)
        assert geo2 == pytest.approx((571000.0, 6750000.0))

    def test_polylines_to_geo_converts_all_points(self):
        polylines = [
            [(0.0, 0.0), (100.0, 100.0)],
            [(50.0, 0.0), (50.0, 100.0)],
        ]
        bbox = (0.0, 0.0, 1.0, 1.0)
        result = PdfImporter.polylines_to_geo(polylines, 100, 100, bbox)
        assert len(result) == 2
        assert result[0][0] == pytest.approx((0.0, 1.0))    # top-left pixel
        assert result[0][1] == pytest.approx((1.0, 0.0))    # bottom-right pixel
        assert result[1][0] == pytest.approx((0.5, 1.0))

    def test_polylines_to_geo_empty_input(self):
        result = PdfImporter.polylines_to_geo([], 100, 100, (0, 0, 1, 1))
        assert result == []
