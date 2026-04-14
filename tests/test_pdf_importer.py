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
    ClassifiedImportResult,
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


# ---------------------------------------------------------------------------
# ImportResult – polyline_pages tracking
# ---------------------------------------------------------------------------

class TestImportResultPageTracking:

    def test_polyline_pages_default_empty(self):
        r = ImportResult(pdf_path="x.pdf")
        assert r.polyline_pages == []

    def test_polyline_pages_length_matches_polylines(self):
        polylines = [[(0, 0), (1, 1)], [(2, 2), (3, 3)]]
        pages     = [0, 1]
        r = ImportResult(pdf_path="x.pdf", polylines=polylines, polyline_pages=pages)
        assert len(r.polyline_pages) == len(r.polylines)

    def test_polyline_pages_values(self):
        polylines = [[(0, 0), (1, 1)]] * 3
        pages     = [0, 0, 1]
        r = ImportResult(pdf_path="x.pdf", polylines=polylines, polyline_pages=pages)
        assert r.polyline_pages == [0, 0, 1]


# ---------------------------------------------------------------------------
# ClassifiedImportResult – dataclass properties
# ---------------------------------------------------------------------------

class TestClassifiedImportResult:

    def test_success_when_no_errors(self):
        r = ClassifiedImportResult(
            pdf_path="x.pdf",
            roads=[[(0, 0), (1, 1)]],
        )
        assert r.success
        assert r.road_count == 1
        assert r.station_count == 0
        assert r.dump_site_count == 0

    def test_failure_when_errors_present(self):
        r = ClassifiedImportResult(pdf_path="x.pdf", errors=["noe gikk galt"])
        assert not r.success

    def test_counts_match_list_lengths(self):
        r = ClassifiedImportResult(
            pdf_path="x.pdf",
            roads=[[(0, 0), (1, 1)], [(2, 2), (3, 3)]],
            stations=[(10.0, 20.0)],
            dump_sites=[[(5, 5), (6, 5), (6, 6), (5, 6)]],
        )
        assert r.road_count == 2
        assert r.station_count == 1
        assert r.dump_site_count == 1

    def test_default_page_lists_are_empty(self):
        r = ClassifiedImportResult(pdf_path="x.pdf")
        assert r.road_pages == []
        assert r.station_pages == []
        assert r.dump_site_pages == []


# ---------------------------------------------------------------------------
# DetectionParams – validation of new blob/area fields
# ---------------------------------------------------------------------------

class TestDetectionParamsNew:

    def test_default_station_params_are_valid(self):
        p = DetectionParams()
        p.validate()  # must not raise

    def test_invalid_station_area_order_raises(self):
        p = DetectionParams(station_min_area=500.0, station_max_area=100.0)
        with pytest.raises(ValueError, match="station_min_area"):
            p.validate()

    def test_zero_station_min_area_raises(self):
        p = DetectionParams(station_min_area=0.0)
        with pytest.raises(ValueError):
            p.validate()

    def test_invalid_circularity_raises(self):
        p = DetectionParams(station_circularity_min=0.0)
        with pytest.raises(ValueError, match="station_circularity_min"):
            p.validate()

    def test_circularity_above_one_raises(self):
        p = DetectionParams(station_circularity_min=1.1)
        with pytest.raises(ValueError):
            p.validate()

    def test_invalid_dump_site_area_order_raises(self):
        p = DetectionParams(dump_site_min_area=5000.0, dump_site_max_area=1000.0)
        with pytest.raises(ValueError, match="dump_site_min_area"):
            p.validate()

    def test_dump_site_max_aspect_below_one_raises(self):
        p = DetectionParams(dump_site_max_aspect=0.5)
        with pytest.raises(ValueError, match="dump_site_max_aspect"):
            p.validate()


# ---------------------------------------------------------------------------
# PdfImporter – detect_features_from_array
# ---------------------------------------------------------------------------

class TestDetectFeaturesFromArray:
    """Tests for the classified feature detection pipeline."""

    def setup_method(self):
        params = DetectionParams(
            canny_low=30,
            canny_high=100,
            hough_threshold=40,
            hough_min_line_length=20.0,
            hough_max_line_gap=5.0,
            station_min_area=30.0,
            station_max_area=1500.0,
            station_circularity_min=0.3,
            dump_site_min_area=1500.0,
            dump_site_max_area=100000.0,
            dump_site_max_aspect=6.0,
        )
        self.importer = PdfImporter(params)

    def test_blank_image_yields_no_features(self):
        img = _white_image()
        roads, stations, dump_sites = self.importer.detect_features_from_array(img)
        assert roads == []
        assert stations == []
        assert dump_sites == []

    def test_road_line_returned_as_road_not_station(self):
        """A long thin line must be detected as a road but not as a station."""
        img = _white_image(300, 300)
        _draw_line(img, 10, 150, 290, 150, thickness=3)
        roads, stations, _ = self.importer.detect_features_from_array(img)
        assert len(roads) >= 1
        # A long line is far too elongated to pass the circularity filter.
        assert stations == []

    def test_filled_circle_detected_as_station(self):
        """A filled circle (radius≈10 px) should be classified as a standplass."""
        img = _white_image(200, 200)
        cv2.circle(img, (100, 100), 10, 0, -1)  # filled black circle
        _, stations, _ = self.importer.detect_features_from_array(img)
        assert len(stations) >= 1
        # Centroid should be near (100, 100)
        cx, cy = stations[0]
        assert abs(cx - 100) < 5
        assert abs(cy - 100) < 5

    def test_filled_rectangle_detected_as_dump_site(self):
        """A filled rectangle should be classified as a velteplass."""
        img = _white_image(300, 300)
        cv2.rectangle(img, (80, 80), (180, 150), 0, -1)  # 100×70 filled rect
        _, _, dump_sites = self.importer.detect_features_from_array(img)
        assert len(dump_sites) >= 1
        # Each dump site is a polygon with at least 3 vertices.
        for poly in dump_sites:
            assert len(poly) >= 3

    def test_station_centroid_is_float_pair(self):
        """Station centroids must be pairs of floats."""
        img = _white_image(200, 200)
        cv2.circle(img, (80, 120), 10, 0, -1)
        _, stations, _ = self.importer.detect_features_from_array(img)
        for cx, cy in stations:
            assert isinstance(cx, float)
            assert isinstance(cy, float)

    def test_dump_site_polygon_vertices_are_float_pairs(self):
        """Dump-site polygon vertices must be pairs of floats."""
        img = _white_image(300, 300)
        cv2.rectangle(img, (50, 50), (200, 150), 0, -1)
        _, _, dump_sites = self.importer.detect_features_from_array(img)
        for poly in dump_sites:
            for x, y in poly:
                assert isinstance(x, float)
                assert isinstance(y, float)

    def test_elongated_rectangle_not_detected_as_dump_site(self):
        """A very elongated rectangle (aspect > 6) must be excluded."""
        img = _white_image(300, 300)
        # 260×10 rectangle → aspect ≈ 26
        cv2.rectangle(img, (10, 145), (270, 155), 0, -1)
        _, _, dump_sites = self.importer.detect_features_from_array(img)
        assert dump_sites == []


# ---------------------------------------------------------------------------
# PdfImporter – import_classified_file error handling
# ---------------------------------------------------------------------------

class TestImportClassifiedFileErrors:

    def setup_method(self):
        self.importer = PdfImporter()

    def test_missing_file_returns_error(self):
        result = self.importer.import_classified_file("/nonexistent/path/map.pdf")
        assert not result.success
        assert any("finnes ikke" in e for e in result.errors)

    def test_missing_file_has_zero_features(self):
        result = self.importer.import_classified_file("/nonexistent/path/map.pdf")
        assert result.road_count == 0
        assert result.station_count == 0
        assert result.dump_site_count == 0

    def test_import_classified_files_returns_one_result_per_path(self):
        paths = ["/no/file1.pdf", "/no/file2.pdf"]
        results = self.importer.import_classified_files(paths)
        assert len(results) == len(paths)
        for r in results:
            assert not r.success
