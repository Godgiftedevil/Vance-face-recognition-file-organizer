"""Unit tests for utility functions."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from face_organizer.utils import (
    face_area,
    get_image_dimensions,
    largest_face,
    list_images,
)


class TestListImages:
    def test_lists_jpg_files(self, tmp_path):
        (tmp_path / "a.jpg").write_text("fake")
        (tmp_path / "b.png").write_text("fake")
        (tmp_path / "c.txt").write_text("not an image")
        result = list_images(tmp_path, [".jpg", ".png"])
        assert len(result) == 2

    def test_empty_directory(self, tmp_path):
        result = list_images(tmp_path, [".jpg"])
        assert result == []

    def test_nonexistent_directory(self):
        result = list_images(Path("/nonexistent"), [".jpg"])
        assert result == []

    def test_recursive(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "deep.jpg").write_text("fake")
        result = list_images(tmp_path, [".jpg"])
        assert len(result) == 1

    def test_sorted_output(self, tmp_path):
        (tmp_path / "b.jpg").write_text("fake")
        (tmp_path / "a.jpg").write_text("fake")
        result = list_images(tmp_path, [".jpg"])
        assert result[0].name == "a.jpg"


class TestFaceArea:
    def test_normal_box(self):
        # top=10, right=110, bottom=60, left=10 → 100 × 50
        assert face_area((10, 110, 60, 10)) == 5000

    def test_zero_area(self):
        assert face_area((0, 0, 0, 0)) == 0


class TestLargestFace:
    def test_single(self):
        assert largest_face([(0, 100, 100, 0)]) == 0

    def test_multiple(self):
        locs = [(0, 50, 50, 0), (0, 200, 200, 0)]
        assert largest_face(locs) == 1

    def test_empty(self):
        assert largest_face([]) is None


class TestGetImageDimensions:
    def test_dimensions(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        w, h = get_image_dimensions(img)
        assert w == 640
        assert h == 480
