"""Unit tests for the organizer module."""

import shutil
from pathlib import Path

import pytest

from face_organizer.config import Config
from face_organizer.organizer import _sanitize, _unique_path, organize_photos


class TestSanitize:
    def test_normal_name(self):
        assert _sanitize("Alice") == "Alice"

    def test_special_characters(self):
        assert _sanitize("John/Doe") == "John_Doe"

    def test_empty_string(self):
        assert _sanitize("") == "unnamed"

    def test_spaces_preserved(self):
        assert _sanitize("Jane Doe") == "Jane Doe"


class TestUniquePath:
    def test_no_conflict(self, tmp_path):
        p = tmp_path / "photo.jpg"
        assert _unique_path(p) == p

    def test_conflict_adds_suffix(self, tmp_path):
        p = tmp_path / "photo.jpg"
        p.write_text("existing")
        result = _unique_path(p)
        assert result.name == "photo_1.jpg"

    def test_multiple_conflicts(self, tmp_path):
        (tmp_path / "photo.jpg").write_text("existing")
        (tmp_path / "photo_1.jpg").write_text("existing")
        result = _unique_path(tmp_path / "photo.jpg")
        assert result.name == "photo_2.jpg"


class TestOrganizePhotos:
    def test_copy_action(self, tmp_path):
        # Create a fake source image
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        img = src_dir / "pic.jpg"
        img.write_text("image data")

        out_dir = tmp_path / "out"
        cfg = Config(output_dir=out_dir, file_action="copy")
        groups = {"Alice": [img]}
        result = organize_photos(groups, cfg)

        assert "Alice" in result
        assert (out_dir / "Alice" / "pic.jpg").exists()
        # Original still exists (copy)
        assert img.exists()

    def test_move_action(self, tmp_path):
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        img = src_dir / "pic.jpg"
        img.write_text("image data")

        out_dir = tmp_path / "out"
        cfg = Config(output_dir=out_dir, file_action="move")
        groups = {"Bob": [img]}
        result = organize_photos(groups, cfg)

        assert (out_dir / "Bob" / "pic.jpg").exists()
        # Original should be gone (move)
        assert not img.exists()
