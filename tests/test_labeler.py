"""Unit tests for the labeler module."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from face_organizer.config import Config
from face_organizer.labeler import assign_labels, load_labels, save_labels


class TestLoadSaveLabels:
    def test_round_trip(self, tmp_path):
        cfg = Config(labels_file=tmp_path / "labels.json")
        save_labels({"0": "Alice", "1": "Bob"}, cfg)
        loaded = load_labels(cfg)
        assert loaded == {"0": "Alice", "1": "Bob"}

    def test_load_missing_file(self, tmp_path):
        cfg = Config(labels_file=tmp_path / "nonexistent.json")
        assert load_labels(cfg) == {}


class TestAssignLabels:
    def test_non_interactive(self, tmp_path):
        cfg = Config(labels_file=tmp_path / "labels.json")
        groups = {0: [Path("a.jpg")], 1: [Path("b.jpg")], -1: [Path("c.jpg")]}
        result = assign_labels(groups, cfg, interactive=False)
        assert "person_0" in result
        assert "person_1" in result
        assert "unknown" in result

    def test_uses_saved_labels(self, tmp_path):
        labels_path = tmp_path / "labels.json"
        labels_path.write_text(json.dumps({"0": "Alice"}))
        cfg = Config(labels_file=labels_path)
        groups = {0: [Path("a.jpg")]}
        result = assign_labels(groups, cfg, interactive=False)
        assert "Alice" in result

    def test_interactive_prompt(self, tmp_path):
        cfg = Config(labels_file=tmp_path / "labels.json")
        groups = {0: [Path("a.jpg")]}
        with patch("builtins.input", return_value="Charlie"):
            result = assign_labels(groups, cfg, interactive=True)
        assert "Charlie" in result
