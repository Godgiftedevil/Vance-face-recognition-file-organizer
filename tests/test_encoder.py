"""Unit tests for the encoder module (mocked — no dlib needed)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from face_organizer.encoder import compare_faces, compute_distance


class TestComputeDistance:
    def test_same_vector(self):
        v = np.ones(128)
        assert compute_distance(v, v) == pytest.approx(0.0)

    def test_known_distance(self):
        a = np.zeros(128)
        b = np.ones(128)
        expected = np.sqrt(128)
        assert compute_distance(a, b) == pytest.approx(expected)


class TestCompareFaces:
    def test_empty_known(self):
        matches, dists = compare_faces([], np.ones(128))
        assert matches == []
        assert dists == []

    def test_match_and_non_match(self):
        known = [np.zeros(128), np.ones(128) * 100]
        face = np.zeros(128)
        matches, dists = compare_faces(known, face, tolerance=0.6)
        assert matches[0] is True
        assert matches[1] is False
        assert dists[0] == pytest.approx(0.0)

    def test_tolerance_boundary(self):
        a = np.zeros(128)
        b = np.full(128, 0.05)  # distance = sqrt(128 * 0.0025) ≈ 0.566
        matches, dists = compare_faces([a], b, tolerance=0.6)
        assert matches[0] is True
