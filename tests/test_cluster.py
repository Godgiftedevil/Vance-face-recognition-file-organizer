"""Unit tests for clustering module."""

import numpy as np
import pytest

from face_organizer.cluster import cluster_encodings, group_by_label
from face_organizer.config import Config


class TestClusterEncodings:
    def test_empty_input(self):
        cfg = Config()
        labels = cluster_encodings([], cfg)
        assert len(labels) == 0

    def test_two_tight_clusters(self):
        """Two groups of near-identical 128-d vectors should form 2 clusters."""
        rng = np.random.RandomState(42)
        cluster_a = [rng.randn(128) * 0.01 for _ in range(5)]
        cluster_b = [rng.randn(128) * 0.01 + 10 for _ in range(5)]
        encodings = cluster_a + cluster_b
        cfg = Config(cluster_threshold=0.6, cluster_min_samples=2)
        labels = cluster_encodings(encodings, cfg)
        unique = set(labels) - {-1}
        assert len(unique) == 2

    def test_all_noise_when_spread(self):
        """Widely spaced points with high min_samples → all noise."""
        rng = np.random.RandomState(0)
        encodings = [rng.randn(128) * 100 for _ in range(5)]
        cfg = Config(cluster_threshold=0.01, cluster_min_samples=5)
        labels = cluster_encodings(encodings, cfg)
        assert all(l == -1 for l in labels)


class TestGroupByLabel:
    def test_basic(self):
        labels = np.array([0, 1, 0, -1])
        items = ["a", "b", "c", "d"]
        groups = group_by_label(labels, items)
        assert groups[0] == ["a", "c"]
        assert groups[1] == ["b"]
        assert groups[-1] == ["d"]

    def test_empty(self):
        groups = group_by_label(np.array([]), [])
        assert groups == {}
