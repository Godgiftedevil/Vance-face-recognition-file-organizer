"""Unit tests for Config validation."""

import pytest

from face_organizer.config import Config


class TestConfigDefaults:
    def test_default_detection_model(self):
        cfg = Config()
        assert cfg.detection_model == "hog"

    def test_default_file_action(self):
        cfg = Config()
        assert cfg.file_action == "copy"

    def test_default_cluster_threshold(self):
        cfg = Config()
        assert cfg.cluster_threshold == 0.6


class TestConfigValidation:
    def test_valid_config_passes(self):
        Config().validate()  # should not raise

    def test_invalid_detection_model(self):
        cfg = Config(detection_model="invalid")
        with pytest.raises(ValueError, match="detection_model"):
            cfg.validate()

    def test_invalid_file_action(self):
        cfg = Config(file_action="delete")
        with pytest.raises(ValueError, match="file_action"):
            cfg.validate()

    def test_threshold_zero(self):
        cfg = Config(cluster_threshold=0)
        with pytest.raises(ValueError, match="cluster_threshold"):
            cfg.validate()

    def test_threshold_above_one(self):
        cfg = Config(cluster_threshold=1.5)
        with pytest.raises(ValueError, match="cluster_threshold"):
            cfg.validate()

    def test_threshold_one_is_valid(self):
        cfg = Config(cluster_threshold=1.0)
        cfg.validate()  # should not raise

    def test_invalid_batch_size(self):
        cfg = Config(batch_size=0)
        with pytest.raises(ValueError, match="batch_size"):
            cfg.validate()
