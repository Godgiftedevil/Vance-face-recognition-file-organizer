"""Centralized configuration for Face Organizer."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Config:
    """All tuneable knobs for the face-organizer pipeline."""

    # ── Paths ───────────────────────────────────────────────────────────
    input_dir: Path = Path("photos")
    output_dir: Path = Path("organized")

    # ── Detection ───────────────────────────────────────────────────────
    # face_recognition uses HOG (fast/CPU) or CNN (accurate/GPU).
    detection_model: str = "hog"  # "hog" | "cnn"
    # Up-sample image N times to find smaller faces (0 = no up-sample).
    upsample_times: int = 1

    # ── Encoding ────────────────────────────────────────────────────────
    # Number of re-samples when computing the 128-d face encoding.
    encoding_num_jitters: int = 1  # higher → slower but more accurate
    encoding_model: str = "small"  # "small" (5-point) | "large" (68-point)

    # ── Clustering ──────────────────────────────────────────────────────
    # Distance threshold – lower = stricter matching.
    cluster_threshold: float = 0.6
    # DBSCAN min_samples: minimum faces to form a cluster.
    cluster_min_samples: int = 2

    # ── Organizer ───────────────────────────────────────────────────────
    # "copy" keeps originals; "move" relocates them.
    file_action: str = "copy"  # "copy" | "move"
    unknown_label: str = "unknown"

    # ── Performance ─────────────────────────────────────────────────────
    batch_size: int = 32
    max_workers: int = 4

    # ── Supported image extensions ──────────────────────────────────────
    image_extensions: List[str] = field(
        default_factory=lambda: [
            ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp",
        ]
    )

    # ── Labeling ────────────────────────────────────────────────────────
    labels_file: Path = Path("face_labels.json")

    def validate(self) -> None:
        """Raise *ValueError* if any setting is invalid."""
        if self.detection_model not in ("hog", "cnn"):
            raise ValueError(
                f"detection_model must be 'hog' or 'cnn', "
                f"got '{self.detection_model}'"
            )
        if self.file_action not in ("copy", "move"):
            raise ValueError(
                f"file_action must be 'copy' or 'move', "
                f"got '{self.file_action}'"
            )
        if self.cluster_threshold <= 0 or self.cluster_threshold > 1:
            raise ValueError(
                f"cluster_threshold must be in (0, 1], "
                f"got {self.cluster_threshold}"
            )
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
