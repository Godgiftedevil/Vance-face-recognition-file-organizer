"""Utility helpers — image I/O, validation, EXIF extraction."""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def list_images(directory: Path, extensions: List[str]) -> List[Path]:
    """Return sorted list of image paths inside *directory* (recursive)."""
    images: List[Path] = []
    if not directory.is_dir():
        logger.warning("Directory does not exist: %s", directory)
        return images
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in extensions:
            images.append(path)
    return images


def load_image(path: Path) -> Optional[np.ndarray]:
    """Load an image file into a NumPy RGB array.

    Returns *None* when the file cannot be decoded.
    """
    try:
        from PIL import Image  # lazy import to keep startup fast

        img = Image.open(path)
        img = img.convert("RGB")
        return np.array(img)
    except Exception:
        logger.warning("Could not load image: %s", path)
        return None


def get_image_dimensions(image: np.ndarray) -> Tuple[int, int]:
    """Return (width, height) of a NumPy image array."""
    h, w = image.shape[:2]
    return w, h


def face_area(location: Tuple[int, int, int, int]) -> int:
    """Compute pixel area of a face bounding box (top, right, bottom, left)."""
    top, right, bottom, left = location
    return max(0, right - left) * max(0, bottom - top)


def largest_face(
    locations: List[Tuple[int, int, int, int]],
) -> Optional[int]:
    """Return the index of the largest face by bounding-box area.

    Returns *None* when *locations* is empty.
    """
    if not locations:
        return None
    return max(range(len(locations)), key=lambda i: face_area(locations[i]))
