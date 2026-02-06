"""Face detection — locate faces in images."""

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .config import Config
from .utils import load_image

logger = logging.getLogger(__name__)

# Type alias: (top, right, bottom, left)
FaceLocation = Tuple[int, int, int, int]


def detect_faces(
    image: np.ndarray,
    config: Config,
) -> List[FaceLocation]:
    """Detect faces in a single RGB image.

    Returns a list of bounding boxes as (top, right, bottom, left).
    """
    import face_recognition

    locations = face_recognition.face_locations(
        image,
        number_of_times_to_upsample=config.upsample_times,
        model=config.detection_model,
    )
    return locations


def detect_faces_in_file(
    path: Path,
    config: Config,
) -> Tuple[np.ndarray | None, List[FaceLocation]]:
    """Load an image from *path* and return (image, face_locations).

    Returns ``(None, [])`` when the image cannot be loaded.
    """
    image = load_image(path)
    if image is None:
        return None, []
    locations = detect_faces(image, config)
    logger.debug("Found %d face(s) in %s", len(locations), path.name)
    return image, locations
