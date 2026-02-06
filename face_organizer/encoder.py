"""Face encoding — produce 128-d embeddings for detected faces."""

import logging
from typing import List, Tuple

import numpy as np

from .config import Config
from .detector import FaceLocation

logger = logging.getLogger(__name__)


def encode_faces(
    image: np.ndarray,
    locations: List[FaceLocation],
    config: Config,
) -> List[np.ndarray]:
    """Compute 128-dimensional face encodings for each detected face.

    Parameters
    ----------
    image : np.ndarray
        RGB image array.
    locations : list of FaceLocation
        Bounding boxes from :func:`detector.detect_faces`.
    config : Config
        Pipeline configuration.

    Returns
    -------
    list of np.ndarray
        One 128-d vector per face, in the same order as *locations*.
    """
    import face_recognition

    encodings = face_recognition.face_encodings(
        image,
        known_face_locations=locations,
        num_jitters=config.encoding_num_jitters,
        model=config.encoding_model,
    )
    return encodings


def compute_distance(
    encoding_a: np.ndarray,
    encoding_b: np.ndarray,
) -> float:
    """Return the Euclidean distance between two face encodings."""
    return float(np.linalg.norm(encoding_a - encoding_b))


def compare_faces(
    known_encodings: List[np.ndarray],
    face_encoding: np.ndarray,
    tolerance: float = 0.6,
) -> Tuple[List[bool], List[float]]:
    """Compare *face_encoding* against a list of *known_encodings*.

    Returns
    -------
    matches : list of bool
        True where distance < *tolerance*.
    distances : list of float
        Euclidean distances.
    """
    if not known_encodings:
        return [], []
    distances = [compute_distance(k, face_encoding) for k in known_encodings]
    matches = [d < tolerance for d in distances]
    return matches, distances
