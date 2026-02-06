"""End-to-end pipeline — detect → encode → cluster → label → organize."""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .cluster import cluster_encodings, group_by_label
from .config import Config
from .detector import FaceLocation, detect_faces_in_file
from .encoder import encode_faces
from .labeler import assign_labels
from .organizer import organize_photos
from .utils import largest_face, list_images

logger = logging.getLogger(__name__)


def scan_photos(
    config: Config,
) -> Tuple[List[Path], List[np.ndarray], List[FaceLocation]]:
    """Scan all images, detect faces, and compute encodings.

    For images with multiple faces the *largest* face is chosen so that
    each photo is associated with a single primary identity.

    Returns
    -------
    photo_paths : list of Path
        Images where at least one face was detected.
    encodings : list of np.ndarray
        128-d encoding per photo.
    locations : list of FaceLocation
        Bounding box per photo.
    """
    images = list_images(Path(config.input_dir), config.image_extensions)
    logger.info("Found %d image(s) in %s", len(images), config.input_dir)

    photo_paths: List[Path] = []
    all_encodings: List[np.ndarray] = []
    all_locations: List[FaceLocation] = []

    for idx, img_path in enumerate(images, 1):
        if idx % 50 == 0 or idx == len(images):
            logger.info("Processing %d / %d …", idx, len(images))

        image, locs = detect_faces_in_file(img_path, config)
        if image is None or not locs:
            continue

        encs = encode_faces(image, locs, config)
        if not encs:
            continue

        # Strategy: pick the largest face as the "primary" face.
        best_idx = largest_face(locs)
        if best_idx is None:
            continue

        photo_paths.append(img_path)
        all_encodings.append(encs[best_idx])
        all_locations.append(locs[best_idx])

    logger.info(
        "Detected faces in %d / %d image(s)", len(photo_paths), len(images)
    )
    return photo_paths, all_encodings, all_locations


def run_pipeline(
    config: Config,
    interactive: bool = True,
) -> Dict[str, Path]:
    """Execute the full organize-by-face pipeline.

    Parameters
    ----------
    config : Config
        Pipeline configuration.
    interactive : bool
        Prompt user for face labels when *True*.

    Returns
    -------
    dict
        ``label → output_directory`` for each created group.
    """
    config.validate()

    # 1. Scan & encode
    photo_paths, encodings, _ = scan_photos(config)
    if not encodings:
        logger.warning("No faces detected — nothing to organize.")
        return {}

    # 2. Cluster
    labels = cluster_encodings(encodings, config)

    # 3. Group photos by cluster label
    groups = group_by_label(labels, photo_paths)

    # 4. Assign human-readable names
    named_groups = assign_labels(groups, config, interactive=interactive)

    # 5. Organize files
    result = organize_photos(named_groups, config)
    logger.info("Done — organized into %d group(s).", len(result))
    return result
