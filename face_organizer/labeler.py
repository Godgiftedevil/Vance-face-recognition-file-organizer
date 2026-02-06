"""Interactive face labeling — ask the user to name unknown clusters."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .config import Config

logger = logging.getLogger(__name__)


def load_labels(config: Config) -> Dict[str, str]:
    """Load previously saved cluster→name mapping from disk."""
    path = Path(config.labels_file)
    if path.is_file():
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            logger.info("Loaded %d label(s) from %s", len(data), path)
            return data
    return {}


def save_labels(labels: Dict[str, str], config: Config) -> None:
    """Persist cluster→name mapping to disk."""
    path = Path(config.labels_file)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(labels, fh, indent=2)
    logger.info("Saved %d label(s) to %s", len(labels), path)


def prompt_for_label(
    cluster_id: int,
    sample_paths: List[Path],
    existing_labels: Dict[str, str],
) -> str:
    """Interactively ask the user to label a face cluster.

    Parameters
    ----------
    cluster_id : int
        Numeric cluster identifier.
    sample_paths : list of Path
        A few example image paths containing this face.
    existing_labels : dict
        Previously assigned labels (for display).

    Returns
    -------
    str
        The user-chosen label (or ``"person_<N>"`` when skipped).
    """
    print(f"\n{'='*50}")
    print(f"Cluster #{cluster_id}")
    print(f"Sample images ({len(sample_paths)} shown):")
    for p in sample_paths[:5]:
        print(f"  • {p}")
    if existing_labels:
        print(f"Known labels so far: {list(existing_labels.values())}")
    print("Enter a name for this person (or press Enter to skip):")
    try:
        name = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        name = ""
    if not name:
        name = f"person_{cluster_id}"
    return name


def assign_labels(
    groups: Dict[int, List[Path]],
    config: Config,
    interactive: bool = True,
) -> Dict[str, List[Path]]:
    """Map numeric cluster IDs to human-readable labels.

    Parameters
    ----------
    groups : dict
        ``cluster_id → [image_paths …]`` from clustering.
    config : Config
        Pipeline configuration.
    interactive : bool
        When *True*, prompt the user for labels.  When *False*, use
        ``"person_<N>"`` or ``unknown`` automatically.

    Returns
    -------
    dict
        ``label_string → [image_paths …]``
    """
    saved = load_labels(config)
    labeled: Dict[str, List[Path]] = {}

    for cluster_id, paths in sorted(groups.items()):
        key = str(cluster_id)
        if cluster_id == -1:
            label = config.unknown_label
        elif key in saved:
            label = saved[key]
        elif interactive:
            label = prompt_for_label(cluster_id, paths, saved)
            saved[key] = label
        else:
            label = f"person_{cluster_id}"
            saved[key] = label

        labeled.setdefault(label, []).extend(paths)

    save_labels(saved, config)
    return labeled
