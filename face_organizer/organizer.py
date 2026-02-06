"""Photo organizer — copy / move images into per-person folders."""

import logging
import shutil
from pathlib import Path
from typing import Dict, List

from .config import Config

logger = logging.getLogger(__name__)


def organize_photos(
    groups: Dict[str, List[Path]],
    config: Config,
) -> Dict[str, Path]:
    """Create per-person directories and copy/move photos.

    Parameters
    ----------
    groups : dict
        Mapping of ``person_label → [image_paths …]``.
    config : Config
        Pipeline configuration (uses *output_dir* and *file_action*).

    Returns
    -------
    dict
        Mapping of ``person_label → created_directory``.
    """
    created: Dict[str, Path] = {}
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    for label, paths in groups.items():
        person_dir = output / _sanitize(label)
        person_dir.mkdir(parents=True, exist_ok=True)
        created[label] = person_dir

        for src in paths:
            dst = person_dir / src.name
            dst = _unique_path(dst)
            if config.file_action == "copy":
                shutil.copy2(str(src), str(dst))
                logger.debug("Copied %s → %s", src, dst)
            else:
                shutil.move(str(src), str(dst))
                logger.debug("Moved  %s → %s", src, dst)

        logger.info(
            "Organized %d photo(s) into '%s/'", len(paths), person_dir.name
        )
    return created


def _sanitize(name: str) -> str:
    """Sanitize a label for use as a directory name."""
    safe = "".join(c if c.isalnum() or c in (" ", "_", "-") else "_" for c in name)
    return safe.strip() or "unnamed"


def _unique_path(path: Path) -> Path:
    """Append a numeric suffix when *path* already exists."""
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    counter = 1
    while True:
        candidate = path.with_name(f"{stem}_{counter}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1
