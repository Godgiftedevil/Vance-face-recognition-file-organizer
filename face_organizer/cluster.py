"""Face clustering — group unknown face encodings into identity clusters."""

import logging
from typing import Dict, List

import numpy as np
from sklearn.cluster import DBSCAN

from .config import Config

logger = logging.getLogger(__name__)


def cluster_encodings(
    encodings: List[np.ndarray],
    config: Config,
) -> np.ndarray:
    """Cluster face encodings with DBSCAN.

    Parameters
    ----------
    encodings : list of np.ndarray
        128-d face embedding vectors.
    config : Config
        Pipeline configuration (uses *cluster_threshold* and
        *cluster_min_samples*).

    Returns
    -------
    np.ndarray
        Integer label for each encoding.  ``-1`` means the face could not
        be assigned to any cluster (noise/outlier).
    """
    if len(encodings) == 0:
        return np.array([], dtype=int)

    data = np.array(encodings)
    dbscan = DBSCAN(
        eps=config.cluster_threshold,
        min_samples=config.cluster_min_samples,
        metric="euclidean",
    )
    labels = dbscan.fit_predict(data)
    n_clusters = len(set(labels) - {-1})
    n_noise = int(np.sum(labels == -1))
    logger.info(
        "Clustering: %d clusters, %d noise faces out of %d total",
        n_clusters,
        n_noise,
        len(encodings),
    )
    return labels


def group_by_label(
    labels: np.ndarray,
    items: list,
) -> Dict[int, list]:
    """Group *items* by their cluster *labels*.

    Returns a dict mapping ``label → [items …]``.
    """
    groups: Dict[int, list] = {}
    for label, item in zip(labels, items):
        label = int(label)
        groups.setdefault(label, []).append(item)
    return groups
