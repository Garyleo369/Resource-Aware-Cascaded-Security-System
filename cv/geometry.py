"""
cv/geometry.py — Facial geometry ratio extraction and statistics.

Works with the per-frame geometry vectors [iod_norm, nose_chin_ratio,
lip_curvature] produced by cv/session.py and computes session-level
statistics for matching.
"""

import logging

import numpy as np

log = logging.getLogger(__name__)


def session_geometry_stats(geom_vecs: np.ndarray) -> dict:
    """
    Compute session-level mean geometry vector.

    Parameters
    ----------
    geom_vecs : np.ndarray, shape (N, 3)
        Per-frame geometry vectors.

    Returns
    -------
    dict with keys "mean", "std", "n".
    """
    if geom_vecs.ndim != 2 or geom_vecs.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) array, got {geom_vecs.shape}")

    return {
        "mean": geom_vecs.mean(axis=0),
        "std":  geom_vecs.std(axis=0),
        "n":    len(geom_vecs),
    }
