"""
cv/gaze.py — Iris-tracking and gaze vector utilities.

Provides higher-level functions that build on the raw per-frame gaze vectors
produced by ``cv/session.py`` to compute session-level statistics used for
Mahalanobis matching.
"""

import logging

import numpy as np

log = logging.getLogger(__name__)


def session_gaze_stats(gaze_vecs: np.ndarray) -> dict:
    """
    Compute the session-level mean gaze vector.

    Parameters
    ----------
    gaze_vecs : np.ndarray, shape (N, 3)
        Per-frame gaze vectors [gx, gy, iod_ratio].

    Returns
    -------
    dict with keys:
        "mean"   : np.ndarray (3,)
        "std"    : np.ndarray (3,)
        "n"      : int
    """
    if gaze_vecs.ndim != 2 or gaze_vecs.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) array, got {gaze_vecs.shape}")

    return {
        "mean": gaze_vecs.mean(axis=0),
        "std":  gaze_vecs.std(axis=0),
        "n":    len(gaze_vecs),
    }
