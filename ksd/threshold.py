"""
ksd/threshold.py — Dynamic thresholding for KSD anomaly scores.

Maintains a rolling window of the last N scores and computes an adaptive
threshold as:

    threshold = mean(window) + SIGMA_FACTOR * std(window)

A score that exceeds this threshold triggers the CV module.
"""

import logging
from collections import deque

import numpy as np

log = logging.getLogger(__name__)

WINDOW_SIZE       = 20      # number of historical scores to keep
SIGMA_FACTOR      = 2.0     # how many standard deviations above mean (raised to reduce false positives)
ABS_THRESHOLD     = 0.75    # hard floor: always trigger above this score (raised to reduce false positives)


class DynamicThreshold:
    """
    Rolling adaptive threshold for anomaly score sequences.

    Usage
    -----
    >>> dt = DynamicThreshold()
    >>> triggered = dt.update(score)
    """

    def __init__(self,
                 window_size: int = WINDOW_SIZE,
                 sigma: float = SIGMA_FACTOR,
                 absolute_threshold: float = ABS_THRESHOLD):
        self._window    = deque(maxlen=window_size)
        self._sigma     = sigma
        self._abs_thr   = absolute_threshold

    @property
    def threshold(self) -> float | None:
        """Current threshold value, or None if not enough data yet."""
        if len(self._window) < 2:
            return None
        arr = np.array(self._window)
        return float(arr.mean() + self._sigma * arr.std())

    def update(self, score: float) -> bool:
        """
        Add a new score and test whether it exceeds the current threshold.

        The threshold is evaluated from the *previous* window (before the new
        score is included), so that an anomalous score cannot inflate the
        baseline it is being tested against.

        Parameters
        ----------
        score : float
            Anomaly score in [0, 1].

        Returns
        -------
        bool — True if the score exceeds the threshold and CV should trigger.
        """
        # Capture threshold from historical scores BEFORE adding current score
        thr = self.threshold
        self._window.append(score)
        if thr is None:
            # Still no baseline: only the hard absolute floor applies
            triggered = score > self._abs_thr
        else:
            # Trigger on adaptive OR absolute threshold, whichever fires first
            triggered = score > thr or score > self._abs_thr
        if triggered:
            log.warning(
                "⚠  KSD threshold exceeded  score=%.4f  threshold=%.4f",
                score, thr
            )
        else:
            log.debug("KSD score=%.4f  threshold=%.4f", score, thr)

        return triggered

    def stats(self) -> dict:
        """Return a summary dict for logging / dashboard use."""
        arr = np.array(self._window) if self._window else np.array([0.0])
        return {
            "n":         len(self._window),
            "mean":      float(arr.mean()),
            "std":       float(arr.std()),
            "threshold": self.threshold,
            "last":      float(self._window[-1]) if self._window else None,
        }
