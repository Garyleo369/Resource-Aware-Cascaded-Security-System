"""
cascaded_auth/ksd.py — Keystroke Dynamics for Prototype v2.

Changes from v1
---------------
- Feature vector is now shape (12,):
    indices  0-4  : dwell stats   [mean, std, p10, p50, p90]  (seconds)
    indices  5-9  : flight stats  [mean, std, p10, p50, p90]  (seconds)
    index   10    : typing_rate   (keystrokes / second)
    index   11    : burst_ratio   (std(flights) / mean(flights))

- K = 1.2  — tight band above rolling median for sensitive anomaly detection
- FLOOR = 0.45 — threshold allowed to go lower so genuine anomalies are caught
- WINDOW = 25  — score every 25 keystrokes (faster feedback, faster testing)
- Recalibration loop: model retrained after RECAL_EVERY CV-confirmed windows (Fix 4)
- enrol_vectors.joblib saved at fit time so retraining can blend old + new (Fix 4)

IMPORTANT: incompatible with any model trained on the old (17,) v1 vector.
Delete model.joblib / enrol_vectors.joblib before switching to v2.
"""

import os
import collections
import time
import logging

import numpy as np
import joblib
from sklearn.ensemble import IsolationForest

log = logging.getLogger(__name__)

# ── Where v2 artefacts are written (same directory as this file) ──────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE         = os.path.join(_HERE, "model.joblib")
ENROL_VECTORS_FILE = os.path.join(_HERE, "enrol_vectors.joblib")

# ── Scoring constants ──────────────────────────────────────────────────────────────────────
WINDOW      = 25    # keystrokes per scoring window (reduced for faster feedback)
K           = 1.2   # threshold sensitivity — tighter band catches anomalies sooner
ROLLING_N   = 20    # windows kept for dynamic threshold
FLOOR       = 0.45  # threshold floor — lowered so genuine anomalies are not suppressed
RECAL_EVERY = 3     # retrain after this many CV-confirmed windows (faster adaptation)


# ══════════════════════════════════════════════════════════════════════════════
# KeystrokeBuffer
# ══════════════════════════════════════════════════════════════════════════════

class KeystrokeBuffer:
    """
    Accumulates (key, dwell, flight) tuples from raw keyboard events.

    Usage
    -----
    Attach on_press / on_release to a pynput Listener.
    Poll ready() after each on_release; when True call flush() to get the
    completed window of events and pass them to extract_features().
    """

    def __init__(self):
        self._buffer      = collections.deque(maxlen=WINDOW)
        self._press_times: dict[str, float] = {}
        self._last_release: float = 0.0

    # ── pynput callbacks ──────────────────────────────────────────────────────

    def on_press(self, key):
        """Record the press timestamp for this key."""
        try:
            ch = key.char if (hasattr(key, "char") and key.char) else str(key)
        except Exception:
            ch = str(key)
        self._press_times[ch] = time.perf_counter()

    def on_release(self, key):
        """Compute dwell and flight; append (key, dwell, flight) to buffer."""
        try:
            ch = key.char if (hasattr(key, "char") and key.char) else str(key)
        except Exception:
            ch = str(key)

        if ch not in self._press_times:
            return

        release = time.perf_counter()
        dwell   = release - self._press_times[ch]
        flight  = self._press_times[ch] - self._last_release  # 0.0 for first key

        self._buffer.append((ch, dwell, max(flight, 0.0)))
        self._last_release = release
        del self._press_times[ch]

    # ── State queries ─────────────────────────────────────────────────────────

    def ready(self) -> bool:
        """Return True when the buffer holds exactly WINDOW events."""
        return len(self._buffer) == WINDOW

    def flush(self) -> list[tuple[str, float, float]]:
        """Return a copy of the current buffer and clear it."""
        events = list(self._buffer)
        self._buffer.clear()
        return events


# ══════════════════════════════════════════════════════════════════════════════
# Feature extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_features(events: list[tuple[str, float, float]]) -> np.ndarray:
    """
    Build a 12-D feature vector from one window of keystroke events.

    Parameters
    ----------
    events : list of (key, dwell, flight)
        As produced by KeystrokeBuffer.flush().

    Returns
    -------
    np.ndarray of shape (12,), dtype float64.
    """
    dwells  = [e[1] for e in events]
    flights = [e[2] for e in events if e[2] > 0]
    if not flights:
        flights = [0.0]

    stats: list[float] = []
    for arr in (dwells, flights):
        a = np.asarray(arr, dtype=np.float64)
        stats += [
            float(np.mean(a)),
            float(np.std(a)),
            float(np.percentile(a, 10)),
            float(np.percentile(a, 50)),
            float(np.percentile(a, 90)),
        ]

    # ── Tempo features (Fix 5) ────────────────────────────────────────────────
    total_time  = sum(dwells) + sum(flights)
    typing_rate = len(events) / (total_time + 1e-6)            # keystrokes/sec

    f_arr       = np.asarray(flights, dtype=np.float64)
    burst_ratio = float(np.std(f_arr) / (np.mean(f_arr) + 1e-6))  # rhythm consistency

    stats += [typing_rate, burst_ratio]

    vec = np.array(stats, dtype=np.float64)
    assert vec.shape == (12,), f"Unexpected feature shape: {vec.shape}"
    return vec


# ══════════════════════════════════════════════════════════════════════════════
# AnomalyScorer
# ══════════════════════════════════════════════════════════════════════════════

class AnomalyScorer:
    """
    IsolationForest wrapper with dynamic threshold, floor, and recalibration.

    Scoring convention
    ------------------
    score() returns a float in [0, 1] where:
      0.0 = very normal
      1.0 = very anomalous

    Threshold
    ---------
    threshold() = max(mean(last N scores) + K * std(last N scores), FLOOR)
    Falls back to 0.65 during warm-up (fewer than 5 windows scored).
    """

    def __init__(self):
        self.model: IsolationForest | None = None
        self.scores: collections.deque     = collections.deque(maxlen=ROLLING_N)
        self.confirmed_windows: list       = []   # CV-confirmed vecs for recal

    # ── Model lifecycle ───────────────────────────────────────────────────────

    def fit(self, feature_matrix: np.ndarray, save_vectors: bool = False) -> None:
        """
        Train an IsolationForest on feature_matrix and persist to disk.

        Parameters
        ----------
        feature_matrix : np.ndarray, shape (N, 12)
        save_vectors   : if True, also dump feature_matrix to enrol_vectors.joblib
                         so retraining can blend it with new confirmed data.
        """
        self.model = IsolationForest(
            n_estimators=200,
            contamination=0.02,
            random_state=42,
        )
        self.model.fit(feature_matrix)
        joblib.dump(self.model, MODEL_FILE)
        log.info("IsolationForest fitted on %d windows → %s", len(feature_matrix), MODEL_FILE)

        if save_vectors:
            joblib.dump(feature_matrix, ENROL_VECTORS_FILE)
            log.info("Enrolment vectors saved → %s", ENROL_VECTORS_FILE)

    def load(self) -> None:
        """Load a previously saved model from MODEL_FILE."""
        if not os.path.exists(MODEL_FILE):
            raise FileNotFoundError(
                f"Model not found at {MODEL_FILE}. Run enrol.py first."
            )
        self.model = joblib.load(MODEL_FILE)
        log.info("Model loaded from %s", MODEL_FILE)

    # ── Scoring ───────────────────────────────────────────────────────────────

    def score(self, features: np.ndarray) -> float:
        """
        Score a 12-D feature vector.

        Returns a normalised anomaly score in [0, 1]:
          raw score from IsolationForest.score_samples() is in roughly [-0.5, 0.1];
          we map it so that negative (anomalous) raw scores approach 1.0.
        """
        if self.model is None:
            return 0.5   # neutral fallback

        raw = float(self.model.score_samples([features])[0])
        offset = float(self.model.offset_)
        
        # model.offset_ is a negative number (e.g., -0.65).
        # raw > offset means INLIER (normal). raw < offset means OUTLIER (anomalous).
        # We want to map:
        #   raw == offset  -> 0.5
        #   raw > offset   -> < 0.5 (closer to 0)
        #   raw < offset   -> > 0.5 (closer to 1)
        # We use a logistic-like sigmoid centered on the offset for a smooth distribution.
        # scaling factor of 10.0 spreads the scores nicely across [0, 1].
        normalised = float(1.0 / (1.0 + np.exp(10.0 * (raw - offset))))

        self.scores.append(normalised)
        # Warm-up: cap score at 0.50 for first 3 windows so cold-start
        # typing differences don't immediately trigger an alert.
        if len(self.scores) < 3:
            return min(normalised, 0.50)
        return normalised

    def threshold(self) -> float:
        """
        Compute the current detection threshold.

        During warm-up (fewer than 3 scored windows) returns 0.55.
        Otherwise: max(median + K * std, FLOOR).
        """
        if len(self.scores) < 3:
            return 0.55   # warm-up fallback — lower than before for faster sensitisation

        arr     = np.array(self.scores)
        # Use median instead of mean — robust against outlier sessions
        # inflating the baseline and pushing the threshold too high.
        dynamic = float(np.median(arr) + K * np.std(arr))
        return max(dynamic, FLOOR)

    def is_anomaly(self, score: float) -> bool:
        """Return True if score exceeds the current threshold."""
        return score > self.threshold()

    # ── Recalibration (Fix 4) ─────────────────────────────────────────────────

    def add_confirmed(self, vec: np.ndarray) -> None:
        """
        Register a feature vector confirmed as genuine by CV.

        Accumulates until RECAL_EVERY vectors are collected, then retrains
        by blending the original enrolment data with the new confirmed windows.
        """
        self.confirmed_windows.append(vec)
        log.info(
            "[recal] CV-confirmed window added (%d / %d before retrain)",
            len(self.confirmed_windows), RECAL_EVERY,
        )
        if len(self.confirmed_windows) >= RECAL_EVERY:
            self._retrain()
            self.confirmed_windows.clear()

    def _retrain(self) -> None:
        """Blend enrolment data + confirmed windows and refit the model."""
        try:
            original = joblib.load(ENROL_VECTORS_FILE)
        except FileNotFoundError:
            log.warning("[recal] enrol_vectors.joblib not found — retraining on confirmed only.")
            original = np.empty((0, 12))

        combined = np.vstack([original, np.array(self.confirmed_windows)])
        self.fit(combined)   # also overwrites model.joblib
        print(f"[ksd] Model retrained on {len(combined)} windows")
