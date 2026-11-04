"""
cascaded_auth/recalibrator.py — Verification-Gated Adaptive KSD Recalibration.

Design goals
------------
• Collected verified_samples only accumulate after a *successful* face
  verification (Tier 2 PASS).  Unverified samples never touch the model.
• Threshold nudges are bounded to ±MAX_DELTA per update and clamped to
  [FLOOR_BOUND, CEIL_BOUND] so the system cannot drift into a permanently
  open or permanently closed state.
• Temporal smoothing (ANOMALY_WINDOW / ANOMALY_COUNT) prevents single-window
  glitches from triggering re-calibration or face-check.
• A post-verification cooldown (POST_VERIFY_COOLDOWN_S) prevents the system
  from immediately re-triggering after a successful check.
• High-risk anomalies (score > HIGH_RISK_THRESHOLD) bypass temporal smoothing
  and fire face verification immediately — they are never suppressed.
• The recalibration history is persisted to disk (recal_history.json) so the
  dashboard can display it across restarts.
"""

from __future__ import annotations

import os
import json
import time
import logging
import threading
import collections
from datetime import datetime
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# ── File paths ────────────────────────────────────────────────────────────────
_HERE            = os.path.dirname(os.path.abspath(__file__))
RECAL_HISTORY_FILE = os.path.join(_HERE, "recal_history.json")

# ── Configuration constants ───────────────────────────────────────────────────
# Minimum verified samples before any threshold change is allowed.
MIN_VERIFIED_SAMPLES: int   = 30

# Maximum threshold shift per recalibration update (±).
MAX_DELTA: float             = 0.01

# Absolute safety bounds for the adaptive threshold offset.
FLOOR_BOUND: float           = -0.15   # cannot lower threshold by more than this
CEIL_BOUND: float            =  0.15   # cannot raise threshold by more than this

# Temporal smoothing: require this many anomalies within the last N windows
# before firing Tier 2 (prevents single-glitch false triggers).
ANOMALY_WINDOW: int          = 3       # look-back window size (in scored windows)
ANOMALY_COUNT: int           = 2       # anomalies needed within that window

# High-risk bypass: scores above this value bypass temporal smoothing and
# fire face verification immediately (never suppressed).
HIGH_RISK_THRESHOLD: float   = 0.85

# Post-verification cooldown: seconds after a CV check before another auto-
# trigger is allowed.  Manual triggers always bypass this.
POST_VERIFY_COOLDOWN_S: float = 30.0

# Maximum recalibration history entries to keep on disk.
MAX_HISTORY_ENTRIES: int      = 200


class VerificationGatedRecalibrator:
    """
    Manages adaptive KSD threshold recalibration gated on face verification.

    Thread safety
    -------------
    All state is protected by a single ``_lock``.  The background KSD thread
    calls ``record_window()`` and ``on_cv_result()``.  The Streamlit main
    thread calls ``should_fire_cv()`` and reads display properties.

    Parameters
    ----------
    scorer : AnomalyScorer
        Live reference to the AnomalyScorer used by the KSD background thread.
        The recalibrator reads ``scorer.threshold()`` and writes
        ``scorer.threshold_offset`` (a float attribute this module adds).
    """

    def __init__(self, scorer) -> None:
        self._scorer          = scorer
        self._lock            = threading.Lock()

        # Verified typing samples accumulated since last recalibration.
        self._verified_samples: list[np.ndarray] = []

        # Sliding window of recent anomaly flags for temporal smoothing.
        self._anomaly_window: collections.deque[bool] = collections.deque(
            maxlen=ANOMALY_WINDOW
        )

        # Epoch of the last completed CV check (pass or fail).
        self._last_cv_ts: float = 0.0

        # The current adaptive threshold *offset* (added to the dynamic base).
        # Bounded to [FLOOR_BOUND, CEIL_BOUND].
        self._threshold_offset: float = 0.0

        # The feature vector that most recently triggered the anomaly flag
        # (retained so a subsequent PASS can add it to verified_samples).
        self._pending_anomaly_vec: Optional[np.ndarray] = None

        # Whether a CV check is currently in progress (prevents double-fire).
        self._cv_in_flight: bool = False

        # Recalibration history (list of dicts, newest last).
        self._history: list[dict] = []

        # Load any existing history from disk.
        self._load_history()

        # Ensure the scorer has a threshold_offset attribute.
        if not hasattr(scorer, "threshold_offset"):
            scorer.threshold_offset = 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    def record_window(
        self,
        score:     float,
        vec:       np.ndarray,
        triggered: bool,
    ) -> None:
        """
        Called after every KSD window is scored.

        Updates the temporal anomaly buffer.  Stores ``vec`` as the most
        recent anomaly candidate if ``triggered`` is True.

        Parameters
        ----------
        score     : normalised anomaly score in [0, 1]
        vec       : 12-D feature vector for this window
        triggered : True if the base AnomalyScorer flagged this window
        """
        with self._lock:
            self._anomaly_window.append(triggered)
            if triggered:
                self._pending_anomaly_vec = vec.copy()

    def should_fire_cv(self, score: float, triggered: bool) -> bool:
        """
        Decide whether to fire Tier 2 (face verification) for this window.

        Rules (evaluated in order):
        1. High-risk scores (> HIGH_RISK_THRESHOLD) always return True.
        2. If a CV check is already in-flight, return False.
        3. If within the post-verification cooldown, return False.
        4. If temporal smoothing threshold is met (≥ ANOMALY_COUNT anomalies
           in the last ANOMALY_WINDOW windows), return True.
        5. Otherwise return False.

        Note: ``triggered`` is the raw AnomalyScorer decision.  ``should_fire_cv``
        may return True even when triggered=False (e.g. delayed smoothing) or
        False even when triggered=True (suppressed by cooldown/smoothing).
        """
        with self._lock:
            # Rule 1 — high-risk bypass (never suppressed)
            if score > HIGH_RISK_THRESHOLD:
                log.info(
                    "[recal] High-risk score %.4f — bypassing temporal smoothing.", score
                )
                return True

            if not triggered:
                return False

            # Rule 2 — already checking
            if self._cv_in_flight:
                return False

            # Rule 3 — cooldown
            elapsed = time.time() - self._last_cv_ts
            if elapsed < POST_VERIFY_COOLDOWN_S:
                log.debug(
                    "[recal] Post-verification cooldown active (%.1f / %.1f s).",
                    elapsed, POST_VERIFY_COOLDOWN_S,
                )
                return False

            # Rule 4 — temporal smoothing
            recent_anomalies = sum(self._anomaly_window)
            if recent_anomalies >= ANOMALY_COUNT:
                log.info(
                    "[recal] Temporal threshold met (%d/%d) — firing CV.",
                    recent_anomalies, ANOMALY_WINDOW,
                )
                return True

            log.debug(
                "[recal] Triggered but smoothed out (%d/%d anomalies in window).",
                recent_anomalies, ANOMALY_WINDOW,
            )
            return False

    def mark_cv_started(self) -> None:
        """Call immediately before launching the CV check."""
        with self._lock:
            self._cv_in_flight = True

    def on_cv_result(self, matched: bool, score_at_trigger: Optional[float] = None) -> dict:
        """
        Called after face verification completes.

        Parameters
        ----------
        matched           : True = owner confirmed, False = impostor / fail
        score_at_trigger  : KSD anomaly score that caused this CV check

        Returns
        -------
        dict with keys:
            recalibrated   : bool — True if a threshold update was applied
            old_offset     : float
            new_offset     : float
            verified_count : int — running total of verified samples
            message        : human-readable summary
        """
        with self._lock:
            self._cv_in_flight = False
            self._last_cv_ts   = time.time()

            if not matched:
                # Impostor / failed verification — preserve everything as-is.
                # Do NOT add any samples; do NOT adjust threshold.
                self._pending_anomaly_vec = None
                result = {
                    "recalibrated":   False,
                    "old_offset":     round(self._threshold_offset, 6),
                    "new_offset":     round(self._threshold_offset, 6),
                    "verified_count": len(self._verified_samples),
                    "message":        "Verification FAILED — baseline preserved, security response triggered.",
                    "timestamp":      time.time(),
                }
                log.info("[recal] CV failed — baseline unchanged.")
                return result

            # ── Verification PASSED ──────────────────────────────────────────
            # Add the pending anomaly vector (the typing sample that caused the
            # alert) to the pool of verified samples.
            if self._pending_anomaly_vec is not None:
                self._verified_samples.append(self._pending_anomaly_vec)
                log.info(
                    "[recal] CV passed — verified sample added (%d / %d).",
                    len(self._verified_samples), MIN_VERIFIED_SAMPLES,
                )
            self._pending_anomaly_vec = None

            # Reset temporal window — legitimate typing was confirmed, so we
            # start the smoothing counter fresh.
            self._anomaly_window.clear()

            old_offset = self._threshold_offset
            recalibrated = False

            if len(self._verified_samples) >= MIN_VERIFIED_SAMPLES:
                recalibrated = True
                new_offset   = self._compute_nudge()
                self._threshold_offset = new_offset
                # Propagate to the live scorer.
                self._scorer.threshold_offset = self._threshold_offset
                # Keep only the most recent MIN_VERIFIED_SAMPLES samples (rolling).
                self._verified_samples = self._verified_samples[-MIN_VERIFIED_SAMPLES:]
                log.info(
                    "[recal] Threshold recalibrated: offset %.4f → %.4f.",
                    old_offset, self._threshold_offset,
                )
            else:
                new_offset = old_offset

            message = (
                f"Verification PASSED — verified sample {len(self._verified_samples)}/{MIN_VERIFIED_SAMPLES}."
                + (
                    f"  Threshold offset updated: {old_offset:+.4f} → {new_offset:+.4f}."
                    if recalibrated
                    else f"  Need {MIN_VERIFIED_SAMPLES - len(self._verified_samples)} more samples for recalibration."
                )
            )

            entry = {
                "recalibrated":   recalibrated,
                "old_offset":     round(old_offset, 6),
                "new_offset":     round(new_offset, 6),
                "verified_count": len(self._verified_samples),
                "score_at_trigger": round(score_at_trigger, 6) if score_at_trigger is not None else None,
                "message":        message,
                "timestamp":      time.time(),
                "datetime":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            self._history.append(entry)
            self._history = self._history[-MAX_HISTORY_ENTRIES:]
            self._save_history()
            return entry

    # ── Read-only properties (safe to call without holding the lock) ──────────

    @property
    def threshold_offset(self) -> float:
        with self._lock:
            return self._threshold_offset

    @property
    def verified_count(self) -> int:
        with self._lock:
            return len(self._verified_samples)

    @property
    def history(self) -> list[dict]:
        with self._lock:
            return list(self._history)

    @property
    def cooldown_remaining(self) -> float:
        """Seconds remaining in the post-verification cooldown (0 if expired)."""
        with self._lock:
            elapsed = time.time() - self._last_cv_ts
            return max(0.0, POST_VERIFY_COOLDOWN_S - elapsed)

    @property
    def anomaly_window_state(self) -> tuple[int, int]:
        """(recent anomaly count, window size) for display."""
        with self._lock:
            return (int(sum(self._anomaly_window)), ANOMALY_WINDOW)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _compute_nudge(self) -> float:
        """
        Compute a new threshold offset from the verified sample pool.

        Strategy: the verified samples are *legitimate* typing windows that
        scored above the threshold.  Their mean score tells us how far the
        current threshold is from the genuine distribution.  We nudge the
        offset *downward* (raise effective sensitivity) by a fraction of the
        gap between the sample mean and the current effective threshold,
        clamped to ±MAX_DELTA.

        The offset is then clamped to [FLOOR_BOUND, CEIL_BOUND].
        """
        if not self._verified_samples:
            return self._threshold_offset

        # Score each verified sample against the *current* model (no retraining).
        verified_scores = []
        for vec in self._verified_samples:
            try:
                s = self._scorer.score(vec)
                verified_scores.append(s)
            except Exception:
                pass
        if not verified_scores:
            return self._threshold_offset

        mean_verified = float(np.mean(verified_scores))
        base_thr      = self._scorer.threshold()   # current dynamic threshold (no offset)

        # If verified samples score above the threshold, the threshold is too
        # tight.  Raise it slightly (positive nudge → raise effective threshold).
        # If they score well below, lower it slightly (negative nudge).
        gap   = mean_verified - base_thr
        nudge = float(np.clip(gap * 0.5, -MAX_DELTA, MAX_DELTA))

        new_offset = float(np.clip(
            self._threshold_offset + nudge,
            FLOOR_BOUND,
            CEIL_BOUND,
        ))
        log.info(
            "[recal] mean_verified=%.4f  base_thr=%.4f  gap=%.4f  nudge=%.4f  "
            "offset: %.4f → %.4f",
            mean_verified, base_thr, gap, nudge,
            self._threshold_offset, new_offset,
        )
        return new_offset

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load_history(self) -> None:
        if not os.path.exists(RECAL_HISTORY_FILE):
            return
        try:
            with open(RECAL_HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._history          = data.get("history", [])[-MAX_HISTORY_ENTRIES:]
            self._threshold_offset = float(data.get("threshold_offset", 0.0))
            # Clamp on load in case the file has out-of-range values.
            self._threshold_offset = float(np.clip(
                self._threshold_offset, FLOOR_BOUND, CEIL_BOUND
            ))
            # Re-apply the saved offset to the live scorer.
            if hasattr(self._scorer, "threshold_offset"):
                self._scorer.threshold_offset = self._threshold_offset
            log.info(
                "[recal] Loaded history (%d entries), offset=%.4f.",
                len(self._history), self._threshold_offset,
            )
        except Exception as exc:
            log.warning("[recal] Could not load recal_history.json: %s", exc)

    def _save_history(self) -> None:
        try:
            data = {
                "threshold_offset": round(self._threshold_offset, 6),
                "history":          self._history,
            }
            with open(RECAL_HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as exc:
            log.warning("[recal] Could not save recal_history.json: %s", exc)

