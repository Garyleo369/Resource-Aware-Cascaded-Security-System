"""
ksd/model.py — IsolationForest + One-Class SVM ensemble scoring wrapper.

Loads the pre-trained models from disk and normalises raw anomaly scores to
the [0, 1] range so that higher values indicate more anomalous behaviour.

Model files (in data/):
  model.joblib  — IsolationForest  (always required)
  ocsvm.joblib  — One-Class SVM    (optional; graceful fallback if absent)

The models expect a 17-D feature vector produced by ksd/features.py:
  Indices  0-4  : dwell-time stats   [mean, std, p10, p50, p90]
  Indices  5-9  : flight-time stats  [mean, std, p10, p50, p90]
  Indices 10-13 : top-4 digraph latency means
  Index   14    : typing speed (keystrokes/sec)
  Index   15    : backspace frequency (fraction 0-1)
  Index   16    : digraph latency variance (std, ms)
"""

import os
import logging

import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

log = logging.getLogger(__name__)

DATA_DIR    = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MODEL_PATH  = os.path.join(DATA_DIR, "model.joblib")
OCSVM_PATH  = os.path.join(DATA_DIR, "ocsvm.joblib")

# Relative weight of the OC-SVM score in the ensemble (0.0 → IF-only, 1.0 → OC-SVM only)
OCSVM_WEIGHT: float = 0.5


# ── IsolationForest score normalisation ───────────────────────────────────────
# score_samples() returns values roughly in [-0.5, 0.1] for typical datasets;
# the exact range is model-dependent and cannot be hard-coded.
# We use a robust sigmoid-like shift-and-clip that stays in [0, 1] regardless.
#   raw > 0   → very normal   → score near 0
#   raw = 0   → boundary      → score = 0.5
#   raw < 0   → anomalous     → score near 1
# We clip at ±0.5 to keep the mapping reasonable.
_IF_CLIP = 0.5


class KSDModel:
    """
    Thin wrapper around a saved IsolationForest.

    Scoring convention
    ──────────────────
    ``score()`` returns a float in **[0, 1]** where:
      • 0.0 = very normal (low anomaly)
      • 1.0 = very anomalous

    This makes the score directly comparable with the threshold computed in
    ``threshold.py``.
    """

    def __init__(self):
        self._clf: IsolationForest | None = None
        self._load()

    def _load(self):
        if not os.path.exists(MODEL_PATH):
            log.warning("KSD model not found at %s — run enrol.py first.",
                        MODEL_PATH)
            return
        try:
            self._clf = joblib.load(MODEL_PATH)
            log.info("KSD IsolationForest loaded from %s", MODEL_PATH)
        except Exception as exc:
            log.error("Failed to load KSD model: %s", exc)

    @property
    def ready(self) -> bool:
        return self._clf is not None

    def score(self, feature_vec: np.ndarray) -> float:
        """
        Score a single feature vector.

        Returns
        -------
        float in [0, 1]; higher → more anomalous.
        """
        if not self.ready:
            return 0.5          # neutral fallback

        vec = feature_vec.reshape(1, -1)
        raw = float(self._clf.score_samples(vec)[0])

        # score_samples: more-negative → more anomalous.
        # Map: clip to [-_IF_CLIP, +_IF_CLIP], shift so 0 → 0.5, invert.
        raw_clipped = float(np.clip(raw, -_IF_CLIP, _IF_CLIP))
        # Map [-clip, +clip] → [1, 0]  (low raw score → high anomaly score)
        anomaly_score = (1.0 - (raw_clipped + _IF_CLIP) / (2.0 * _IF_CLIP))
        return float(np.clip(anomaly_score, 0.0, 1.0))


# ── One-Class SVM wrapper ─────────────────────────────────────────────────────

# OC-SVM decision_function range: negative values → inside boundary (normal),
# positive values → outside boundary (anomalous).
# We clip to [-2, 2] to handle wider SVM margins gracefully.
_OCSVM_CLIP = 2.0


class OneClassSVMModel:
    """
    Thin wrapper around a saved OneClassSVM (RBF kernel).

    Scoring convention: same as KSDModel — [0, 1], higher = more anomalous.

    The OC-SVM ``decision_function`` returns positive values for normal points
    (inside the learned hypersphere) and negative for anomalies (outside).
    We invert and normalise to produce the same [0,1] convention.
    """

    def __init__(self):
        self._clf: OneClassSVM | None = None
        self._load()

    def _load(self):
        if not os.path.exists(OCSVM_PATH):
            log.info("OC-SVM model not found at %s — will run in IF-only mode.",
                     OCSVM_PATH)
            return
        try:
            self._clf = joblib.load(OCSVM_PATH)
            log.info("KSD OC-SVM loaded from %s", OCSVM_PATH)
        except Exception as exc:
            log.error("Failed to load OC-SVM model: %s", exc)

    @property
    def ready(self) -> bool:
        return self._clf is not None

    def score(self, feature_vec: np.ndarray) -> float:
        """
        Returns a float in [0, 1]; higher → more anomalous.
        """
        if not self.ready:
            return 0.5

        vec = feature_vec.reshape(1, -1)
        raw = float(self._clf.decision_function(vec)[0])

        # decision_function: positive = normal, negative = anomaly
        # Clip to [-_OCSVM_CLIP, _OCSVM_CLIP] then map to [0, 1] inverted
        raw_clipped = float(np.clip(raw, -_OCSVM_CLIP, _OCSVM_CLIP))
        # Map [-clip, +clip] → [1, 0]  (high raw → low anomaly → score near 0)
        anomaly_score = (1.0 - (raw_clipped + _OCSVM_CLIP) / (2.0 * _OCSVM_CLIP))
        return float(np.clip(anomaly_score, 0.0, 1.0))


# ── Ensemble model ────────────────────────────────────────────────────────────

class EnsembleKSDModel:
    """
    Combines IsolationForest and One-Class SVM via a weighted average.

    If only one model is available the weight is automatically adjusted to
    100 % for the available model (backward-compatible fallback).

    Parameters
    ----------
    ocsvm_weight : float
        Fraction of the final score attributable to the OC-SVM model.
        Remaining weight goes to IsolationForest.
        Default: ``OCSVM_WEIGHT`` module constant (0.5).
    """

    def __init__(self, ocsvm_weight: float = OCSVM_WEIGHT):
        self._if    = KSDModel()
        self._ocsvm = OneClassSVMModel()
        self._ocsvm_weight = ocsvm_weight
        log.info(
            "EnsembleKSDModel: IF=%s  OC-SVM=%s  weight=(IF=%.0f%% / OC-SVM=%.0f%%)",
            "✓" if self._if.ready    else "✗",
            "✓" if self._ocsvm.ready else "✗",
            (1.0 - ocsvm_weight) * 100,
            ocsvm_weight * 100,
        )

    @property
    def ready(self) -> bool:
        """True if at least one underlying model is available."""
        return self._if.ready or self._ocsvm.ready

    def _load(self):
        """Hot-reload both underlying models from disk."""
        self._if._load()
        self._ocsvm._load()

    def score(self, feature_vec: np.ndarray) -> float:
        """
        Return a weighted ensemble anomaly score in [0, 1].

        Falls back to the available model's score if only one is ready.
        Returns 0.5 (neutral) if no model is loaded.
        """
        if_ready    = self._if.ready
        ocsvm_ready = self._ocsvm.ready

        if not if_ready and not ocsvm_ready:
            return 0.5

        if if_ready and ocsvm_ready:
            w_ocsvm = self._ocsvm_weight
            w_if    = 1.0 - w_ocsvm
            if_score    = self._if.score(feature_vec)
            ocsvm_score = self._ocsvm.score(feature_vec)
            combined = w_if * if_score + w_ocsvm * ocsvm_score
            log.debug(
                "Ensemble: IF=%.4f  OC-SVM=%.4f  combined=%.4f",
                if_score, ocsvm_score, combined,
            )
            return float(combined)

        # One model missing — use whichever is available
        if if_ready:
            return self._if.score(feature_vec)
        return self._ocsvm.score(feature_vec)

    def component_scores(self, feature_vec: np.ndarray) -> dict:
        """
        Return individual model scores as a dict (useful for dashboard display).

        Returns
        -------
        dict with keys 'if_score', 'ocsvm_score', 'ensemble_score'.
        """
        if_score    = self._if.score(feature_vec)    if self._if.ready    else None
        ocsvm_score = self._ocsvm.score(feature_vec) if self._ocsvm.ready else None
        ensemble    = self.score(feature_vec)
        return {
            "if_score":       if_score,
            "ocsvm_score":    ocsvm_score,
            "ensemble_score": ensemble,
        }
