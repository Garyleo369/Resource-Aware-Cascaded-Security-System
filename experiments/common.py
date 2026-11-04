"""
experiments/common.py — Shared utilities for all four evaluation experiments.

This module ONLY imports from existing application code (cascaded_auth/, cv/).
It does NOT modify any existing file.

Available helpers
-----------------
- load_v2_scorer()            → fresh AnomalyScorer loaded from cascaded_auth/model.joblib
- load_enrol_vectors()        → np.ndarray (N, 12) from cascaded_auth/enrol_vectors.joblib
- load_session_log()          → dict parsed from data/session_log.json
- generate_genuine(n, rng)    → list of (N,12) feature vectors (owner, 5% noise)
- generate_shift(n, rng)      → list of (N,12) vectors (owner under behavioral drift)
- generate_impostor(n, rng)   → list of (N,12) vectors (2.5–4σ from enrolled mean)
- score_batch(scorer, vecs)   → np.ndarray of anomaly scores
- compute_metrics(y_true, scores, threshold) → dict of classification metrics
- compute_roc(y_true, scores) → (fpr_arr, tpr_arr, thresholds)
- compute_eer(fpr, tpr, thresholds) → (eer, eer_threshold)
- ensure_results_dir()        → creates experiments/results/ if absent
- RESULTS_DIR                 → absolute path to experiments/results/
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings

import joblib
import numpy as np

# ── Path setup — allow imports from project root ─────────────────────────────
_EXP_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT     = os.path.dirname(_EXP_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from cascaded_auth.ksd import AnomalyScorer, WINDOW, K, FLOOR, ROLLING_N

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR        = os.path.join(_EXP_DIR, "results")
_ENROL_VECS_FILE   = os.path.join(_ROOT, "cascaded_auth", "enrol_vectors.joblib")
_SESSION_LOG_FILE  = os.path.join(_ROOT, "data", "session_log.json")

# ── v2 model constants (for reporting) ───────────────────────────────────────
V2_CONFIG = {
    "feature_dim":       12,
    "feature_names":     [
        "dwell_mean", "dwell_std", "dwell_p10", "dwell_p50", "dwell_p90",
        "flight_mean", "flight_std", "flight_p10", "flight_p50", "flight_p90",
        "typing_rate", "burst_ratio",
    ],
    "model":             "IsolationForest",
    "n_estimators":      200,
    "contamination":     0.02,
    "random_state":      42,
    "window_size_keys":  WINDOW,      # 25
    "K":                 K,           # 1.2
    "FLOOR":             FLOOR,       # 0.45
    "ROLLING_N":         ROLLING_N,   # 20
    "threshold_formula": "max(median(last_N_scores) + K * std(last_N_scores), FLOOR)",
    "warm_up_windows":   3,
    "warm_up_threshold": 0.55,
    "warm_up_cap":       0.50,
}


# ═════════════════════════════════════════════════════════════════════════════
# Loader helpers
# ═════════════════════════════════════════════════════════════════════════════

def load_v2_scorer() -> AnomalyScorer:
    """Return a freshly-loaded AnomalyScorer (loads cascaded_auth/model.joblib)."""
    s = AnomalyScorer()
    s.load()
    return s


def load_enrol_vectors() -> np.ndarray:
    """
    Return the (N, 12) enrolment feature matrix.

    Raises FileNotFoundError if the file does not exist.
    """
    if not os.path.exists(_ENROL_VECS_FILE):
        raise FileNotFoundError(
            f"Enrolment vectors not found at {_ENROL_VECS_FILE}. "
            "Run cascaded_auth/enrol.py first."
        )
    vecs = joblib.load(_ENROL_VECS_FILE)
    return np.array(vecs, dtype=np.float64)


def load_session_log() -> dict:
    """Return the parsed data/session_log.json dict."""
    if not os.path.exists(_SESSION_LOG_FILE):
        return {"ksd_scores": [], "cv_results": []}
    with open(_SESSION_LOG_FILE, encoding="utf-8") as fh:
        return json.load(fh)


# ═════════════════════════════════════════════════════════════════════════════
# Synthetic data generators  (calibrated from real enrolment vectors)
# ═════════════════════════════════════════════════════════════════════════════

def _perturb(vec: np.ndarray, noise_scale: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0, noise_scale, size=vec.shape)
    return (vec + vec * noise).clip(0.0)


def generate_genuine(
    n: int,
    rng: np.random.Generator,
    enrol_vecs: np.ndarray | None = None,
) -> list[np.ndarray]:
    """
    Simulate the enrolled user typing normally.

    Samples from real enrolment vectors with 5 % Gaussian noise so that each
    window differs slightly (realistic intra-user variation).
    """
    if enrol_vecs is None:
        enrol_vecs = load_enrol_vectors()
    indices = rng.integers(0, len(enrol_vecs), size=n)
    return [_perturb(enrol_vecs[i], 0.05, rng) for i in indices]


def generate_shift(
    n: int,
    rng: np.random.Generator,
    enrol_vecs: np.ndarray | None = None,
) -> list[np.ndarray]:
    """
    Simulate the legitimate user under a behavioral shift
    (tired / different keyboard / stressed).

    Scales dwell ×1.4–1.7, flight ×1.2–1.5, typing_rate ×0.5–0.7,
    burst_ratio ×1.3–2.0, plus 5 % random noise.
    """
    if enrol_vecs is None:
        enrol_vecs = load_enrol_vectors()
    indices = rng.integers(0, len(enrol_vecs), size=n)
    dwell_scale  = rng.uniform(1.4, 1.7)
    flight_scale = rng.uniform(1.2, 1.5)
    rate_scale   = rng.uniform(0.5, 0.7)
    burst_scale  = rng.uniform(1.3, 2.0)
    windows = []
    for i in indices:
        v = enrol_vecs[i].copy()
        v[0:5]  *= dwell_scale
        v[5:10] *= flight_scale
        v[10]   *= rate_scale
        v[11]   *= burst_scale
        windows.append(_perturb(v, 0.05, rng).clip(0.0))
    return windows


def generate_impostor(
    n: int,
    rng: np.random.Generator,
    enrol_vecs: np.ndarray | None = None,
) -> list[np.ndarray]:
    """
    Simulate an impostor whose typing pattern is 2.5–4σ from the enrolled mean.

    A consistent impostor profile is generated once per call (same person
    typing consistently with their own rhythm), then 5 % noise is added per
    window to simulate natural intra-impostor variability.
    """
    if enrol_vecs is None:
        enrol_vecs = load_enrol_vectors()
    mean = enrol_vecs.mean(axis=0)
    std  = enrol_vecs.std(axis=0)

    direction = rng.choice([-1, 1], size=12)
    shift     = std * rng.uniform(2.5, 4.0, size=12) * direction
    profile   = (mean + shift).clip(0.0)
    # Force clearly different dwell and typing rate
    profile[0]  = abs(mean[0]) * rng.uniform(1.8, 3.0)
    profile[10] = abs(mean[10]) * rng.uniform(0.3, 0.6)

    return [_perturb(profile, 0.05, rng).clip(0.0) for _ in range(n)]


# ═════════════════════════════════════════════════════════════════════════════
# Scoring helpers
# ═════════════════════════════════════════════════════════════════════════════

def score_batch(
    scorer: AnomalyScorer,
    feature_vecs: list[np.ndarray],
    warm_up_reset: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run feature_vecs through scorer.score() and return
    (scores_array, thresholds_array).

    warm_up_reset=True creates a fresh scorer for each batch (recommended so
    the warm-up window does not bleed across test batches).
    """
    scores     = []
    thresholds = []
    for vec in feature_vecs:
        s = scorer.score(vec)
        t = scorer.threshold()
        scores.append(s)
        thresholds.append(t)
    return np.array(scores, dtype=np.float64), np.array(thresholds, dtype=np.float64)


def score_batch_timed(
    scorer: AnomalyScorer,
    feature_vecs: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Like score_batch but also returns total wall-clock time (seconds) for
    all scoring calls (detection latency proxy).
    """
    scores     = []
    thresholds = []
    t0 = time.perf_counter()
    for vec in feature_vecs:
        scores.append(scorer.score(vec))
        thresholds.append(scorer.threshold())
    elapsed = time.perf_counter() - t0
    return (
        np.array(scores, dtype=np.float64),
        np.array(thresholds, dtype=np.float64),
        elapsed,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Classification metrics
# ═════════════════════════════════════════════════════════════════════════════

def compute_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict:
    """
    Compute binary classification metrics at a fixed threshold.

    Parameters
    ----------
    y_true    : 1-D array — 1 = anomalous/impostor, 0 = genuine
    scores    : 1-D array — anomaly scores in [0, 1]
    threshold : decision boundary

    Returns
    -------
    dict with keys: TP, TN, FP, FN, TPR, FPR, precision, F1, FAR, FRR, accuracy
    """
    y_pred = (scores >= threshold).astype(int)

    TP = int(np.sum((y_pred == 1) & (y_true == 1)))
    TN = int(np.sum((y_pred == 0) & (y_true == 0)))
    FP = int(np.sum((y_pred == 1) & (y_true == 0)))
    FN = int(np.sum((y_pred == 0) & (y_true == 1)))

    TPR       = TP / (TP + FN) if (TP + FN) > 0 else 0.0   # Recall / Sensitivity
    FPR       = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    F1        = (2 * precision * TPR / (precision + TPR)) if (precision + TPR) > 0 else 0.0
    FAR       = FP / (FP + TN) if (FP + TN) > 0 else 0.0   # = FPR in this context
    FRR       = FN / (FN + TP) if (FN + TP) > 0 else 0.0
    accuracy  = (TP + TN) / len(y_true) if len(y_true) > 0 else 0.0

    return {
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "TPR":       round(TPR,       4),
        "FPR":       round(FPR,       4),
        "precision": round(precision, 4),
        "F1":        round(F1,        4),
        "FAR":       round(FAR,       4),
        "FRR":       round(FRR,       4),
        "accuracy":  round(accuracy,  4),
        "threshold": round(threshold, 4),
        "n_samples": len(y_true),
        "n_genuine":  int(np.sum(y_true == 0)),
        "n_impostor": int(np.sum(y_true == 1)),
    }


def compute_roc(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC curve by sweeping thresholds.

    Returns (fpr_arr, tpr_arr, thresholds_arr).
    Thresholds are swept from max to min score.
    """
    thresholds = np.unique(scores)[::-1]   # descending
    fprs, tprs = [], []
    for thr in thresholds:
        m = compute_metrics(y_true, scores, thr)
        fprs.append(m["FPR"])
        tprs.append(m["TPR"])
    # Add (0,0) and (1,1) end-points for clean curves
    fprs = np.array([0.0] + list(fprs) + [1.0])
    tprs = np.array([0.0] + list(tprs) + [1.0])
    thresholds = np.array([thresholds[0] + 1e-6] + list(thresholds) + [thresholds[-1] - 1e-6])
    return fprs, tprs, thresholds


def compute_auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Trapezoidal AUC (compatible with NumPy ≥1.x and ≥2.0)."""
    try:
        # NumPy >= 2.0
        return float(np.trapezoid(tpr, fpr))
    except AttributeError:
        # NumPy < 2.0
        return float(np.trapz(tpr, fpr))


def compute_eer(
    fpr: np.ndarray,
    tpr: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[float, float]:
    """
    Equal Error Rate: the point where FAR ≈ FRR.

    FAR = FPR, FRR = 1 - TPR.

    Returns (eer_value, eer_threshold).
    """
    frr = 1.0 - tpr
    # Find threshold where |FAR - FRR| is minimised
    diff = np.abs(fpr - frr)
    idx  = int(np.argmin(diff))
    eer  = float((fpr[idx] + frr[idx]) / 2.0)
    return eer, float(thresholds[idx])


# ═════════════════════════════════════════════════════════════════════════════
# Session log analysis
# ═════════════════════════════════════════════════════════════════════════════

def parse_session_log_ksd(log_data: dict) -> dict:
    """
    Parse the KSD score stream from session_log.json.

    Returns
    -------
    dict with:
        raw_scores       : full list of scores (all values)
        valid_scores     : only scores in [0.0, 1.0] (normalised range)
        n_raw            : total count
        n_valid          : count of normalised-range scores
        has_legacy_scores: True if any score > 1.0 detected (old convention)
        score_range      : (min, max) of raw_scores
    """
    raw = log_data.get("ksd_scores", [])
    raw_arr   = np.array(raw, dtype=np.float64)
    valid_mask = (raw_arr >= 0.0) & (raw_arr <= 1.0)
    return {
        "raw_scores":        raw_arr,
        "valid_scores":      raw_arr[valid_mask],
        "n_raw":             len(raw_arr),
        "n_valid":           int(valid_mask.sum()),
        "has_legacy_scores": bool((raw_arr > 1.0).any()),
        "score_range":       (float(raw_arr.min()), float(raw_arr.max())) if len(raw_arr) > 0 else (0.0, 0.0),
    }


def parse_session_log_cv(log_data: dict) -> dict:
    """
    Parse the CV result records from session_log.json.

    Returns
    -------
    dict with:
        results          : list of cv dicts
        n_total          : total CV checks
        n_match          : CV returned True (face match)
        n_fail           : CV returned False
        n_error          : CV returned error (no face / no data)
        match_distances  : distances where match=True
        fail_distances   : distances where match=False (no error)
        timestamps       : list of epoch timestamps
    """
    cv_list = log_data.get("cv_results", [])
    n_match, n_fail, n_error = 0, 0, 0
    match_dists, fail_dists = [], []
    timestamps = []

    for r in cv_list:
        ts = r.get("timestamp")
        if ts:
            timestamps.append(float(ts))
        error = r.get("error")
        match = r.get("match", False)
        dist  = r.get("combined")

        if error and error not in (None, "null"):
            n_error += 1
        elif match:
            n_match += 1
            if dist is not None:
                match_dists.append(float(dist))
        else:
            n_fail += 1
            if dist is not None:
                fail_dists.append(float(dist))

    return {
        "results":       cv_list,
        "n_total":       len(cv_list),
        "n_match":       n_match,
        "n_fail":        n_fail,
        "n_error":       n_error,
        "match_distances": match_dists,
        "fail_distances":  fail_dists,
        "timestamps":    timestamps,
    }


def estimate_cv_timing_from_log(cv_data: dict) -> dict:
    """
    Use timestamp differences between consecutive CV records to estimate
    mean CV processing time (proxy for camera-active time per check).
    """
    ts = cv_data["timestamps"]
    if len(ts) < 2:
        return {"mean_cv_duration_s": None, "n_intervals": 0}

    intervals = np.diff(ts)
    # Filter out large gaps (>120 s) which represent inter-session pauses
    short = intervals[intervals < 120]
    if len(short) == 0:
        return {"mean_cv_duration_s": None, "n_intervals": 0}

    return {
        "mean_cv_duration_s": float(np.mean(short)),
        "median_cv_duration_s": float(np.median(short)),
        "std_cv_duration_s":  float(np.std(short)),
        "n_intervals": len(short),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Filesystem
# ═════════════════════════════════════════════════════════════════════════════

def ensure_results_dir() -> str:
    """Create experiments/results/ if it doesn't exist. Return its path."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return RESULTS_DIR


def save_json(data: dict, filename: str) -> str:
    """Save dict as JSON to experiments/results/<filename>. Return full path."""
    ensure_results_dir()
    path = os.path.join(RESULTS_DIR, filename)
    # Convert numpy types to plain Python for JSON serialisation
    def _convert(obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return obj
    import json as _json
    with open(path, "w", encoding="utf-8") as fh:
        _json.dump(data, fh, indent=2, default=_convert)
    return path
