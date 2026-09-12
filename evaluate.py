"""
evaluate.py — Offline Accuracy Evaluation for the Cascaded Authentication System
==================================================================================

Tests the system against the five Expected System Operational Outcomes:

  Scenario 1 — Legitimate User (Normal Typing)
  Scenario 2 — Legitimate User (Behavioral Shift)
  Scenario 3 — Unauthorized User (Intruder)
  Scenario 4 — Legitimate User (Extended Stability)
  Scenario 5 — Unauthorized User (Intruder) with CV confirmation

For each scenario synthetic keystroke feature windows are generated that mimic
the statistical fingerprint of the enrolled user or an intruder.  The KSD
pipeline (AnomalyScorer + dynamic threshold) is driven offline — no webcam,
no pynput listener.

CV is mocked so the evaluation is fully automated:
  - legitimate user sessions → mock_cv_result = True  (face matches)
  - intruder sessions        → mock_cv_result = False (face does NOT match)

Usage
-----
    python evaluate.py                     # uses cascaded_auth/model.joblib
    python evaluate.py --windows 200       # run more test windows per scenario

Outputs
-------
  • Per-scenario pass/fail table printed to the console
  • Accuracy, Precision, Recall, F1, FAR, FRR metrics
  • Results saved to  evaluation_results.json

Requirements
------------
  Enrolment must have been run at least once so that
  cascaded_auth/model.joblib  and  cascaded_auth/enrol_vectors.joblib  exist.
"""

import sys
import os
import json
import argparse
import numpy as np
import time

# ── Path resolution ──────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

from cascaded_auth.ksd import AnomalyScorer, extract_features, WINDOW, RECAL_EVERY

# ── Evaluation constants ─────────────────────────────────────────────────────
CONSECUTIVE_ANOMALY_TRIGGER = 2   # windows in a row before CV fires (matches monitor.py)
STABILITY_WINDOW_COUNT      = 96  # windows representing ~48 hours of stable use
STABILITY_RETRAIN_THRESHOLD = 0.30 # retraining triggered if mean score stays below this


# ════════════════════════════════════════════════════════════════════════════
# Synthetic data generators
# ════════════════════════════════════════════════════════════════════════════

# ── Calibration: load real enrolment vectors to derive feature distributions ──
_ENROL_FILE = os.path.join(_ROOT, "cascaded_auth", "enrol_vectors.joblib")


def _load_enrol_stats():
    """
    Read cascaded_auth/enrol_vectors.joblib and return (mean_vec, std_vec)
    so synthetic generators can match the real enrolled distribution.
    Returns None if the file does not exist.
    """
    try:
        import joblib as _jl
        vecs = _jl.load(_ENROL_FILE)   # shape (N, 12)
        return vecs.mean(axis=0), vecs.std(axis=0), vecs
    except Exception:
        return None, None, None


_ENROL_MEAN, _ENROL_STD, _ENROL_VECS = _load_enrol_stats()


def _make_feature_vector_direct(feature_vec: np.ndarray) -> np.ndarray:
    """Return a 12-D feature vector with a small random perturbation applied."""
    assert feature_vec.shape == (12,)
    return feature_vec.copy()


def _perturb(vec: np.ndarray, noise_scale: float, rng: np.random.Generator) -> np.ndarray:
    """Add Gaussian noise to a feature vector (scale relative to each feature)."""
    noise = rng.normal(0, noise_scale, size=vec.shape)
    return (vec + vec * noise).clip(0.0)


def generate_owner_windows(n: int, rng: np.random.Generator) -> list:
    """
    Simulate the enrolled user typing normally.
    If real enrolment vectors exist, sample from them with small perturbation.
    Otherwise fall back to hardcoded typical values.
    """
    if _ENROL_VECS is not None and len(_ENROL_VECS) > 0:
        # Sample real enrolment vectors with slight random perturbation (5% noise)
        indices = rng.integers(0, len(_ENROL_VECS), size=n)
        return [_perturb(_ENROL_VECS[i], 0.05, rng) for i in indices]

    # Fallback (no enrolment file found)
    return [_build_fallback_owner(rng) for _ in range(n)]


def _build_fallback_owner(rng):
    """Hardcoded fallback owner window (used only if enrol_vectors.joblib missing)."""
    dwells  = rng.normal(0.113, 0.030, WINDOW).clip(0.02, 1.0)
    flights = np.abs(rng.normal(0.09, 0.15, WINDOW))   # heavy-tailed like real data
    def _s(a): return [float(f(a)) for f in (np.mean, np.std,
                        lambda x: np.percentile(x,10),
                        lambda x: np.percentile(x,50),
                        lambda x: np.percentile(x,90))]
    vec = np.array(_s(dwells) + _s(flights) + [4.1, 1.6], dtype=np.float64)
    return vec


def generate_behavioral_shift_windows(n: int, rng: np.random.Generator) -> list:
    """
    Simulate the legitimate user under a temporary behavioural shift
    (tired, different keyboard, different posture).
    Scores are elevated but the face should still pass CV.
    Based on the real enrolled distribution scaled up by ~30-60%%.
    """
    if _ENROL_VECS is not None and len(_ENROL_VECS) > 0:
        # Take real vectors but perturb them significantly (25%% noise + slow-down)
        indices = rng.integers(0, len(_ENROL_VECS), size=n)
        windows = []
        # Pick consistent shift parameters for the whole tired session
        dwell_scale  = rng.uniform(1.4, 1.7)
        flight_scale = rng.uniform(1.2, 1.5)
        rate_scale   = rng.uniform(0.5, 0.7)
        burst_scale  = rng.uniform(1.3, 2.0)

        for i in indices:
            v = _ENROL_VECS[i].copy()
            v[0:5]  = v[0:5]  * dwell_scale
            v[5:10] = v[5:10] * flight_scale
            v[10]   = v[10]   * rate_scale
            v[11]   = v[11]   * burst_scale
            # Add back 5% natural noise so the windows aren't rigidly identical
            v = _perturb(v, 0.05, rng)
            windows.append(v.clip(0.0))
        return windows

    # Fallback
    return [_build_fallback_shift(rng) for _ in range(n)]


def _build_fallback_shift(rng):
    dwells  = rng.normal(0.170, 0.055, WINDOW).clip(0.02, 1.0)
    flights = np.abs(rng.normal(0.18, 0.30, WINDOW))
    def _s(a): return [float(f(a)) for f in (np.mean, np.std,
                        lambda x: np.percentile(x,10),
                        lambda x: np.percentile(x,50),
                        lambda x: np.percentile(x,90))]
    vec = np.array(_s(dwells) + _s(flights) + [2.0, 3.5], dtype=np.float64)
    return vec


def generate_intruder_windows(n: int, rng: np.random.Generator) -> list:
    """
    Simulate an intruder whose typing pattern is significantly different.
    CV check will return False (no face match).
    Uses feature values well outside the enrolled distribution.
    """
    if _ENROL_VECS is not None and len(_ENROL_VECS) > 0:
        # Generate vectors that are 2-3 standard deviations from enrolled mean
        # Generate the consistent profile for this specific intruder
        intruder_profile = _ENROL_MEAN.copy()
        direction = rng.choice([-1, 1], size=12)
        shift = _ENROL_STD * rng.uniform(2.5, 4.0, size=12) * direction
        intruder_profile = (intruder_profile + shift).clip(0.0)
        intruder_profile[0] = abs(_ENROL_MEAN[0]) * rng.uniform(1.8, 3.0)
        intruder_profile[10] = abs(_ENROL_MEAN[10]) * rng.uniform(0.3, 0.6)

        windows = []
        for _ in range(n):
            # The intruder types with their own consistent pattern + some natural noise
            v = _perturb(intruder_profile, 0.05, rng)
            windows.append(v.clip(0.0))
        return windows

    # Fallback
    return [_build_fallback_intruder(rng) for _ in range(n)]


def _build_fallback_intruder(rng):
    dwells  = rng.normal(0.250, 0.100, WINDOW).clip(0.02, 2.0)
    flights = np.abs(rng.normal(0.30, 0.50, WINDOW))
    def _s(a): return [float(f(a)) for f in (np.mean, np.std,
                        lambda x: np.percentile(x,10),
                        lambda x: np.percentile(x,50),
                        lambda x: np.percentile(x,90))]
    vec = np.array(_s(dwells) + _s(flights) + [1.0, 5.0], dtype=np.float64)
    return vec


def generate_stability_windows(n: int, rng: np.random.Generator) -> list:
    """Simulate the legitimate user over an extended, stable typing period."""
    if _ENROL_VECS is not None and len(_ENROL_VECS) > 0:
        # Very tight perturbation (2%% noise) to simulate a stable session
        indices = rng.integers(0, len(_ENROL_VECS), size=n)
        return [_perturb(_ENROL_VECS[i], 0.02, rng) for i in indices]

    # Fallback
    return [_build_fallback_stability(rng) for _ in range(n)]


def _build_fallback_stability(rng):
    dwells  = rng.normal(0.112, 0.007, WINDOW).clip(0.02, 1.0)
    flights = np.abs(rng.normal(0.09, 0.10, WINDOW))
    def _s(a): return [float(f(a)) for f in (np.mean, np.std,
                        lambda x: np.percentile(x,10),
                        lambda x: np.percentile(x,50),
                        lambda x: np.percentile(x,90))]
    vec = np.array(_s(dwells) + _s(flights) + [4.2, 1.5], dtype=np.float64)
    return vec


# ════════════════════════════════════════════════════════════════════════════
# Scenario runner helpers
# ════════════════════════════════════════════════════════════════════════════

class MockCVResult:
    """Provides a deterministic CV result without opening the webcam."""
    def __init__(self, always_match: bool):
        self.always_match = always_match
        self.call_count   = 0

    def __call__(self) -> bool:
        self.call_count += 1
        return self.always_match


def _run_ksd_pipeline(
    windows,
    scorer,
    mock_cv,
    consec_required = CONSECUTIVE_ANOMALY_TRIGGER,
):
    """
    Drive the KSD anomaly scorer through a list of pre-computed feature windows.
    Returns a dict with per-window decisions and aggregate counts.
    """
    consecutive_anomalies = 0
    last_anomaly_vec      = None
    cv_triggers           = 0
    cv_passes             = 0
    cv_fails              = 0
    locked                = False
    lock_events           = []
    window_details        = []

    for idx, vec in enumerate(windows):
        score   = scorer.score(vec)
        thresh  = scorer.threshold()
        is_anom = scorer.is_anomaly(score)

        if is_anom:
            consecutive_anomalies += 1
            last_anomaly_vec       = vec
        else:
            consecutive_anomalies = 0
            last_anomaly_vec      = None

        cv_triggered_this_window = False
        cv_result_this_window    = None

        if consecutive_anomalies >= consec_required:
            cv_triggers += 1
            cv_triggered_this_window = True
            cv_result = mock_cv()
            cv_result_this_window = cv_result
            consecutive_anomalies = 0

            if cv_result:
                cv_passes += 1
                if last_anomaly_vec is not None:
                    scorer.add_confirmed(last_anomaly_vec)
            else:
                cv_fails += 1
                locked = True
                lock_events.append(idx)

        window_details.append({
            "window_idx":   idx,
            "score":        round(float(score), 4),
            "threshold":    round(float(thresh), 4),
            "is_anomaly":   bool(is_anom),
            "cv_triggered": cv_triggered_this_window,
            "cv_passed":    cv_result_this_window,
        })

    return {
        "total_windows":   len(windows),
        "anomaly_windows": sum(1 for w in window_details if w["is_anomaly"]),
        "cv_triggers":     cv_triggers,
        "cv_passes":       cv_passes,
        "cv_fails":        cv_fails,
        "locked":          locked,
        "lock_events":     lock_events,
        "window_details":  window_details,
        "mean_score":      float(np.mean([w["score"] for w in window_details])),
        "max_score":       float(np.max( [w["score"] for w in window_details])),
    }


# ════════════════════════════════════════════════════════════════════════════
# Five scenario definitions
# ════════════════════════════════════════════════════════════════════════════

def scenario_1_normal_typing(scorer, n_windows, rng):
    """
    Scenario 1 — Legitimate User (Normal Typing)
    Expected: session continues transparently, no CV triggered, no lock.
    """
    windows = generate_owner_windows(n_windows, rng)
    mock_cv = MockCVResult(always_match=True)
    result  = _run_ksd_pipeline(windows, scorer, mock_cv)

    false_lock_rate = result["locked"]
    cv_trigger_rate = result["cv_triggers"] / max(result["total_windows"], 1)
    passes = not false_lock_rate and cv_trigger_rate <= 0.05

    return {
        "scenario":         "1 — Legitimate User (Normal Typing)",
        "expected_outcome": "Session continues transparently, no lock",
        "passed":           passes,
        "false_lock":       false_lock_rate,
        "cv_trigger_rate":  round(cv_trigger_rate, 4),
        **{k: result[k] for k in ("total_windows","anomaly_windows","cv_triggers","cv_fails","mean_score","max_score")},
    }


def scenario_2_behavioral_shift(scorer, n_windows, rng):
    """
    Scenario 2 — Legitimate User (Behavioral Shift)
    Expected: CV activates and confirms identity; session continues, no lock.
    """
    warm_up  = generate_owner_windows(max(5, n_windows // 4), rng)
    shifted  = generate_behavioral_shift_windows(n_windows, rng)
    recovery = generate_owner_windows(max(5, n_windows // 4), rng)
    windows  = warm_up + shifted + recovery

    mock_cv = MockCVResult(always_match=True)
    result  = _run_ksd_pipeline(windows, scorer, mock_cv)

    cv_triggered_at_least_once = result["cv_triggers"] >= 1
    no_lock = not result["locked"]
    passes  = cv_triggered_at_least_once and no_lock

    return {
        "scenario":              "2 — Legitimate User (Behavioral Shift)",
        "expected_outcome":      "CV activates, confirms identity, session continues",
        "passed":                passes,
        "false_lock":            result["locked"],
        "cv_triggered_once":     cv_triggered_at_least_once,
        **{k: result[k] for k in ("total_windows","anomaly_windows","cv_triggers","cv_passes","cv_fails","mean_score","max_score")},
    }


def scenario_3_intruder(scorer, n_windows, rng):
    """
    Scenario 3 — Unauthorized User (Intruder)
    Expected: anomaly detected, CV triggered, CV returns False, screen locked.
    """
    warm_up  = generate_owner_windows(max(5, n_windows // 4), rng)
    intruder = generate_intruder_windows(n_windows, rng)
    windows  = warm_up + intruder

    mock_cv = MockCVResult(always_match=False)
    result  = _run_ksd_pipeline(windows, scorer, mock_cv)

    passes = result["locked"] and result["cv_fails"] >= 1

    return {
        "scenario":         "3 — Unauthorized User (Intruder)",
        "expected_outcome": "Screen locked, security log generated",
        "passed":           passes,
        "correctly_locked": result["locked"],
        **{k: result[k] for k in ("total_windows","anomaly_windows","cv_triggers","cv_passes","cv_fails","mean_score","max_score")},
    }


def scenario_4_extended_stability(scorer, rng):
    """
    Scenario 4 — Legitimate User (Extended Stability)
    Expected: scores stay below stability threshold; retraining is triggered.
    """
    windows = generate_stability_windows(STABILITY_WINDOW_COUNT, rng)
    mock_cv = MockCVResult(always_match=True)

    scores               = []
    retraining_triggered = False

    for idx, vec in enumerate(windows):
        score = scorer.score(vec)
        scores.append(score)

        if idx > 0 and idx % 48 == 0:
            recent_mean = float(np.mean(scores[-48:]))
            if recent_mean < STABILITY_RETRAIN_THRESHOLD:
                retraining_triggered = True
                if len(scores) >= RECAL_EVERY:
                    scorer.add_confirmed(vec)

    mean_score   = float(np.mean(scores))
    stable_ratio = sum(1 for s in scores if s < STABILITY_RETRAIN_THRESHOLD) / len(scores)
    passes       = stable_ratio >= 0.80 and retraining_triggered

    return {
        "scenario":             "4 — Legitimate User (Extended Stability)",
        "expected_outcome":     "Lightweight retraining performed, session uninterrupted",
        "passed":               passes,
        "mean_score":           round(mean_score, 4),
        "stable_ratio":         round(stable_ratio, 4),
        "retraining_triggered": retraining_triggered,
        "total_windows":        len(windows),
        "cv_triggers":          0,
    }


def scenario_5_intruder_cv_confirmed(scorer, n_windows, rng):
    """
    Scenario 5 — Unauthorized User (CV Compound Confirmation)
    Expected: screen immediately locked via compound KSD + CV signal.
    """
    warm_up  = generate_owner_windows(max(5, n_windows // 4), rng)
    intruder = generate_intruder_windows(n_windows, rng)
    windows  = warm_up + intruder

    mock_cv = MockCVResult(always_match=False)
    result  = _run_ksd_pipeline(windows, scorer, mock_cv)

    # Measure how elevated the intruder scores are independently
    temp_scorer = AnomalyScorer()
    try:
        temp_scorer.load()
    except FileNotFoundError:
        pass

    intruder_scores = [temp_scorer.score(v) for v in intruder]
    mean_intruder_score = float(np.mean(intruder_scores)) if intruder_scores else 0.0

    passes = result["locked"] and mean_intruder_score >= 0.45

    return {
        "scenario":            "5 — Unauthorized User (CV Compound Confirmation)",
        "expected_outcome":    "Screen immediately locked, quiet security log entry",
        "passed":              passes,
        "correctly_locked":    result["locked"],
        "mean_intruder_score": round(mean_intruder_score, 4),
        **{k: result[k] for k in ("total_windows","anomaly_windows","cv_triggers","cv_fails","mean_score","max_score")},
    }


# ════════════════════════════════════════════════════════════════════════════
# Aggregate metrics
# ════════════════════════════════════════════════════════════════════════════

def compute_metrics(results):
    """
    Compute system-level accuracy metrics across all scenarios.

    Positive (P) = event that SHOULD lock (intruder).
    Negative (N) = event that SHOULD NOT lock (legitimate user).
    """
    legit_scenarios    = [0, 1, 3]   # Scenarios 1, 2, 4
    intruder_scenarios = [2, 4]      # Scenarios 3, 5

    TP = sum(1 for i in intruder_scenarios if results[i].get("correctly_locked", results[i].get("passed", False)))
    FN = len(intruder_scenarios) - TP
    TN = sum(1 for i in legit_scenarios if not results[i].get("false_lock", False))
    FP = len(legit_scenarios) - TN

    accuracy  = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    far       = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    frr       = FN / (FN + TP) if (FN + TP) > 0 else 0.0

    return {
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "accuracy":  round(accuracy,  4),
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1_score":  round(f1,        4),
        "FAR":       round(far,       4),
        "FRR":       round(frr,       4),
    }


# ════════════════════════════════════════════════════════════════════════════
# Pretty printer
# ════════════════════════════════════════════════════════════════════════════

_W = 72

def _bar(label, value, width=30):
    filled = int(round(value * width))
    bar    = "#" * filled + "-" * (width - filled)
    return f"  {label:<22} [{bar}] {value * 100:6.1f}%"


def print_report(scenario_results, metrics):
    sep  = "=" * _W
    dash = "-" * _W

    print(f"\n{sep}")
    print(f"  CASCADED AUTHENTICATION SYSTEM -- EVALUATION REPORT")
    print(f"  Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(sep)

    for res in scenario_results:
        status = "[PASS]" if res["passed"] else "[FAIL]"
        print(f"\n  {status}  {res['scenario']}")
        print(f"  Expected : {res['expected_outcome']}")
        print(dash)

        skip = {"scenario", "expected_outcome", "passed", "window_details", "lock_events"}
        for k, v in res.items():
            if k in skip:
                continue
            if isinstance(v, float):
                print(f"    {k:<34} {v:.4f}")
            elif isinstance(v, bool):
                print(f"    {k:<34} {'Yes' if v else 'No'}")
            else:
                print(f"    {k:<34} {v}")

    print(f"\n{sep}")
    print("  AGGREGATE METRICS")
    print(sep)
    print(f"\n  Confusion Matrix")
    print(f"    True Positives  (TP): {metrics['TP']:>4}  intruder correctly locked")
    print(f"    True Negatives  (TN): {metrics['TN']:>4}  legitimate user not locked")
    print(f"    False Positives (FP): {metrics['FP']:>4}  legitimate user wrongly locked")
    print(f"    False Negatives (FN): {metrics['FN']:>4}  intruder not caught")
    print()
    print(_bar("Accuracy",  metrics["accuracy"]))
    print(_bar("Precision", metrics["precision"]))
    print(_bar("Recall",    metrics["recall"]))
    print(_bar("F1 Score",  metrics["f1_score"]))
    print()
    print(f"  {'FAR (False Acceptance Rate)':<34} {metrics['FAR'] * 100:6.1f}%  intruders let through")
    print(f"  {'FRR (False Rejection Rate)':<34} {metrics['FRR'] * 100:6.1f}%  legit users locked out")
    print(f"\n{sep}\n")

    overall = all(r["passed"] for r in scenario_results)
    if overall:
        print("  All 5 scenarios PASSED -- system behaves as expected.\n")
    else:
        failed = [r["scenario"] for r in scenario_results if not r["passed"]]
        print(f"  WARNING: {len(failed)} scenario(s) FAILED:")
        for f in failed:
            print(f"       * {f}")
        print()


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Offline accuracy evaluation for the Cascaded Authentication System"
    )
    parser.add_argument(
        "--windows", type=int, default=100,
        help="Synthetic keystroke windows per scenario (default: 100)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--output", type=str, default="evaluation_results.json",
        help="Output JSON file path (default: evaluation_results.json)"
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print(f"\n{'='*72}")
    print(f"  Loading KSD model from cascaded_auth/model.joblib …")
    print(f"{'='*72}")

    # Quick load test
    test_scorer = AnomalyScorer()
    try:
        test_scorer.load()
        print("  [OK] Model loaded successfully.\n")
    except FileNotFoundError as exc:
        print(f"\n  [ERROR] {exc}")
        print("  Run  python cascaded_auth/enrol.py  first to create the model.\n")
        sys.exit(1)

    def fresh_scorer():
        s = AnomalyScorer()
        s.load()
        return s

    # ── Model health-check diagnostic ─────────────────────────────────────────
    import joblib as _jl
    print("  MODEL DIAGNOSTIC")
    print("  " + "-" * 68)

    n_enrol = 0
    if _ENROL_VECS is not None:
        n_enrol = len(_ENROL_VECS)
        raw_scores = test_scorer.model.score_samples(_ENROL_VECS)
        offset = float(test_scorer.model.offset_)
        enrol_scores_normalised = [
            float(1.0 / (1.0 + np.exp(10.0 * (r - offset)))) for r in raw_scores
        ]
        print(f"  Enrolment windows        : {n_enrol}")
        print(f"  Raw IF score range       : [{raw_scores.min():.4f}, {raw_scores.max():.4f}]")
        print(f"  Normalised score on enrol: mean={np.mean(enrol_scores_normalised):.4f}  "
              f"max={np.max(enrol_scores_normalised):.4f}")
        print(f"  (Ideal: normalised scores on enrolment data should be near 0.0)")
    else:
        print("  [WARN] enrol_vectors.joblib not found -- cannot probe model baseline.")

    if n_enrol < 30:
        print()
        print("  [!] WARNING: Only", n_enrol, "windows were collected during enrolment.")
        print("  [!] The IsolationForest requires at LEAST 30-50 windows (ideally 100+)")
        print("  [!] to build a reliable baseline. With too few samples the model marks")
        print("  [!] almost everything as anomalous (even the owner's own typing).")
        print()
        print("  [!] RECOMMENDED FIX:")
        print("  [!]   Re-run  python cascaded_auth/enrol.py  and type for at least")
        print("  [!]   5-10 minutes continuously to collect 50+ windows.")
        print("  [!]   Results below reflect the CURRENT (undertrained) model state.")
    print("  " + "-" * 68 + "\n")

    print(f"  Running {args.windows} windows per scenario  (seed={args.seed})\n")
    t0 = time.time()


    print("  [1/5] Scenario 1 -- Normal Typing ...")
    r1 = scenario_1_normal_typing(fresh_scorer(), args.windows, rng)

    print("  [2/5] Scenario 2 -- Behavioral Shift ...")
    r2 = scenario_2_behavioral_shift(fresh_scorer(), args.windows, rng)

    print("  [3/5] Scenario 3 -- Intruder ...")
    r3 = scenario_3_intruder(fresh_scorer(), args.windows, rng)

    print("  [4/5] Scenario 4 -- Extended Stability ...")
    r4 = scenario_4_extended_stability(fresh_scorer(), rng)

    print("  [5/5] Scenario 5 -- Intruder (CV Compound) ...")
    r5 = scenario_5_intruder_cv_confirmed(fresh_scorer(), args.windows, rng)

    elapsed = time.time() - t0
    print(f"\n  Evaluation completed in {elapsed:.2f} s")

    scenario_results = [r1, r2, r3, r4, r5]
    metrics          = compute_metrics(scenario_results)

    print_report(scenario_results, metrics)

    output_path = os.path.join(_ROOT, args.output)
    payload = {
        "timestamp":             time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed":                  args.seed,
        "windows_per_scenario":  args.windows,
        "scenario_results": [
            {k: v for k, v in r.items() if k != "window_details"}
            for r in scenario_results
        ],
        "metrics": metrics,
    }
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(f"  Results saved -> {output_path}\n")


if __name__ == "__main__":
    main()
