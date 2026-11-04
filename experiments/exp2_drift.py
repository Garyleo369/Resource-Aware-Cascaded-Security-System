"""
experiments/exp2_drift.py
===========================
Experiment 2 — Static Profile vs Behavioral Drift
--------------------------------------------------

Goal
----
Determine how sensitive the current KSD system is to natural changes in the
user's typing behaviour, WITHOUT assuming every deviation is an attack.

Sub-experiments
---------------
A. Same-session split
   Train/profile established on the first half of the enrolment vectors.
   Test on the second half (intra-session variability).

B. Cross-session (simulated)
   Train on all enrolment vectors.
   Test on vectors from a behaviorally-shifted session (same user, different
   context — tired / different keyboard / stressed). Generated using the
   generate_shift() function calibrated from real enrolment data.

C. Cross-day analysis using session_log.json
   Parse the real KSD score stream from data/session_log.json.
   Check whether timestamps span multiple calendar days.
   Identify score pattern changes across time segments.
   (No retraining here — the point is to expose whether the fixed profile
   still works when the user's behaviour naturally changes.)

Metrics
-------
  • FAR, FRR, EER per sub-experiment
  • Anomaly score distributions (train vs test)
  • Number of false alarms (genuine windows flagged as anomalous)
  • Detection latency

Outputs (experiments/results/)
-------------------------------
  exp2_summary.json
  exp2_score_comparison.png
  exp2_far_frr_by_scenario.png
  exp2_session_log_timeline.png  (if session_log data is available)

Usage
-----
    python experiments/exp2_drift.py [--windows 200] [--seed 42]
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np

_EXP_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.dirname(_EXP_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from experiments.common import (
    V2_CONFIG,
    compute_eer,
    compute_metrics,
    compute_roc,
    compute_auc,
    ensure_results_dir,
    generate_genuine,
    generate_shift,
    generate_impostor,
    load_enrol_vectors,
    load_session_log,
    load_v2_scorer,
    parse_session_log_ksd,
    save_json,
    score_batch,
    RESULTS_DIR,
)

warnings.filterwarnings("ignore")


# ═════════════════════════════════════════════════════════════════════════════
# Sub-experiments
# ═════════════════════════════════════════════════════════════════════════════

def _run_sub(
    name: str,
    test_vecs: list[np.ndarray],
    label: int,
    scorer_fn,
    operational_threshold: float,
    verbose: bool = True,
) -> dict:
    """
    Score test_vecs through a fresh scorer and compute metrics.

    label : 0 = genuine (should NOT trigger), 1 = anomalous (SHOULD trigger)
    """
    scorer = scorer_fn()
    scores, thresholds = score_batch(scorer, test_vecs)
    n = len(test_vecs)
    y_true = np.full(n, label, dtype=int)

    m = compute_metrics(y_true, scores, operational_threshold)
    fpr, tpr, thr_arr = compute_roc(y_true, scores)
    eer, eer_thr = compute_eer(fpr, tpr, thr_arr)
    auc = compute_auc(fpr, tpr)

    false_alarms = m["FP"] if label == 0 else m["FN"]
    false_alarm_rate = false_alarms / n if n > 0 else 0.0

    if verbose:
        print(f"\n  [{name}]")
        print(f"    n_windows     : {n}")
        print(f"    label         : {'genuine (should not trigger)' if label == 0 else 'shift/impostor (should trigger)'}")
        print(f"    score mean    : {scores.mean():.4f}   std: {scores.std():.4f}")
        print(f"    false alarms  : {false_alarms} ({false_alarm_rate*100:.1f}%)")
        print(f"    FAR           : {m['FAR']:.4f}")
        print(f"    FRR           : {m['FRR']:.4f}")
        print(f"    EER           : {eer:.4f}  (thr={eer_thr:.4f})")

    return {
        "name":               name,
        "n_windows":          n,
        "label":              label,
        "score_mean":         round(float(scores.mean()), 4),
        "score_std":          round(float(scores.std()),  4),
        "score_min":          round(float(scores.min()),  4),
        "score_max":          round(float(scores.max()),  4),
        "score_p10":          round(float(np.percentile(scores, 10)), 4),
        "score_p50":          round(float(np.percentile(scores, 50)), 4),
        "score_p90":          round(float(np.percentile(scores, 90)), 4),
        "false_alarms":       false_alarms,
        "false_alarm_rate":   round(false_alarm_rate, 4),
        "FAR":                m["FAR"],
        "FRR":                m["FRR"],
        "EER":                round(eer, 4),
        "EER_threshold":      round(eer_thr, 4),
        "ROC_AUC":            round(auc, 4),
        "metrics":            m,
        "_scores":            scores.tolist(),
    }


def run(n_windows: int = 200, seed: int = 42, verbose: bool = True) -> dict:
    rng = np.random.default_rng(seed)

    _p = lambda *a: print(*a) if verbose else None
    _p("\n" + "=" * 65)
    _p("  EXPERIMENT 2 — Static Profile vs Behavioral Drift")
    _p("=" * 65)

    enrol_vecs = load_enrol_vectors()
    n_enrol    = len(enrol_vecs)
    _p(f"\n  Enrolment windows available : {n_enrol}")

    # Operational threshold derived from genuine typing steady-state
    # (same approach as Experiment 1 — consistent across experiments)
    scorer_full = load_v2_scorer()
    baseline_scores, _ = score_batch(scorer_full,
                                     list(generate_genuine(50, rng, enrol_vecs)))
    operational_threshold = float(np.median(baseline_scores)
                                  + V2_CONFIG["K"] * np.std(baseline_scores))
    operational_threshold = max(operational_threshold, V2_CONFIG["FLOOR"])
    _p(f"  Operational threshold       : {operational_threshold:.4f}")

    def fresh_scorer():
        return load_v2_scorer()

    # ── Sub-experiment A: Same-session split ──────────────────────────────────
    _p("\n  Sub-experiment A: Same-session split")
    _p("  ─────────────────────────────────────")
    _p(f"  Enrolment data split: first {n_enrol//2} → train basis, "
       f"last {n_enrol - n_enrol//2} → test")

    if n_enrol >= 4:
        split = n_enrol // 2
        # Test on second half of enrolment vectors (these are genuine — should NOT trigger)
        test_a_vecs = list(enrol_vecs[split:])
        # Pad with perturbation if too few
        if len(test_a_vecs) < 20:
            extra = generate_genuine(20 - len(test_a_vecs), rng, enrol_vecs[split:])
            test_a_vecs = test_a_vecs + extra
        sub_a = _run_sub(
            "A — Same-session (2nd half)",
            test_a_vecs, label=0, scorer_fn=fresh_scorer,
            operational_threshold=operational_threshold, verbose=verbose,
        )
    else:
        _p("  [WARN] Too few enrolment windows for split — generating synthetic same-session windows")
        test_a_vecs = generate_genuine(n_windows, rng, enrol_vecs)
        sub_a = _run_sub(
            "A — Same-session (synthetic 2nd half)",
            test_a_vecs, label=0, scorer_fn=fresh_scorer,
            operational_threshold=operational_threshold, verbose=verbose,
        )

    # ── Sub-experiment B: Cross-session behavioral shift ──────────────────────
    _p("\n  Sub-experiment B: Cross-session behavioral shift")
    _p("  ─────────────────────────────────────────────────")
    _p("  Shift parameters: dwell ×1.4–1.7, flight ×1.2–1.5,")
    _p("  typing_rate ×0.5–0.7, burst_ratio ×1.3–2.0")

    shift_vecs = generate_shift(n_windows, rng, enrol_vecs)
    # Shifted genuine user: label=0 (should ideally NOT be permanently locked)
    # But the current system has no adaptation — this measures false alarm rate
    sub_b = _run_sub(
        "B — Cross-session shift (legitimate user, different context)",
        shift_vecs, label=0, scorer_fn=fresh_scorer,
        operational_threshold=operational_threshold, verbose=verbose,
    )

    # ── Sub-experiment C: session_log.json cross-day analysis ─────────────────
    _p("\n  Sub-experiment C: Real session_log.json timeline analysis")
    _p("  ─────────────────────────────────────────────────────────")

    log_data  = load_session_log()
    ksd_info  = parse_session_log_ksd(log_data)
    raw_scores = ksd_info["raw_scores"]
    valid_scores = ksd_info["valid_scores"]

    _p(f"  KSD scores in log  : {ksd_info['n_raw']} total, "
       f"{ksd_info['n_valid']} in [0,1] range")
    _p(f"  Score range        : [{ksd_info['score_range'][0]:.4f}, {ksd_info['score_range'][1]:.4f}]")
    _p(f"  Legacy scores >1.0 : {ksd_info['has_legacy_scores']}")

    # Check CV timestamps for calendar-day span
    cv_timestamps = [r.get("timestamp") for r in log_data.get("cv_results", [])
                     if r.get("timestamp")]
    day_span_info = {}
    if len(cv_timestamps) >= 2:
        import datetime
        dates = [datetime.datetime.fromtimestamp(t).date()
                 for t in cv_timestamps]
        unique_days = sorted(set(dates))
        day_span_info = {
            "n_calendar_days": len(unique_days),
            "first_day":       str(unique_days[0]),
            "last_day":        str(unique_days[-1]),
            "days":            [str(d) for d in unique_days],
        }
        _p(f"  Calendar days spanned : {len(unique_days)}")
        for d in unique_days:
            _p(f"    {d}")
    else:
        day_span_info = {"n_calendar_days": 0, "note": "No timestamp data available"}
        _p("  No timestamp data available for day analysis")

    # Segment analysis of valid scores (if enough data)
    segment_info = {}
    if len(valid_scores) >= 10:
        mid = len(valid_scores) // 2
        seg1, seg2 = valid_scores[:mid], valid_scores[mid:]
        op_thr_counts_1 = int((seg1 >= operational_threshold).sum())
        op_thr_counts_2 = int((seg2 >= operational_threshold).sum())
        segment_info = {
            "segment_1": {
                "n": len(seg1),
                "mean": round(float(seg1.mean()), 4),
                "std":  round(float(seg1.std()),  4),
                "anomaly_rate_at_op_thr": round(op_thr_counts_1 / len(seg1), 4),
            },
            "segment_2": {
                "n": len(seg2),
                "mean": round(float(seg2.mean()), 4),
                "std":  round(float(seg2.std()),  4),
                "anomaly_rate_at_op_thr": round(op_thr_counts_2 / len(seg2), 4),
            },
        }
        _p(f"\n  Segment 1 (first {len(seg1)} valid scores):")
        _p(f"    mean={seg1.mean():.4f}  std={seg1.std():.4f}  anomaly_rate={op_thr_counts_1/len(seg1):.4f}")
        _p(f"  Segment 2 (last {len(seg2)} valid scores):")
        _p(f"    mean={seg2.mean():.4f}  std={seg2.std():.4f}  anomaly_rate={op_thr_counts_2/len(seg2):.4f}")

    sub_c = {
        "name":              "C — Real session_log timeline",
        "n_valid_ksd":       ksd_info["n_valid"],
        "n_raw_ksd":         ksd_info["n_raw"],
        "has_legacy_scores": ksd_info["has_legacy_scores"],
        "day_span":          day_span_info,
        "segments":          segment_info,
        "_valid_scores":     valid_scores.tolist(),
    }

    # ── Summary ───────────────────────────────────────────────────────────────
    _p("\n  Summary")
    _p("  ───────")
    _p(f"  Sub-A false alarm rate (same-session): {sub_a['false_alarm_rate']*100:.1f}%")
    _p(f"  Sub-B false alarm rate (cross-session shift): {sub_b['false_alarm_rate']*100:.1f}%")
    _p(f"  → Drift sensitivity: shift causes "
       f"{(sub_b['false_alarm_rate'] - sub_a['false_alarm_rate'])*100:.1f} pp "
       f"increase in false alarms")

    result = {
        "experiment":             "2 — Static Profile vs Behavioral Drift",
        "config": {
            **V2_CONFIG,
            "seed":      seed,
            "n_windows": n_windows,
            "n_enrol":   n_enrol,
        },
        "operational_threshold":  round(operational_threshold, 4),
        "sub_A_same_session":     {k: v for k, v in sub_a.items() if k != "_scores"},
        "sub_B_cross_session":    {k: v for k, v in sub_b.items() if k != "_scores"},
        "sub_C_real_log":         {k: v for k, v in sub_c.items() if k != "_valid_scores"},
        "drift_delta_false_alarm_pp": round(
            (sub_b["false_alarm_rate"] - sub_a["false_alarm_rate"]) * 100, 2
        ),
        "_plot_data": {
            "a_scores":      sub_a.get("_scores", []),
            "b_scores":      sub_b.get("_scores", []),
            "c_valid_scores": sub_c.get("_valid_scores", []),
        },
    }
    return result


# ═════════════════════════════════════════════════════════════════════════════
# Plotting
# ═════════════════════════════════════════════════════════════════════════════

def plot_results(result: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARN] matplotlib not available — skipping plots.")
        return

    ensure_results_dir()
    pd = result["_plot_data"]
    op_thr = result["operational_threshold"]

    a_scores = np.array(pd["a_scores"])
    b_scores = np.array(pd["b_scores"])
    c_scores = np.array(pd["c_valid_scores"])

    # ── Score comparison: A, B (and C if available) ──────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    bins = np.linspace(0, 1, 31)

    # Left: A vs B
    ax = axes[0]
    if len(a_scores) > 0:
        ax.hist(a_scores, bins=bins, alpha=0.65, color="#4CAF50", density=True,
                label=f"A — Same-session (n={len(a_scores)})")
    if len(b_scores) > 0:
        ax.hist(b_scores, bins=bins, alpha=0.65, color="#FF9800", density=True,
                label=f"B — Cross-session shift (n={len(b_scores)})")
    ax.axvline(op_thr, color="#F44336", linewidth=2, linestyle="--",
               label=f"Op. threshold ({op_thr:.3f})")
    ax.set_xlabel("Anomaly score", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Exp 2 — Genuine User Score Distributions\n(same-session vs behavioral shift)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)

    # Right: real session log scores over time
    ax2 = axes[1]
    if len(c_scores) > 0:
        ax2.plot(c_scores, color="#2196F3", linewidth=1, alpha=0.8, label="KSD scores (real log)")
        ax2.axhline(op_thr, color="#F44336", linewidth=2, linestyle="--",
                    label=f"Op. threshold ({op_thr:.3f})")
        ax2.set_xlabel("Window index (chronological)", fontsize=11)
        ax2.set_ylabel("Anomaly score", fontsize=11)
        ax2.set_title("Exp 2C — Real Session Log KSD Scores Over Time\n(valid [0,1] scores only)", fontsize=11, fontweight="bold")
        ax2.legend(fontsize=9)
        ax2.set_ylim(0, 1)
    else:
        ax2.text(0.5, 0.5, "No valid scores\nin session_log.json",
                 ha="center", va="center", transform=ax2.transAxes, fontsize=12)
        ax2.set_title("Exp 2C — Real Session Log (no valid data)", fontsize=11)

    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "exp2_score_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {path}")

    # ── FAR / FRR bar chart ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    sub_names = ["A — Same-session", "B — Cross-session shift"]
    far_vals  = [result["sub_A_same_session"]["FAR"],
                 result["sub_B_cross_session"]["FAR"]]
    frr_vals  = [result["sub_A_same_session"]["FRR"],
                 result["sub_B_cross_session"]["FRR"]]
    eer_vals  = [result["sub_A_same_session"]["EER"],
                 result["sub_B_cross_session"]["EER"]]

    x = np.arange(len(sub_names))
    width = 0.25
    ax.bar(x - width, far_vals,  width, label="FAR", color="#F44336", alpha=0.8)
    ax.bar(x,         frr_vals,  width, label="FRR", color="#2196F3", alpha=0.8)
    ax.bar(x + width, eer_vals,  width, label="EER", color="#9C27B0", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(sub_names, fontsize=11)
    ax.set_ylabel("Rate", fontsize=11)
    ax.set_title("Exp 2 — FAR / FRR / EER by Scenario", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1)
    for i, (f, r, e) in enumerate(zip(far_vals, frr_vals, eer_vals)):
        ax.text(i - width, f + 0.02, f"{f:.2f}", ha="center", fontsize=9)
        ax.text(i,         r + 0.02, f"{r:.2f}", ha="center", fontsize=9)
        ax.text(i + width, e + 0.02, f"{e:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "exp2_far_frr_by_scenario.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {path}")


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Exp 2 — Behavioral Drift")
    parser.add_argument("--windows", type=int, default=200)
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    result = run(n_windows=args.windows, seed=args.seed)
    plot_results(result)

    result_save = {k: v for k, v in result.items() if k != "_plot_data"}
    path = save_json(result_save, "exp2_summary.json")
    print(f"\n  Results saved → {path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
