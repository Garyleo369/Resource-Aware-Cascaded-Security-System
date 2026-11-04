"""
experiments/exp1_ksd_performance.py
====================================
Experiment 1 — KSD Anomaly Detection Performance
-------------------------------------------------

Goal
----
Evaluate how well the current IsolationForest (v2, cascaded_auth/ksd.py)
detects deviations from the legitimate user's typing behaviour.

Test sets
---------
  Genuine  : sampled from real enrolment vectors with 5% Gaussian noise.
  Impostor : vectors constructed 2.5–4σ from the enrolled mean.

Both sets are scored by the loaded model WITHOUT retraining — the model is
evaluated exactly as it runs in production.

Metrics
-------
  • TPR / Recall          (at operational threshold)
  • FPR                   (at operational threshold)
  • Precision             (at operational threshold)
  • F1-score              (at operational threshold)
  • FAR  (= FPR)
  • FRR  (= 1 − TPR)
  • ROC-AUC               (continuous sweep over all thresholds)
  • EER                   (threshold where FAR ≈ FRR)
  • Detection latency     (time to score one window, in ms)
  • Anomaly-score distributions for genuine vs impostor

Outputs (experiments/results/)
-------------------------------
  exp1_summary.json
  exp1_confusion_matrix.png
  exp1_roc_curve.png
  exp1_score_distributions.png

Usage
-----
    python experiments/exp1_ksd_performance.py [--windows 200] [--seed 42]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

import numpy as np

_EXP_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.dirname(_EXP_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from experiments.common import (
    V2_CONFIG,
    compute_auc,
    compute_eer,
    compute_metrics,
    compute_roc,
    ensure_results_dir,
    generate_genuine,
    generate_impostor,
    load_enrol_vectors,
    load_v2_scorer,
    save_json,
    score_batch_timed,
)

warnings.filterwarnings("ignore")


# ═════════════════════════════════════════════════════════════════════════════
# Core experiment
# ═════════════════════════════════════════════════════════════════════════════

def run(n_windows: int = 200, seed: int = 42, verbose: bool = True) -> dict:
    """
    Run Experiment 1 and return a result dict.

    Parameters
    ----------
    n_windows : number of genuine AND impostor windows to generate
    seed      : random seed for reproducibility
    verbose   : print progress to stdout
    """
    rng = np.random.default_rng(seed)

    _p = lambda *a: print(*a) if verbose else None
    _p("\n" + "=" * 65)
    _p("  EXPERIMENT 1 — KSD Anomaly Detection Performance")
    _p("=" * 65)

    # ── Load model and enrolment data ─────────────────────────────────────────
    _p("\n  Loading v2 IsolationForest model …")
    scorer      = load_v2_scorer()
    enrol_vecs  = load_enrol_vectors()
    n_enrol     = len(enrol_vecs)
    model_offset = float(scorer.model.offset_)

    _p(f"  Enrolment windows      : {n_enrol}")
    _p(f"  Model offset_          : {model_offset:.6f}")
    _p(f"  Feature dimension      : {V2_CONFIG['feature_dim']}")
    _p(f"  n_estimators           : {V2_CONFIG['n_estimators']}")
    _p(f"  contamination          : {V2_CONFIG['contamination']}")

    # ── Model health check on enrolment data ──────────────────────────────────
    raw_on_enrol = scorer.model.score_samples(enrol_vecs)
    norm_on_enrol = 1.0 / (1.0 + np.exp(10.0 * (raw_on_enrol - model_offset)))
    enrol_score_mean = float(norm_on_enrol.mean())
    enrol_score_max  = float(norm_on_enrol.max())
    _p(f"\n  Model health on enrolment data:")
    _p(f"    Normalised score mean : {enrol_score_mean:.4f}  (ideal ≤ 0.25)")
    _p(f"    Normalised score max  : {enrol_score_max:.4f}  (ideal ≤ 0.50)")
    if enrol_score_mean <= 0.25:
        health = "GOOD"
    elif enrol_score_mean <= 0.50:
        health = "ACCEPTABLE"
    else:
        health = "POOR"
    _p(f"    Health rating         : {health}")

    # ── Generate test sets ────────────────────────────────────────────────────
    _p(f"\n  Generating {n_windows} genuine windows …")
    genuine_vecs  = generate_genuine(n_windows, rng, enrol_vecs)

    _p(f"  Generating {n_windows} impostor windows …")
    impostor_vecs = generate_impostor(n_windows, rng, enrol_vecs)

    # ── Score both sets (fresh scorer per batch to avoid cross-contamination) ──
    _p("\n  Scoring genuine windows …")
    scorer_g = load_v2_scorer()
    g_scores, g_thresholds, g_time = score_batch_timed(scorer_g, genuine_vecs)

    _p("  Scoring impostor windows …")
    scorer_i = load_v2_scorer()
    i_scores, i_thresholds, i_time = score_batch_timed(scorer_i, impostor_vecs)

    total_windows = n_windows * 2
    all_scores    = np.concatenate([g_scores, i_scores])
    y_true        = np.array([0] * n_windows + [1] * n_windows, dtype=int)

    # ── Detection latency ─────────────────────────────────────────────────────
    latency_ms_per_window = (g_time + i_time) / total_windows * 1000.0

    # ── Operational threshold (as used by the current system) ─────────────────
    # The AnomalyScorer computes threshold() dynamically from its rolling score
    # buffer. We compute what the threshold settles to in steady-state for
    # genuine typing (warm-up complete).
    steady_g_threshold = float(np.median(g_scores) + V2_CONFIG["K"] * np.std(g_scores))
    steady_g_threshold = max(steady_g_threshold, V2_CONFIG["FLOOR"])
    operational_threshold = steady_g_threshold

    _p(f"\n  Operational threshold (genuine steady-state) : {operational_threshold:.4f}")
    _p(f"  (Floor = {V2_CONFIG['FLOOR']}, K = {V2_CONFIG['K']})")

    # ── Metrics at operational threshold ─────────────────────────────────────
    metrics_op = compute_metrics(y_true, all_scores, operational_threshold)

    # Also compute at the FLOOR threshold
    metrics_floor = compute_metrics(y_true, all_scores, V2_CONFIG["FLOOR"])

    # ── ROC curve and AUC ─────────────────────────────────────────────────────
    fpr, tpr, roc_thresholds = compute_roc(y_true, all_scores)
    auc = compute_auc(fpr, tpr)
    eer, eer_threshold = compute_eer(fpr, tpr, roc_thresholds)

    _p(f"\n  ROC-AUC                : {auc:.4f}")
    _p(f"  EER                    : {eer:.4f}  (at threshold={eer_threshold:.4f})")

    _p(f"\n  Metrics at operational threshold ({operational_threshold:.4f}):")
    for k in ("TPR", "FPR", "precision", "F1", "FAR", "FRR", "accuracy"):
        _p(f"    {k:<12}: {metrics_op[k]:.4f}")

    _p(f"\n  Detection latency      : {latency_ms_per_window:.3f} ms/window")
    _p(f"  (window = {V2_CONFIG['window_size_keys']} keystrokes, ~{V2_CONFIG['window_size_keys']*0.15:.0f}–{V2_CONFIG['window_size_keys']*0.35:.0f} s at typical speeds)")

    # ── Build result dict ─────────────────────────────────────────────────────
    result = {
        "experiment": "1 — KSD Anomaly Detection Performance",
        "config": {
            **V2_CONFIG,
            "seed":      seed,
            "n_genuine": n_windows,
            "n_impostor": n_windows,
            "n_enrol":   n_enrol,
        },
        "model_health": {
            "enrol_score_mean": round(enrol_score_mean, 4),
            "enrol_score_max":  round(enrol_score_max,  4),
            "rating":           health,
            "model_offset":     round(model_offset, 6),
        },
        "score_distributions": {
            "genuine": {
                "mean":   round(float(g_scores.mean()), 4),
                "std":    round(float(g_scores.std()),  4),
                "min":    round(float(g_scores.min()),  4),
                "max":    round(float(g_scores.max()),  4),
                "p10":    round(float(np.percentile(g_scores, 10)), 4),
                "p50":    round(float(np.percentile(g_scores, 50)), 4),
                "p90":    round(float(np.percentile(g_scores, 90)), 4),
            },
            "impostor": {
                "mean":   round(float(i_scores.mean()), 4),
                "std":    round(float(i_scores.std()),  4),
                "min":    round(float(i_scores.min()),  4),
                "max":    round(float(i_scores.max()),  4),
                "p10":    round(float(np.percentile(i_scores, 10)), 4),
                "p50":    round(float(np.percentile(i_scores, 50)), 4),
                "p90":    round(float(np.percentile(i_scores, 90)), 4),
            },
        },
        "operational_threshold":    round(operational_threshold, 4),
        "metrics_at_operational":   metrics_op,
        "metrics_at_floor":         metrics_floor,
        "roc_auc":                  round(auc, 4),
        "eer":                      round(eer, 4),
        "eer_threshold":            round(eer_threshold, 4),
        "detection_latency_ms":     round(latency_ms_per_window, 4),
        # Store arrays for plotting (as lists)
        "_plot_data": {
            "g_scores":       g_scores.tolist(),
            "i_scores":       i_scores.tolist(),
            "roc_fpr":        fpr.tolist(),
            "roc_tpr":        tpr.tolist(),
            "roc_thresholds": roc_thresholds.tolist(),
        },
    }

    return result


# ═════════════════════════════════════════════════════════════════════════════
# Plotting
# ═════════════════════════════════════════════════════════════════════════════

def plot_results(result: dict) -> None:
    """Generate and save the three plots for Experiment 1."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("  [WARN] matplotlib not available — skipping plots.")
        return

    ensure_results_dir()
    from experiments.common import RESULTS_DIR

    pd = result["_plot_data"]
    g_scores = np.array(pd["g_scores"])
    i_scores = np.array(pd["i_scores"])
    fpr      = np.array(pd["roc_fpr"])
    tpr      = np.array(pd["roc_tpr"])
    op_thr   = result["operational_threshold"]

    # ── 1. Score distributions ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bins = np.linspace(0, 1, 41)
    ax.hist(g_scores, bins=bins, alpha=0.6, color="#2196F3", label="Genuine",  density=True)
    ax.hist(i_scores, bins=bins, alpha=0.6, color="#F44336", label="Impostor", density=True)
    ax.axvline(op_thr, color="#FF9800", linewidth=2, linestyle="--",
               label=f"Op. threshold ({op_thr:.3f})")
    ax.axvline(result["eer_threshold"], color="#9C27B0", linewidth=1.5, linestyle=":",
               label=f"EER threshold ({result['eer_threshold']:.3f})")
    ax.set_xlabel("Anomaly score (v2 IsolationForest)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Exp 1 — Anomaly Score Distributions: Genuine vs Impostor", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "exp1_score_distributions.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {path}")

    # ── 2. ROC curve ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="#2196F3", linewidth=2,
            label=f"ROC (AUC = {result['roc_auc']:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC = 0.50)")

    # Mark EER point
    eer_val = result["eer"]
    ax.plot(eer_val, 1 - eer_val, "o", color="#F44336", markersize=8,
            label=f"EER = {eer_val:.4f}")

    # Mark operational threshold point
    m = result["metrics_at_operational"]
    ax.plot(m["FPR"], m["TPR"], "s", color="#FF9800", markersize=8,
            label=f"Op. threshold (FPR={m['FPR']:.3f}, TPR={m['TPR']:.3f})")

    ax.set_xlabel("False Positive Rate (FAR)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
    ax.set_title("Exp 1 — ROC Curve (KSD v2 IsolationForest)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "exp1_roc_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {path}")

    # ── 3. Confusion matrix ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 4.5))
    m = result["metrics_at_operational"]
    cm = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])
    im = ax.imshow(cm, cmap="Blues", vmin=0)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted Normal", "Predicted Anomaly"], fontsize=10)
    ax.set_yticklabels(["Actual Genuine", "Actual Impostor"], fontsize=10)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=16, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() * 0.6 else "black")
    ax.set_title(
        f"Exp 1 — Confusion Matrix\n"
        f"(threshold={result['operational_threshold']:.3f}, "
        f"F1={m['F1']:.3f})",
        fontsize=11, fontweight="bold"
    )
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "exp1_confusion_matrix.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {path}")


# ═════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Exp 1 — KSD Anomaly Detection Performance")
    parser.add_argument("--windows", type=int, default=200,
                        help="Number of genuine / impostor windows (default: 200)")
    parser.add_argument("--seed",    type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    result = run(n_windows=args.windows, seed=args.seed)
    plot_results(result)

    # Strip plot data before saving JSON (arrays are large)
    result_save = {k: v for k, v in result.items() if k != "_plot_data"}
    path = save_json(result_save, "exp1_summary.json")
    print(f"\n  Results saved → {path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
