"""
experiments/exp3_always_on_vs_event_driven.py
==============================================
Experiment 3 — Always-On CV vs Event-Driven CV
-----------------------------------------------

Goal
----
Evaluate whether the current architecture benefits from activating computer
vision only after a KSD anomaly instead of continuously running the CV system.

System Configurations
---------------------
  System A — Always-On
    KSD scoring every window + CV check on EVERY window.

  System B — Event-Driven  (current monitor.py behaviour)
    KSD scoring every window + CV triggered only when
    AnomalyScorer.is_anomaly() returns True.

The same test sessions (genuine + impostor) are fed to both configurations.
CV is mocked deterministically (genuine→True, impostor→False) so results
are reproducible without a webcam.

Resource usage
--------------
  CPU/RAM: measured with psutil during the KSD scoring loop (actual).
  CV processing time: estimated from real CV timestamps in session_log.json
  (if available), otherwise uses a fixed 3.0 s assumption.
  Camera-active time proxy = cv_activation_count × mean_cv_duration_s.

Metrics
-------
Security:
  • FAR, FRR per configuration
  • False challenge rate (CV triggered on genuine windows)
  • Authentication latency (KSD scoring time + CV time when triggered)

Efficiency:
  • CV activation count (System A always = n_windows; System B depends on anomalies)
  • % of total session time camera is active (proxy for privacy/overhead)
  • Mean CPU% during KSD scoring loop
  • Peak RAM delta (MB)

Outputs (experiments/results/)
-------------------------------
  exp3_summary.json
  exp3_comparison_table.png
  exp3_cv_activation.png
  exp3_latency_comparison.png

Usage
-----
    python experiments/exp3_always_on_vs_event_driven.py [--windows 200] [--seed 42]
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
    compute_metrics,
    ensure_results_dir,
    estimate_cv_timing_from_log,
    generate_genuine,
    generate_impostor,
    load_enrol_vectors,
    load_session_log,
    load_v2_scorer,
    parse_session_log_cv,
    save_json,
    score_batch,
    RESULTS_DIR,
)

warnings.filterwarnings("ignore")


# ═════════════════════════════════════════════════════════════════════════════
# Resource measurement helpers
# ═════════════════════════════════════════════════════════════════════════════

try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False


def _measure_ksd_loop(scorer, feature_vecs: list) -> dict:
    """
    Run feature_vecs through scorer.score() while sampling CPU/RAM.

    Returns
    -------
    dict: scores, thresholds, elapsed_s, cpu_samples, ram_delta_mb
    """
    if _PSUTIL:
        proc = psutil.Process()
        ram_before = proc.memory_info().rss / 1024 / 1024
        # Warm up CPU sampler
        proc.cpu_percent(interval=None)

    t0 = time.perf_counter()
    scores, thresholds = [], []
    cpu_samples = []

    for i, vec in enumerate(feature_vecs):
        scores.append(scorer.score(vec))
        thresholds.append(scorer.threshold())
        if _PSUTIL and i % 10 == 0:
            cpu_samples.append(proc.cpu_percent(interval=None))

    elapsed = time.perf_counter() - t0

    if _PSUTIL:
        ram_after  = proc.memory_info().rss / 1024 / 1024
        ram_delta  = ram_after - ram_before
    else:
        ram_delta  = None

    return {
        "scores":        np.array(scores),
        "thresholds":    np.array(thresholds),
        "elapsed_s":     elapsed,
        "cpu_samples":   cpu_samples if cpu_samples else None,
        "mean_cpu":      float(np.mean(cpu_samples)) if cpu_samples else None,
        "ram_delta_mb":  round(ram_delta, 2) if ram_delta is not None else None,
    }


# ═════════════════════════════════════════════════════════════════════════════
# System simulation
# ═════════════════════════════════════════════════════════════════════════════

def _run_system(
    name: str,
    always_on: bool,
    genuine_vecs: list,
    impostor_vecs: list,
    operational_threshold: float,
    mock_cv_duration_s: float,
    verbose: bool = True,
) -> dict:
    """
    Simulate a complete session: warm_up genuine windows, then test windows.

    always_on=True  → CV fires on EVERY window
    always_on=False → CV fires only when is_anomaly() returns True
    """
    _p = lambda *a: print(*a) if verbose else None

    # Mock CV: genuine → True, impostor → False
    def mock_cv_genuine():  return True
    def mock_cv_impostor(): return False

    all_vecs  = genuine_vecs + impostor_vecs
    all_labels = [0] * len(genuine_vecs) + [1] * len(impostor_vecs)

    scorer = load_v2_scorer()
    loop_result = _measure_ksd_loop(scorer, all_vecs)
    scores     = loop_result["scores"]
    thresholds = loop_result["thresholds"]

    # Re-score with fresh scorer to get is_anomaly() decisions
    scorer2 = load_v2_scorer()
    is_anom = []
    for vec in all_vecs:
        s = scorer2.score(vec)
        is_anom.append(scorer2.is_anomaly(s))

    # CV activation logic
    cv_activations_g = 0   # CV triggered on genuine windows
    cv_activations_i = 0   # CV triggered on impostor windows
    cv_correct       = 0   # CV returned correct decision
    cv_incorrect     = 0
    final_decisions  = []  # 0 = authenticated, 1 = rejected

    for idx, (vec, label, anom) in enumerate(zip(all_vecs, all_labels, is_anom)):
        trigger_cv = always_on or anom

        if trigger_cv:
            if label == 0:
                cv_activations_g += 1
                cv_result = mock_cv_genuine()
                cv_correct   += 1
                final_decisions.append(0)   # genuine confirmed → allow
            else:
                cv_activations_i += 1
                cv_result = mock_cv_impostor()
                cv_correct   += 1
                final_decisions.append(1)   # impostor detected → reject
        else:
            # No CV: rely on KSD score alone
            final_decisions.append(int(anom))

    y_pred = np.array(final_decisions, dtype=int)
    y_true = np.array(all_labels, dtype=int)

    TP = int(np.sum((y_pred == 1) & (y_true == 1)))
    TN = int(np.sum((y_pred == 0) & (y_true == 0)))
    FP = int(np.sum((y_pred == 1) & (y_true == 0)))
    FN = int(np.sum((y_pred == 0) & (y_true == 1)))

    n = len(all_vecs)
    n_genuine  = len(genuine_vecs)
    n_impostor = len(impostor_vecs)
    total_cv   = cv_activations_g + cv_activations_i

    # Camera-active time proxy
    camera_active_s     = total_cv * mock_cv_duration_s
    session_ksd_time_s  = loop_result["elapsed_s"]
    # Estimated total session time = time to type all windows
    # At WINDOW=25 keys/window and ~5 keys/s = ~5 s/window
    estimated_session_s = n * (V2_CONFIG["window_size_keys"] / 5.0) + total_cv * mock_cv_duration_s
    camera_pct          = camera_active_s / estimated_session_s * 100 if estimated_session_s > 0 else 0.0

    # Latency: KSD time per window + CV time if triggered
    ksd_ms_per_window   = loop_result["elapsed_s"] / n * 1000 if n > 0 else 0.0
    cv_latency_s        = total_cv * mock_cv_duration_s
    auth_latency_ms     = ksd_ms_per_window + (mock_cv_duration_s * 1000 if always_on else
                          (cv_latency_s / n * 1000 if n > 0 else 0.0))

    FAR = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    FRR = FN / (FN + TP) if (FN + TP) > 0 else 0.0
    false_challenge_rate = cv_activations_g / n_genuine if n_genuine > 0 else 0.0

    _p(f"\n  [{name}]")
    _p(f"    CV activations (genuine windows)  : {cv_activations_g}/{n_genuine}")
    _p(f"    CV activations (impostor windows) : {cv_activations_i}/{n_impostor}")
    _p(f"    Total CV activations              : {total_cv}/{n}")
    _p(f"    False challenge rate              : {false_challenge_rate*100:.1f}%")
    _p(f"    FAR                               : {FAR:.4f}")
    _p(f"    FRR                               : {FRR:.4f}")
    _p(f"    Camera-active time proxy          : {camera_active_s:.1f} s ({camera_pct:.1f}% of session)")
    _p(f"    KSD scoring time (ms/window)      : {ksd_ms_per_window:.3f}")
    _p(f"    Mean CPU% (KSD loop)              : {loop_result['mean_cpu']}")

    return {
        "name":                  name,
        "mode":                  "always_on" if always_on else "event_driven",
        "n_genuine":             n_genuine,
        "n_impostor":            n_impostor,
        "n_windows":             n,
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "FAR":                   round(FAR, 4),
        "FRR":                   round(FRR, 4),
        "false_challenge_rate":  round(false_challenge_rate, 4),
        "cv_activations_genuine": cv_activations_g,
        "cv_activations_impostor": cv_activations_i,
        "cv_activations_total":  total_cv,
        "cv_activation_pct":     round(total_cv / n * 100, 2) if n > 0 else 0.0,
        "camera_active_s":       round(camera_active_s, 2),
        "camera_active_pct":     round(camera_pct, 2),
        "ksd_ms_per_window":     round(ksd_ms_per_window, 4),
        "auth_latency_ms_est":   round(auth_latency_ms, 2),
        "mean_cpu_pct":          loop_result["mean_cpu"],
        "ram_delta_mb":          loop_result["ram_delta_mb"],
        "psutil_available":      _PSUTIL,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Main runner
# ═════════════════════════════════════════════════════════════════════════════

def run(n_windows: int = 200, seed: int = 42, verbose: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    _p = lambda *a: print(*a) if verbose else None

    _p("\n" + "=" * 65)
    _p("  EXPERIMENT 3 — Always-On CV vs Event-Driven CV")
    _p("=" * 65)

    enrol_vecs = load_enrol_vectors()
    _p(f"\n  Enrolment windows : {len(enrol_vecs)}")

    # Estimate mean CV duration from real session log
    log_data = load_session_log()
    cv_info  = parse_session_log_cv(log_data)
    timing   = estimate_cv_timing_from_log(cv_info)
    if timing["mean_cv_duration_s"] is not None:
        mock_cv_duration_s = timing["mean_cv_duration_s"]
        _p(f"  CV duration estimate: {mock_cv_duration_s:.2f} s "
           f"(from {timing['n_intervals']} real intervals in session_log.json)")
    else:
        mock_cv_duration_s = 3.0
        _p(f"  CV duration estimate: {mock_cv_duration_s:.1f} s "
           "(default — no timing data in session_log.json)")

    _p(f"  Real CV record stats from session_log.json:")
    _p(f"    n_total={cv_info['n_total']}  match={cv_info['n_match']}  "
       f"fail={cv_info['n_fail']}  error={cv_info['n_error']}")

    # Operational threshold
    scorer_base = load_v2_scorer()
    bl_scores, _ = score_batch(scorer_base,
                               list(generate_genuine(50, rng, enrol_vecs)))
    op_thr = max(
        float(np.median(bl_scores) + V2_CONFIG["K"] * np.std(bl_scores)),
        V2_CONFIG["FLOOR"]
    )
    _p(f"  Operational threshold : {op_thr:.4f}")

    # Generate test set (identical for both systems)
    genuine_vecs  = generate_genuine(n_windows, rng, enrol_vecs)
    impostor_vecs = generate_impostor(n_windows, rng, enrol_vecs)

    _p(f"\n  Test set: {n_windows} genuine + {n_windows} impostor windows")
    _p("  (same vectors fed to both systems)")

    # ── System A — Always-On ──────────────────────────────────────────────────
    _p("\n  Running System A (Always-On) …")
    sys_a = _run_system(
        "System A — Always-On",
        always_on=True,
        genuine_vecs=genuine_vecs,
        impostor_vecs=impostor_vecs,
        operational_threshold=op_thr,
        mock_cv_duration_s=mock_cv_duration_s,
        verbose=verbose,
    )

    # ── System B — Event-Driven ───────────────────────────────────────────────
    _p("\n  Running System B (Event-Driven) …")
    sys_b = _run_system(
        "System B — Event-Driven",
        always_on=False,
        genuine_vecs=genuine_vecs,
        impostor_vecs=impostor_vecs,
        operational_threshold=op_thr,
        mock_cv_duration_s=mock_cv_duration_s,
        verbose=verbose,
    )

    # ── Delta analysis ────────────────────────────────────────────────────────
    cv_reduction_pct = (
        (sys_a["cv_activations_total"] - sys_b["cv_activations_total"])
        / sys_a["cv_activations_total"] * 100
        if sys_a["cv_activations_total"] > 0 else 0.0
    )
    camera_reduction_s = sys_a["camera_active_s"] - sys_b["camera_active_s"]

    _p("\n  Comparison")
    _p("  ──────────")
    _p(f"  CV activation reduction (A→B)   : {cv_reduction_pct:.1f}%")
    _p(f"  Camera time reduction (A→B)     : {camera_reduction_s:.1f} s")
    _p(f"  FAR change (A→B)                : {sys_a['FAR']:.4f} → {sys_b['FAR']:.4f}")
    _p(f"  FRR change (A→B)                : {sys_a['FRR']:.4f} → {sys_b['FRR']:.4f}")

    result = {
        "experiment":              "3 — Always-On CV vs Event-Driven CV",
        "config": {
            **V2_CONFIG,
            "seed":              seed,
            "n_genuine":         n_windows,
            "n_impostor":        n_windows,
            "mock_cv_duration_s": round(mock_cv_duration_s, 3),
            "cv_duration_source": "session_log.json" if timing["mean_cv_duration_s"] else "default_3s",
        },
        "real_cv_log_stats":       cv_info,
        "operational_threshold":   round(op_thr, 4),
        "system_A":                sys_a,
        "system_B":                sys_b,
        "delta": {
            "cv_reduction_pct":     round(cv_reduction_pct, 2),
            "camera_reduction_s":   round(camera_reduction_s, 2),
            "far_delta":            round(sys_b["FAR"] - sys_a["FAR"], 4),
            "frr_delta":            round(sys_b["FRR"] - sys_a["FRR"], 4),
            "latency_delta_ms":     round(sys_b["auth_latency_ms_est"] - sys_a["auth_latency_ms_est"], 2),
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
    sA = result["system_A"]
    sB = result["system_B"]

    # ── CV activation comparison ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    labels = ["System A\n(Always-On)", "System B\n(Event-Driven)"]

    ax = axes[0]
    cv_counts = [sA["cv_activations_total"], sB["cv_activations_total"]]
    bars = ax.bar(labels, cv_counts, color=["#F44336", "#4CAF50"], alpha=0.8, width=0.5)
    ax.set_ylabel("Total CV activations", fontsize=11)
    ax.set_title("CV Activations\nper Test Session", fontsize=11, fontweight="bold")
    for bar, val in zip(bars, cv_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(val), ha="center", fontsize=12, fontweight="bold")

    ax = axes[1]
    cam_times = [sA["camera_active_s"], sB["camera_active_s"]]
    bars = ax.bar(labels, cam_times, color=["#FF9800", "#2196F3"], alpha=0.8, width=0.5)
    ax.set_ylabel("Camera-active time (s)", fontsize=11)
    ax.set_title("Camera-Active Time\nProxy", fontsize=11, fontweight="bold")
    for bar, val in zip(bars, cam_times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.0f}s", ha="center", fontsize=12, fontweight="bold")

    ax = axes[2]
    metrics = [["FAR", sA["FAR"], sB["FAR"]],
               ["FRR", sA["FRR"], sB["FRR"]],
               ["False Challenge", sA["false_challenge_rate"], sB["false_challenge_rate"]]]
    m_labels = [m[0] for m in metrics]
    a_vals   = [m[1] for m in metrics]
    b_vals   = [m[2] for m in metrics]
    x = np.arange(len(m_labels))
    w = 0.35
    ax.bar(x - w/2, a_vals, w, label="System A", color="#F44336", alpha=0.8)
    ax.bar(x + w/2, b_vals, w, label="System B", color="#4CAF50", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(m_labels, fontsize=9)
    ax.set_ylabel("Rate", fontsize=11)
    ax.set_title("Security Metrics\nComparison", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(max(a_vals), max(b_vals)) * 1.3 + 0.01)

    fig.suptitle("Experiment 3 — Always-On vs Event-Driven CV", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "exp3_cv_activation.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {path}")

    # ── Comparison table as image ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.axis("off")
    rows = [
        ["Metric", "System A (Always-On)", "System B (Event-Driven)", "Delta"],
        ["CV activations",
         f"{sA['cv_activations_total']}",
         f"{sB['cv_activations_total']}",
         f"−{result['delta']['cv_reduction_pct']:.1f}%"],
        ["Camera-active time",
         f"{sA['camera_active_s']:.1f} s ({sA['camera_active_pct']:.1f}%)",
         f"{sB['camera_active_s']:.1f} s ({sB['camera_active_pct']:.1f}%)",
         f"−{result['delta']['camera_reduction_s']:.1f} s"],
        ["FAR",
         f"{sA['FAR']:.4f}",
         f"{sB['FAR']:.4f}",
         f"{result['delta']['far_delta']:+.4f}"],
        ["FRR",
         f"{sA['FRR']:.4f}",
         f"{sB['FRR']:.4f}",
         f"{result['delta']['frr_delta']:+.4f}"],
        ["False challenge rate",
         f"{sA['false_challenge_rate']:.4f}",
         f"{sB['false_challenge_rate']:.4f}",
         f"{sB['false_challenge_rate'] - sA['false_challenge_rate']:+.4f}"],
        ["Auth latency est.",
         f"{sA['auth_latency_ms_est']:.1f} ms",
         f"{sB['auth_latency_ms_est']:.1f} ms",
         f"{result['delta']['latency_delta_ms']:+.1f} ms"],
        ["Mean CPU% (KSD loop)",
         f"{sA['mean_cpu_pct']}" if sA["mean_cpu_pct"] else "N/A (no psutil)",
         f"{sB['mean_cpu_pct']}" if sB["mean_cpu_pct"] else "N/A (no psutil)",
         ""],
    ]
    tbl = ax.table(cellText=rows[1:], colLabels=rows[0],
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)
    # Header styling
    for j in range(4):
        tbl[0, j].set_facecolor("#1976D2")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    ax.set_title("Experiment 3 — System Comparison Table", fontsize=12, fontweight="bold", pad=10)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "exp3_comparison_table.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Exp 3 — Always-On vs Event-Driven CV")
    parser.add_argument("--windows", type=int, default=200)
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    result = run(n_windows=args.windows, seed=args.seed)
    plot_results(result)
    # Remove large nested dicts not needed in JSON
    result_save = {k: v for k, v in result.items()
                   if k not in ("real_cv_log_stats",)}
    # Remove results key that has redundant raw list
    result_save["real_cv_log_stats_summary"] = {
        "n_total": result["real_cv_log_stats"]["n_total"],
        "n_match": result["real_cv_log_stats"]["n_match"],
        "n_fail":  result["real_cv_log_stats"]["n_fail"],
        "n_error": result["real_cv_log_stats"]["n_error"],
    }
    path = save_json(result_save, "exp3_summary.json")
    print(f"\n  Results saved → {path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
