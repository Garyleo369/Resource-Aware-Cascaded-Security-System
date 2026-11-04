"""
experiments/exp4_ksd_vs_cascaded.py
=====================================
Experiment 4 — KSD vs KSD+Face vs KSD+Face+Gaze
-------------------------------------------------

Goal
----
Determine whether adding the existing CV/face/gaze verification layer improves
authentication compared with KSD alone.

System configurations
---------------------
  System A — KSD only
    AnomalyScorer.is_anomaly() → final decision.

  System B — KSD → Face verification
    KSD anomaly → mock Face CV (genuine→True, impostor→False).
    This matches the current cascaded_auth/monitor.py behaviour.

  System C — KSD → Face + Gaze verification
    KSD anomaly → mock Face AND mock Gaze check.
    Gaze is evaluated using the real CV result data from session_log.json:
    - The real gaze-distance pass rate from session_log.json is used to
      determine the gaze reliability proxy.
    - In the mock, gaze confirms genuine if face confirmed it (same mock
      result), but with an independent 5% random failure rate added to
      simulate gaze errors observed in the real log.
    System C is included because gaze IS functional (cascaded_auth/cv_check.py
    uses MediaPipe FaceMesh iris tracking + Mahalanobis distance).

Metrics
-------
  • FAR, FRR, EER per system
  • Auth success rate (legitimate user authenticated)
  • Impostor rejection rate
  • False challenge rate (CV triggered on genuine)
  • Average authentication time (KSD + CV overhead)
  • Number of CV/gaze checks required

Decision-flow breakdown
-----------------------
  • KSD detects anomaly → CV confirms impostor (TP caught by CV)
  • KSD false alarm → CV rescues genuine user (FP rescued by CV)
  • KSD misses anomaly → not catchable in current architecture (documented)

Outputs (experiments/results/)
-------------------------------
  exp4_summary.json
  exp4_confusion_matrices.png
  exp4_far_frr_chart.png
  exp4_latency_comparison.png
  exp4_decision_flow.png

Usage
-----
    python experiments/exp4_ksd_vs_cascaded.py [--windows 200] [--seed 42]
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

# Estimated gaze error rate based on real CV log observations
# (the log shows ~15% of CV checks fail due to no-face or distance issues
#  even for the genuine user — this is the gaze reliability proxy)
_GAZE_ERROR_RATE = 0.15   # 15% chance gaze disagrees with face even for genuine


# ═════════════════════════════════════════════════════════════════════════════
# System simulation
# ═════════════════════════════════════════════════════════════════════════════

def _run_system(
    name: str,
    use_face_cv: bool,
    use_gaze_cv: bool,
    genuine_vecs: list,
    impostor_vecs: list,
    scorer,
    operational_threshold: float,
    mock_cv_duration_s: float,
    gaze_error_rate: float,
    rng: np.random.Generator,
    verbose: bool = True,
) -> dict:
    """
    Simulate authentication for all test windows and collect decision stats.
    """
    _p = lambda *a: print(*a) if verbose else None

    all_vecs   = genuine_vecs + impostor_vecs
    all_labels = np.array([0]*len(genuine_vecs) + [1]*len(impostor_vecs), dtype=int)
    n          = len(all_vecs)
    n_genuine  = len(genuine_vecs)
    n_impostor = len(impostor_vecs)

    # KSD raw scores for all windows
    scores_arr, _ = score_batch(scorer, all_vecs)
    ksd_anomaly   = scores_arr >= operational_threshold   # bool array

    final_decisions   = np.zeros(n, dtype=int)
    cv_checks_face    = 0
    cv_checks_gaze    = 0
    gaze_errors       = 0

    # Decision-flow counters
    ksd_anom_cv_confirms_impostor = 0   # KSD+CV TP
    ksd_fa_cv_rescues_genuine     = 0   # KSD FP rescued by CV
    ksd_misses_impostor           = 0   # KSD FN (impostor slips through KSD)

    for idx in range(n):
        label      = int(all_labels[idx])
        ksd_flagged = bool(ksd_anomaly[idx])

        if not use_face_cv:
            # System A: KSD decision only
            final_decisions[idx] = int(ksd_flagged)
            if ksd_flagged and label == 0:
                pass  # FP
            if not ksd_flagged and label == 1:
                ksd_misses_impostor += 1

        else:
            # System B / C: KSD → Face (→ Gaze)
            if ksd_flagged:
                # Trigger face CV
                cv_checks_face += 1
                face_match = (label == 0)   # mock: genuine always matches, impostor never

                if face_match:
                    # CV rescued a KSD false alarm
                    if label == 0:
                        ksd_fa_cv_rescues_genuine += 1
                    final_decisions[idx] = 0   # authenticated

                    if use_gaze_cv:
                        cv_checks_gaze += 1
                        # Gaze: match if face match, but with error rate
                        gaze_fail = rng.random() < gaze_error_rate
                        if gaze_fail:
                            gaze_errors += 1
                            # Gaze disagrees → conservative: reject
                            final_decisions[idx] = 1
                        else:
                            final_decisions[idx] = 0
                else:
                    # Face says impostor → reject
                    if label == 1:
                        ksd_anom_cv_confirms_impostor += 1
                    final_decisions[idx] = 1

                    if use_gaze_cv:
                        # Gaze also run (compound check)
                        cv_checks_gaze += 1
                        # Impostor: gaze also fails (same mock), no error
                        # decision already = reject

            else:
                # KSD clean window
                final_decisions[idx] = 0
                if label == 1:
                    ksd_misses_impostor += 1

    y_pred = final_decisions
    y_true = all_labels

    TP = int(np.sum((y_pred == 1) & (y_true == 1)))
    TN = int(np.sum((y_pred == 0) & (y_true == 0)))
    FP = int(np.sum((y_pred == 1) & (y_true == 0)))
    FN = int(np.sum((y_pred == 0) & (y_true == 1)))

    FAR = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    FRR = FN / (FN + TP) if (FN + TP) > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    fpr_arr, tpr_arr, thr_arr = compute_roc(y_true, scores_arr)
    eer, eer_thr = compute_eer(fpr_arr, tpr_arr, thr_arr)
    auc = compute_auc(fpr_arr, tpr_arr)

    # False challenge rate = (genuine windows triggering CV) / n_genuine
    genuine_ksd_flags = int(np.sum(ksd_anomaly[:n_genuine]))
    false_challenge_rate = genuine_ksd_flags / n_genuine if n_genuine > 0 else 0.0

    # Auth time estimate
    ksd_only_s    = V2_CONFIG["window_size_keys"] / 5.0   # typing time per window
    cv_overhead_s = 0.0
    if use_face_cv:
        cv_overhead_s += mock_cv_duration_s * cv_checks_face / n if n > 0 else 0.0
    if use_gaze_cv:
        cv_overhead_s += mock_cv_duration_s * cv_checks_gaze / n if n > 0 else 0.0
    auth_latency_ms = (ksd_only_s + cv_overhead_s) * 1000

    legit_auth_rate = TN / n_genuine if n_genuine > 0 else 0.0
    impostor_rej    = TP / n_impostor if n_impostor > 0 else 0.0

    _p(f"\n  [{name}]")
    _p(f"    TP={TP} TN={TN} FP={FP} FN={FN}")
    _p(f"    FAR           : {FAR:.4f}")
    _p(f"    FRR           : {FRR:.4f}")
    _p(f"    EER           : {eer:.4f}  (thr={eer_thr:.4f})")
    _p(f"    Legit auth    : {legit_auth_rate*100:.1f}%")
    _p(f"    Impostor rej  : {impostor_rej*100:.1f}%")
    _p(f"    False challenge rate: {false_challenge_rate*100:.1f}%")
    if use_face_cv:
        _p(f"    Face CV checks: {cv_checks_face}")
    if use_gaze_cv:
        _p(f"    Gaze CV checks: {cv_checks_gaze}  (gaze errors: {gaze_errors})")
    _p(f"    Auth latency est : {auth_latency_ms:.0f} ms")

    return {
        "name": name,
        "uses_face_cv": use_face_cv,
        "uses_gaze_cv": use_gaze_cv,
        "n_genuine": n_genuine, "n_impostor": n_impostor,
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "FAR": round(FAR, 4),
        "FRR": round(FRR, 4),
        "EER": round(eer, 4),
        "EER_threshold": round(eer_thr, 4),
        "ROC_AUC": round(auc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "legit_auth_rate": round(legit_auth_rate, 4),
        "impostor_rejection_rate": round(impostor_rej, 4),
        "false_challenge_rate": round(false_challenge_rate, 4),
        "cv_checks_face": cv_checks_face,
        "cv_checks_gaze": cv_checks_gaze,
        "gaze_errors": gaze_errors,
        "auth_latency_ms_est": round(auth_latency_ms, 0),
        "decision_flow": {
            "ksd_anomaly_cv_confirms_impostor": ksd_anom_cv_confirms_impostor,
            "ksd_false_alarm_cv_rescues_genuine": ksd_fa_cv_rescues_genuine,
            "ksd_misses_impostor": ksd_misses_impostor,
        },
        "_scores": scores_arr.tolist(),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Main runner
# ═════════════════════════════════════════════════════════════════════════════

def run(n_windows: int = 200, seed: int = 42, verbose: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    _p = lambda *a: print(*a) if verbose else None

    _p("\n" + "=" * 65)
    _p("  EXPERIMENT 4 — KSD vs KSD+Face vs KSD+Face+Gaze")
    _p("=" * 65)

    enrol_vecs = load_enrol_vectors()
    _p(f"\n  Enrolment windows : {len(enrol_vecs)}")

    # CV timing estimate
    log_data = load_session_log()
    cv_info  = parse_session_log_cv(log_data)
    timing   = estimate_cv_timing_from_log(cv_info)
    mock_cv_duration_s = timing["mean_cv_duration_s"] if timing["mean_cv_duration_s"] else 3.0

    # Gaze reliability from real session_log.json
    real_fail_rate = (cv_info["n_fail"] + cv_info["n_error"]) / cv_info["n_total"] \
        if cv_info["n_total"] > 0 else _GAZE_ERROR_RATE
    gaze_error_rate = min(real_fail_rate, 0.40)   # cap at 40% to avoid degenerate model
    _p(f"  Real CV fail+error rate (proxy for gaze unreliability): {real_fail_rate*100:.1f}%")
    _p(f"  Gaze error rate used for System C: {gaze_error_rate*100:.1f}%")
    _p(f"  Mock CV duration: {mock_cv_duration_s:.2f} s")

    # Operational threshold
    scorer_bl = load_v2_scorer()
    bl_scores, _ = score_batch(scorer_bl, list(generate_genuine(50, rng, enrol_vecs)))
    op_thr = max(
        float(np.median(bl_scores) + V2_CONFIG["K"] * np.std(bl_scores)),
        V2_CONFIG["FLOOR"]
    )
    _p(f"  Operational threshold : {op_thr:.4f}")

    # Test data — identical across all systems
    genuine_vecs  = generate_genuine(n_windows, rng, enrol_vecs)
    impostor_vecs = generate_impostor(n_windows, rng, enrol_vecs)
    _p(f"\n  Test: {n_windows} genuine + {n_windows} impostor (same for all systems)")

    # Fresh scorers per system to avoid state bleed
    def fresh():
        return load_v2_scorer()

    sys_a = _run_system(
        "System A — KSD only",
        use_face_cv=False, use_gaze_cv=False,
        genuine_vecs=genuine_vecs, impostor_vecs=impostor_vecs,
        scorer=fresh(), operational_threshold=op_thr,
        mock_cv_duration_s=mock_cv_duration_s,
        gaze_error_rate=gaze_error_rate, rng=rng, verbose=verbose,
    )

    sys_b = _run_system(
        "System B — KSD + Face",
        use_face_cv=True, use_gaze_cv=False,
        genuine_vecs=genuine_vecs, impostor_vecs=impostor_vecs,
        scorer=fresh(), operational_threshold=op_thr,
        mock_cv_duration_s=mock_cv_duration_s,
        gaze_error_rate=gaze_error_rate, rng=rng, verbose=verbose,
    )

    sys_c = _run_system(
        "System C — KSD + Face + Gaze",
        use_face_cv=True, use_gaze_cv=True,
        genuine_vecs=genuine_vecs, impostor_vecs=impostor_vecs,
        scorer=fresh(), operational_threshold=op_thr,
        mock_cv_duration_s=mock_cv_duration_s,
        gaze_error_rate=gaze_error_rate, rng=rng, verbose=verbose,
    )

    # Summary comparison
    _p("\n  Summary Table")
    _p("  ─────────────")
    _p(f"  {'System':<25} {'FAR':>7} {'FRR':>7} {'EER':>7} {'Auth%':>8} {'ImpRej%':>9} {'Latency':>10}")
    for sys in [sys_a, sys_b, sys_c]:
        _p(f"  {sys['name']:<25} "
           f"{sys['FAR']:>7.4f} {sys['FRR']:>7.4f} {sys['EER']:>7.4f} "
           f"{sys['legit_auth_rate']*100:>7.1f}% "
           f"{sys['impostor_rejection_rate']*100:>8.1f}% "
           f"{sys['auth_latency_ms_est']:>8.0f} ms")

    result = {
        "experiment": "4 — KSD vs KSD+Face vs KSD+Face+Gaze",
        "config": {
            **V2_CONFIG,
            "seed": seed,
            "n_genuine": n_windows,
            "n_impostor": n_windows,
            "n_enrol": len(enrol_vecs),
            "mock_cv_duration_s": round(mock_cv_duration_s, 3),
            "gaze_error_rate_used": round(gaze_error_rate, 4),
            "gaze_error_source": "real session_log.json CV fail+error rate",
        },
        "operational_threshold": round(op_thr, 4),
        "system_A": {k: v for k, v in sys_a.items() if k != "_scores"},
        "system_B": {k: v for k, v in sys_b.items() if k != "_scores"},
        "system_C": {k: v for k, v in sys_c.items() if k != "_scores"},
        "_plot_data": {
            "a_scores": sys_a["_scores"],
            "b_scores": sys_b["_scores"],
            "c_scores": sys_c["_scores"],
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
    sA, sB, sC = result["system_A"], result["system_B"], result["system_C"]
    systems    = [sA, sB, sC]
    s_labels   = ["A: KSD only", "B: KSD+Face", "C: KSD+Face+Gaze"]
    colors     = ["#2196F3", "#4CAF50", "#9C27B0"]

    # ── 1. FAR / FRR / EER chart ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(3)
    w = 0.25
    far_vals = [s["FAR"] for s in systems]
    frr_vals = [s["FRR"] for s in systems]
    eer_vals = [s["EER"] for s in systems]
    ax.bar(x - w, far_vals, w, label="FAR", color="#F44336", alpha=0.85)
    ax.bar(x,     frr_vals, w, label="FRR", color="#2196F3", alpha=0.85)
    ax.bar(x + w, eer_vals, w, label="EER", color="#9C27B0", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(s_labels, fontsize=10)
    ax.set_ylabel("Rate", fontsize=11)
    ax.set_title("Exp 4 — FAR / FRR / EER by System Configuration", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, min(1.1, max(far_vals + frr_vals + eer_vals) * 1.4 + 0.05))
    for i, (f, r, e) in enumerate(zip(far_vals, frr_vals, eer_vals)):
        ax.text(i - w, f + 0.005, f"{f:.3f}", ha="center", fontsize=8)
        ax.text(i,     r + 0.005, f"{r:.3f}", ha="center", fontsize=8)
        ax.text(i + w, e + 0.005, f"{e:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "exp4_far_frr_chart.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {path}")

    # ── 2. Confusion matrices (3 side by side) ────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, s, lbl, col in zip(axes, systems, s_labels, colors):
        cm = np.array([[s["TN"], s["FP"]], [s["FN"], s["TP"]]])
        ax.imshow(cm, cmap="Blues", vmin=0)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Normal", "Anomaly"], fontsize=9)
        ax.set_yticklabels(["Genuine", "Impostor"], fontsize=9)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=14, fontweight="bold",
                        color="white" if cm[i, j] > cm.max() * 0.6 else "black")
        ax.set_title(f"{lbl}\nF1={2*s['precision']*s['recall']/(s['precision']+s['recall']+1e-9):.3f}",
                     fontsize=10, fontweight="bold")
    fig.suptitle("Exp 4 — Confusion Matrices", fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "exp4_confusion_matrices.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {path}")

    # ── 3. Auth latency comparison ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    lat_vals = [s["auth_latency_ms_est"] for s in systems]
    bars = ax.bar(s_labels, lat_vals, color=colors, alpha=0.85, width=0.45)
    ax.set_ylabel("Estimated auth latency (ms)", fontsize=11)
    ax.set_title("Exp 4 — Authentication Latency by System", fontsize=12, fontweight="bold")
    for bar, v in zip(bars, lat_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f"{v:.0f} ms", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(lat_vals) * 1.25 + 50)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "exp4_latency_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {path}")

    # ── 4. Decision-flow breakdown ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Flow bar chart for System B and C (KSD-only has no CV)
    ax = axes[0]
    flow_labels = ["KSD anom→\nCV confirms imp.", "KSD FA→\nCV rescues genuine", "KSD misses\nImpostor"]
    b_flow = [sB["decision_flow"]["ksd_anomaly_cv_confirms_impostor"],
              sB["decision_flow"]["ksd_false_alarm_cv_rescues_genuine"],
              sB["decision_flow"]["ksd_misses_impostor"]]
    c_flow = [sC["decision_flow"]["ksd_anomaly_cv_confirms_impostor"],
              sC["decision_flow"]["ksd_false_alarm_cv_rescues_genuine"],
              sC["decision_flow"]["ksd_misses_impostor"]]
    x = np.arange(3)
    w = 0.35
    ax.bar(x - w/2, b_flow, w, label="System B (KSD+Face)", color="#4CAF50", alpha=0.85)
    ax.bar(x + w/2, c_flow, w, label="System C (KSD+F+G)",  color="#9C27B0", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(flow_labels, fontsize=9)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Decision-Flow Statistics\n(Systems B & C)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)

    # CV checks required
    ax2 = axes[1]
    cv_face = [0, sB["cv_checks_face"], sC["cv_checks_face"]]
    cv_gaze = [0, 0, sC["cv_checks_gaze"]]
    x2 = np.arange(3)
    ax2.bar(x2,     cv_face, 0.4, label="Face checks", color="#2196F3", alpha=0.85)
    ax2.bar(x2, cv_gaze, 0.4, bottom=cv_face, label="Gaze checks", color="#FF9800", alpha=0.85)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(s_labels, fontsize=9)
    ax2.set_ylabel("Number of CV checks", fontsize=11)
    ax2.set_title("CV Checks Required\nper System", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)

    fig.suptitle("Exp 4 — Decision Flow & CV Overhead", fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "exp4_decision_flow.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {path}")


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Exp 4 — KSD vs KSD+Face vs KSD+Face+Gaze")
    parser.add_argument("--windows", type=int, default=200)
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    result = run(n_windows=args.windows, seed=args.seed)
    plot_results(result)
    result_save = {k: v for k, v in result.items() if k != "_plot_data"}
    path = save_json(result_save, "exp4_summary.json")
    print(f"\n  Results saved → {path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
