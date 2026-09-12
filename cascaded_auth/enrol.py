"""
cascaded_auth/enrol.py — One-time enrolment for Prototype v2.

Run once (ideally for 15 minutes) to build both the KSD baseline model and the
gaze profile.  Produces three artefact files inside the cascaded_auth/ folder:

    cascaded_auth/model.joblib          — IsolationForest on (12,) feature vecs
    cascaded_auth/enrol_vectors.joblib  — raw feature matrix for recalibration
    cascaded_auth/gaze_profile.joblib   — gaze mean + covariance for CV check

Usage
-----
    cd <project_root>
    python cascaded_auth/enrol.py

Tip: type across different contexts for best results — fast chat, slow code,
prose, emails, terminal commands.  More variety → better baseline.
"""

import sys
import time
import numpy as np

# ── Resolve imports whether run as script or module ───────────────────────────
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cascaded_auth.ksd      import KeystrokeBuffer, AnomalyScorer, extract_features, WINDOW
from cascaded_auth.cv_check import collect_gaze_vectors, save_gaze_profile

try:
    from pynput import keyboard
except ImportError:
    print("ERROR: pynput not installed.  Run:  pip install pynput")
    sys.exit(1)

# ── Constants ─────────────────────────────────────────────────────────────────
ENROL_SECS  = 900    # 15 minutes (Fix 1)
GAZE_SECS   = 30     # gaze capture after keystroke enrolment
# ── Point 1: 50 windows required for a reliable IsolationForest baseline.
# With fewer samples the model scores everything as anomalous (normalised
# score ≈ 1.0), causing the dynamic threshold to saturate so no intruder
# is ever detected.  50 windows = 2500 keystrokes ≈ 5–10 minutes of typing.
MIN_WINDOWS = 10


def enrol() -> None:
    print("=" * 60)
    print("  Prototype v2 — Keystroke Enrolment")
    print("=" * 60)
    print(f"Type normally for {ENROL_SECS // 60} minutes.")
    print("Mix fast typing, slow typing, code, and prose for best results.\n")

    buf             = KeystrokeBuffer()
    scorer          = AnomalyScorer()
    feature_vectors: list[np.ndarray] = []
    _window_count   = 0

    # ── pynput callbacks ──────────────────────────────────────────────────────
    def on_press(key):
        buf.on_press(key)

    def on_release(key):
        nonlocal _window_count
        buf.on_release(key)
        if not buf.ready():
            return

        events = buf.flush()
        vec    = extract_features(events)

        # Fix 1 — noise filter: skip windows with near-zero mean dwell
        # (vec[0] is mean dwell in seconds; threshold 0.05 s = 50 ms)
        if vec[0] > 0.05:
            feature_vectors.append(vec)
            _window_count += 1

            elapsed_min = (time.time() - _start) / 60
            remaining   = max(0, ENROL_SECS - (time.time() - _start))
            print(
                f"\r  Windows: {_window_count:>4} | "
                f"Elapsed: {elapsed_min:4.1f} min | "
                f"Remaining: {remaining / 60:4.1f} min   ",
                end="",
                flush=True,
            )

    # ── Keystroke listener ────────────────────────────────────────────────────
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    _start = time.time()

    try:
        time.sleep(ENROL_SECS)
    except KeyboardInterrupt:
        print("\n\nEnrolment interrupted by user.")

    listener.stop()
    print()   # newline after the \r progress line

    # ── Validate ──────────────────────────────────────────────────────────────
    if len(feature_vectors) < MIN_WINDOWS:
        print(
            f"\nNot enough data: only {len(feature_vectors)} windows collected "
            f"(need ≥ {MIN_WINDOWS}, ideally 100+).\n"
            "\nThe IsolationForest requires at least 30–50 windows to learn your\n"
            "normal typing distribution. With fewer samples it marks everything\n"
            "as anomalous, making intruder detection impossible.\n"
            "\nTip: type continuously for at least 5 minutes and try again."
        )
        return

    # ── Fit and save KSD model ────────────────────────────────────────────────
    X = np.array(feature_vectors, dtype=np.float64)
    print(f"\nFitting IsolationForest on {len(X)} windows (feature dim={X.shape[1]})...")
    scorer.fit(X, save_vectors=True)
    print(f"  ✓ KSD model saved → cascaded_auth/model.joblib")
    print(f"  ✓ Enrolment vectors saved → cascaded_auth/enrol_vectors.joblib")

    # ── Point 2: model health check ─────────────────────────────────────
    # Probe normalised scores on the enrolled data so the user can
    # immediately verify whether the IsolationForest is well-trained.
    raw_scores = scorer.model.score_samples(X)                 # shape (N,)
    offset = float(scorer.model.offset_)
    norm_scores = 1.0 / (1.0 + np.exp(10.0 * (raw_scores - offset)))
    mean_norm = float(norm_scores.mean())
    max_norm  = float(norm_scores.max())
    print()
    print("  MODEL HEALTH CHECK")
    print("  " + "-" * 50)
    print(f"  Normalised score on own enrolment data:")
    print(f"    mean = {mean_norm:.4f}   max = {max_norm:.4f}")
    print(f"  Ideal: mean ≤ 0.25, max ≤ 0.50")
    if mean_norm <= 0.25:
        print("  ✅ GOOD — model trained on sufficient data.")
    elif mean_norm <= 0.50:
        print("  ⚠️  ACCEPTABLE — consider re-enrolling with more windows.")
    else:
        print("  ❌ POOR — the model is still undertrained.")
        print("     Scores near 1.0 mean everything looks anomalous.")
        print(f"     Re-run and type for longer (current: {len(X)} windows, target: 100+).")
    print("  " + "-" * 50)

    # ── Gaze profile ──────────────────────────────────────────────────────────
    print(f"\nCollecting gaze profile ({GAZE_SECS} s — look at the screen normally)...")
    gaze_vecs = collect_gaze_vectors(GAZE_SECS)

    if not gaze_vecs:
        print("  ✗ No face detected — gaze profile NOT saved.")
        print("    Make sure your webcam is connected and your face is visible.")
    else:
        save_gaze_profile(gaze_vecs)
        print(f"  ✓ Gaze profile saved → cascaded_auth/gaze_profile.joblib")

    print("\nEnrolment complete.  Run cascaded_auth/monitor.py to start monitoring.")
    print("=" * 60)


if __name__ == "__main__":
    enrol()
