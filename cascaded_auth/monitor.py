"""
cascaded_auth/monitor.py — Live continuous authentication monitor for Prototype v2.

Run after enrol.py has completed.

Behaviour
---------
1. A pynput listener captures keystrokes globally.
2. Every WINDOW (25) keystrokes a 12-D feature vector is scored by the
   IsolationForest model.
3. A dynamic threshold (K=1.2 above rolling median, floor=0.45) decides
   whether a window is anomalous.
4. Consecutive anomaly buffer: CV fires after 1 anomalous window.
   A single clean window resets the streak.
5. CV runs in a dedicated daemon thread to avoid blocking the listener.
6. Fix 4 — Recalibration: if CV confirms the owner on an anomalous window,
   that feature vector is added to AnomalyScorer for eventual retraining.

Usage
-----
    cd <project_root>
    python cascaded_auth/monitor.py
    # Press Ctrl+C to stop.
"""

import sys
import os
import queue
import threading
import time
import ctypes

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cascaded_auth.ksd      import KeystrokeBuffer, AnomalyScorer, extract_features
from cascaded_auth.cv_check import run_cv_check

try:
    from pynput import keyboard
except ImportError:
    print("ERROR: pynput not installed.  Run:  pip install pynput")
    sys.exit(1)


def main() -> None:
    print("=" * 60)
    print("  Prototype v2 — Continuous Authentication Monitor")
    print("=" * 60)
    print("Loading KSD model…")

    buf    = KeystrokeBuffer()
    scorer = AnomalyScorer()
    try:
        scorer.load()
    except FileNotFoundError as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)

    result_q = queue.Queue()
    cv_event = threading.Event()

    # Fix 3 — consecutive anomaly counter
    consecutive_anomalies = 0
    last_anomaly_vec      = None   # feature vec that triggered the last alert

    # ── CV worker thread ──────────────────────────────────────────────────────
    def cv_worker():
        while True:
            cv_event.wait()      # block until a trigger is signalled
            cv_event.clear()
            print("\n[CV] Running identity check… (3 s webcam session)")
            try:
                match = run_cv_check()
            except Exception as exc:
                print(f"[CV] Error during check: {exc}")
                match = True     # fail-open on error
            result_q.put(match)

    threading.Thread(target=cv_worker, daemon=True, name="CV-worker").start()

    # ── Keystroke callbacks ───────────────────────────────────────────────────
    def on_press(key):
        buf.on_press(key)

    def on_release(key):
        nonlocal consecutive_anomalies, last_anomaly_vec

        buf.on_release(key)
        if not buf.ready():
            return

        # ── Score the completed window ────────────────────────────────────────
        events = buf.flush()
        vec    = extract_features(events)
        score  = scorer.score(vec)
        thresh = scorer.threshold()

        status_parts = [f"score={score:.3f}  threshold={thresh:.3f}"]

        if scorer.is_anomaly(score):
            # ── Anomalous window ────────────────────────────────────────────────────
            consecutive_anomalies += 1
            last_anomaly_vec       = vec
            status_parts.append(f"ANOMALY ({consecutive_anomalies}/1 consecutive)")

            if consecutive_anomalies >= 1:   # fire CV immediately on first anomalous window
                status_parts.append("→ triggering CV check")
                cv_event.set()
                consecutive_anomalies = 0    # reset after firing
        else:
            # ── Normal window — reset streak ────────────────────────────────────────
            if consecutive_anomalies > 0:
                status_parts.append(f"(streak reset from {consecutive_anomalies})")
            consecutive_anomalies = 0
            last_anomaly_vec      = None
            status_parts.append("OK")

        print("  ".join(status_parts))

        # ── Drain any pending CV result ───────────────────────────────────────
        try:
            match = result_q.get_nowait()
            if match:
                print("[CV] ✓ Owner confirmed — identity verified.")
                # Fix 4 — recalibration: add this confirmed vec to the scorer
                if last_anomaly_vec is not None:
                    scorer.add_confirmed(last_anomaly_vec)
                    last_anomaly_vec = None
            else:
                print(
                    "[CV] ✗ IDENTITY MISMATCH — possible impostor detected.\n"
                    "     Locking session."
                )
                ctypes.windll.user32.LockWorkStation()
        except queue.Empty:
            pass

    # ── Start ─────────────────────────────────────────────────────────────────
    print("Monitor running.  Type normally.  Press Ctrl+C to stop.\n")

    with keyboard.Listener(on_press=on_press, on_release=on_release):
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
