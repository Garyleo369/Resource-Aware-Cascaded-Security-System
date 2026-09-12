"""
main.py — Entry point for the two-tier cascaded continuous authentication system.

Thread topology
───────────────
  ┌──────────────┐   cv_trigger (Event)   ┌────────────────┐
  │  KSD Thread  │ ─────────────────────► │   CV Thread    │
  │  (Tier 1)    │                         │   (Tier 2)     │
  └──────────────┘                         └────────────────┘
        │                                         │
        └──────────────── result_queue ───────────┘
                               │
                        ┌──────▼──────┐
                        │  Main Thread│ (blocks on Ctrl-C)
                        └─────────────┘

Usage
─────
  python main.py            # run authentication system
  python enrol.py           # run one-time enrolment first
  streamlit run dashboard.py  # optional live dashboard
"""

import json
import os
import queue
import signal
import threading
import logging
import time

from ksd.listener   import KeystrokeListener
from ksd.features   import compute_features_from_raw
from ksd.model      import EnsembleKSDModel
from ksd.threshold  import DynamicThreshold
from cv.session     import capture_verification_session
from cv.gaze        import session_gaze_stats
from cv.geometry    import session_geometry_stats
from cv.matcher     import CVMatcher

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)-14s] [%(levelname)s] %(message)s",
)
log = logging.getLogger("main")

# ── Configuration ──────────────────────────────────────────────────────────────
KSD_WINDOW_SIZE = 80          # keystrokes per scoring window (raised to 80 to smooth transient typing variations)
CV_DURATION_S   = 3.0         # webcam capture duration (seconds)
CV_FPS          = 15.0        # target FPS during CV capture

# ── Persistent session log ─────────────────────────────────────────────────────
DATA_DIR     = os.path.join(os.path.dirname(__file__), "data")
SESSION_LOG  = os.path.join(DATA_DIR, "session_log.json")
_LOG_SCHEMA  = {"ksd_scores": [], "cv_results": [], "trigger_count": 0, "total_windows": 0, "ksd_triggered_events": []}
MAX_STORED   = 500   # cap stored scores


def _append_to_log(item: dict) -> None:
    """
    Thread-safe append of a result dict to the persistent session_log.json.
    KSD items update ksd_scores / total_windows; CV items update cv_results.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    try:
        if os.path.exists(SESSION_LOG):
            with open(SESSION_LOG, "r", encoding="utf-8") as f:
                data = json.load(f)
            for k, v in _LOG_SCHEMA.items():
                data.setdefault(k, v)
        else:
            data = dict(_LOG_SCHEMA)

        tier = item.get("tier")
        if tier == 1:
            data["ksd_scores"].append(round(item["score"], 6))
            data["ksd_scores"] = data["ksd_scores"][-MAX_STORED:]
            data["total_windows"] = data["total_windows"] + 1
            if item.get("triggered"):
                data["trigger_count"] = data["trigger_count"] + 1
                # Record the triggered event timestamp so the dashboard can detect it
                data.setdefault("ksd_triggered_events", [])
                data["ksd_triggered_events"].append(round(item["timestamp"], 3))
                data["ksd_triggered_events"] = data["ksd_triggered_events"][-MAX_STORED:]
        elif tier == 2:
            log_entry = {
                k: v for k, v in item.items()
                if k not in ("snapshot_b64", "tier")
            }
            data["cv_results"].append(log_entry)
            data["cv_results"] = data["cv_results"][-MAX_STORED:]

        with open(SESSION_LOG, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as exc:
        log.debug("session_log write error: %s", exc)


# ══════════════════════════════════════════════════════════════════════════════
# Tier 1 — Keystroke Dynamics thread
# ══════════════════════════════════════════════════════════════════════════════

def ksd_thread_fn(
    cv_trigger: threading.Event,
    result_queue: queue.Queue,
    stop_event: threading.Event,
) -> None:
    """
    Continuously listens for keystrokes, extracts features every 50-key
    window, scores each window with the IsolationForest, and fires the
    cv_trigger Event whenever the dynamic threshold is exceeded.
    """
    log.info("KSD thread starting.")
    model      = EnsembleKSDModel()
    thresholder = DynamicThreshold()

    # Raw event queue shared between pynput callback and this thread
    raw_q: queue.Queue = queue.Queue()
    listener = KeystrokeListener(event_queue=raw_q, stop_event=stop_event)
    listener.start()

    buffer: list[tuple] = []   # accumulated raw events for current window

    try:
        while not stop_event.is_set():
            try:
                event = raw_q.get(timeout=0.5)
            except queue.Empty:
                continue

            buffer.append(event)

            # Count only press events toward window size
            press_count = sum(1 for e in buffer if e[0] == "press")
            if press_count < KSD_WINDOW_SIZE:
                continue

            # ── Extract features ───────────────────────────────────────────────
            vec = compute_features_from_raw(buffer)
            buffer.clear()

            if vec is None:
                log.debug("Feature extraction returned None — skipping window.")
                continue

            # ── Score ──────────────────────────────────────────────────────────
            score = model.score(vec)
            triggered = thresholder.update(score)

            stats = thresholder.stats()
            ksd_item = {
                "tier":      1,
                "score":     round(score, 4),
                "threshold": round(stats["threshold"], 4) if stats["threshold"] else None,
                "triggered": triggered,
                "timestamp": time.time(),
            }
            result_queue.put(ksd_item)
            _append_to_log(ksd_item)

            # ── Wake CV if anomaly detected ────────────────────────────────────
            if triggered and not cv_trigger.is_set():
                log.warning("🚨 KSD anomaly — triggering CV verification.")
                cv_trigger.set()

    finally:
        listener.stop()
        log.info("KSD thread stopped.")


# ══════════════════════════════════════════════════════════════════════════════
# Tier 2 — Computer Vision verification thread
# ══════════════════════════════════════════════════════════════════════════════

def cv_thread_fn(
    cv_trigger: threading.Event,
    result_queue: queue.Queue,
    stop_event: threading.Event,
) -> None:
    """
    Sleeps until cv_trigger is set, captures a webcam session, computes
    Mahalanobis distances, and pushes the match result onto result_queue.
    Repeats until stop_event is set.
    """
    log.info("CV thread starting — waiting for trigger.")
    matcher = CVMatcher()

    while not stop_event.is_set():
        # ── Sleep until triggered ──────────────────────────────────────────────
        triggered = cv_trigger.wait(timeout=1.0)
        if not triggered:
            continue                        # keep waiting
        if stop_event.is_set():
            break

        cv_trigger.clear()                  # reset for next trigger
        log.info("CV thread woke — capturing %gs of webcam frames.", CV_DURATION_S)

        # ── Capture & analyse ──────────────────────────────────────────────────
        data = None
        try:
            data = capture_verification_session(
                duration_s=CV_DURATION_S,
                fps=CV_FPS,
            )
        finally:
            pass   # capture_verification_session handles cap.release() internally

        if data is None:
            log.error("CV session failed — no data returned.")
            result_queue.put({
                "tier":      2,
                "match":     False,
                "combined":  None,
                "error":     "No face data captured",
                "timestamp": time.time(),
            })
            continue

        gaze_stats = session_gaze_stats(data["gaze"])
        geom_stats = session_geometry_stats(data["geometry"])

        decision = matcher.match(
            session_gaze_mean=gaze_stats["mean"],
            session_geom_mean=geom_stats["mean"],
        )

        cv_item = {
            "tier":      2,
            "match":     decision["match"],
            "combined":  decision["combined"],
            "gaze_dist": decision["gaze_dist"],
            "geom_dist": decision["geom_dist"],
            "threshold": decision["threshold"],
            "timestamp": time.time(),
        }
        result_queue.put(cv_item)
        _append_to_log(cv_item)

        if decision["match"]:
            log.info("✅ CV verification PASSED — user identity confirmed.")
        else:
            log.warning("❌ CV verification FAILED — possible impostor.")

    log.info("CV thread stopped.")


# ══════════════════════════════════════════════════════════════════════════════
# Result consumer (runs in main thread)
# ══════════════════════════════════════════════════════════════════════════════

def consume_results(result_queue: queue.Queue, stop_event: threading.Event):
    """
    Drain the result queue and log each decision to stdout / a results log.
    Runs until stop_event is set and the queue is empty.
    """
    while not (stop_event.is_set() and result_queue.empty()):
        try:
            item = result_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        tier = item.get("tier")
        if tier == 1:
            log.info(
                "[KSD] score=%.4f  threshold=%s  triggered=%s",
                item["score"],
                item["threshold"],
                item["triggered"],
            )
        elif tier == 2:
            if item.get("error"):
                log.error("[CV ] error=%s", item["error"])
            else:
                status = "PASS ✅" if item["match"] else "FAIL ❌"
                log.info(
                    "[CV ] %s  combined=%.4f  gaze=%.4f  geom=%.4f",
                    status,
                    item["combined"],
                    item["gaze_dist"],
                    item["geom_dist"],
                )


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 60)
    log.info("Two-Tier Cascaded Continuous Authentication — starting")
    log.info("=" * 60)
    log.info("Press Ctrl+C to stop.")

    # ── Shared primitives ──────────────────────────────────────────────────────
    cv_trigger   = threading.Event()
    result_queue = queue.Queue()
    stop_event   = threading.Event()

    # ── Graceful shutdown on SIGINT (Ctrl-C) ───────────────────────────────────
    def _shutdown(sig, frame):
        log.info("Shutdown signal received — stopping threads…")
        stop_event.set()
        cv_trigger.set()   # unblock CV thread if waiting

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ── Launch daemon threads ──────────────────────────────────────────────────
    t_ksd = threading.Thread(
        target=ksd_thread_fn,
        args=(cv_trigger, result_queue, stop_event),
        name="KSD-Tier1",
        daemon=True,
    )
    t_cv = threading.Thread(
        target=cv_thread_fn,
        args=(cv_trigger, result_queue, stop_event),
        name="CV-Tier2",
        daemon=True,
    )

    t_ksd.start()
    t_cv.start()

    log.info("Threads launched: %s, %s", t_ksd.name, t_cv.name)

    # ── Block main thread — drain result queue until stopped ───────────────────
    try:
        consume_results(result_queue, stop_event)
    except KeyboardInterrupt:
        stop_event.set()
        cv_trigger.set()

    # ── Wait for threads to finish ─────────────────────────────────────────────
    t_ksd.join(timeout=5.0)
    t_cv.join(timeout=5.0)

    log.info("Authentication system shut down cleanly.")


if __name__ == "__main__":
    main()
