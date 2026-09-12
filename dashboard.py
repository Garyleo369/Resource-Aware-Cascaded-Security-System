"""
dashboard.py — Self-contained two-tier auth dashboard.

Everything lives here — no separate main.py required:
  • Built-in KSD background thread (pynput) scores every 50-keystroke window.
  • A Typing Pad gives the user a focused place to type inside the dashboard.
  • When a KSD anomaly is detected the webcam fires automatically.

Run with:
    streamlit run dashboard.py
"""

import os
import io
import json
import time
import queue
import base64
import logging
import threading
import collections
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="Two-Tier Auth Monitor",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR     = os.path.join(os.path.dirname(__file__), "data")
SESSION_LOG  = os.path.join(DATA_DIR, "session_log.json")

os.makedirs(DATA_DIR, exist_ok=True)

# ── v2 Paths (cascaded_auth/) ──────────────────────────────────────────────────
V2_DIR          = os.path.join(os.path.dirname(__file__), "cascaded_auth")
V2_MODEL_PATH   = os.path.join(V2_DIR, "model.joblib")
V2_ENROL_VEC    = os.path.join(V2_DIR, "enrol_vectors.joblib")
V2_GAZE_PATH    = os.path.join(V2_DIR, "gaze_profile.joblib")

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("dashboard")

# ── KSD constants ──────────────────────────────────────────────────────────────
try:
    from cascaded_auth.ksd import WINDOW as KSD_WINDOW_SIZE
except Exception:
    KSD_WINDOW_SIZE = 50

# ── Typing prompts ─────────────────────────────────────────────────────────────
TYPING_PROMPTS = [
    "The quick brown fox jumps over the lazy dog and keeps on running fast.",
    "Continuous authentication silently verifies your identity as you type.",
    "Security systems must balance usability with strong and robust protection.",
    "Keystroke dynamics analysis uses your unique typing rhythm as a biometric.",
    "Machine learning models can detect impostors by analysing key timing patterns.",
    "Please type this sentence naturally at your comfortable everyday typing pace.",
    "The dashboard monitors your keystrokes to build a continuous identity score.",
    "Authentication does not stop after login — it continues throughout the session.",
]

# ══════════════════════════════════════════════════════════════════════════════
# Styling
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background: #0d1117; }

div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #161b22 0%, #21262d 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}
div[data-testid="metric-container"] label {
    color: #8b949e !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
div[data-testid="metric-container"] div[data-testid="metric-value"] {
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: #e6edf3 !important;
}

.badge-ok   { background:#1a4a2e; color:#3fb950; border-radius:8px;
               padding:6px 16px; font-weight:700; display:inline-block;
               font-size:1.1rem; letter-spacing:0.04em; }
.badge-warn { background:#4a2a00; color:#f0883e; border-radius:8px;
               padding:6px 16px; font-weight:700; display:inline-block;
               font-size:1.1rem; }
.badge-fail { background:#4a1a1a; color:#f85149; border-radius:8px;
               padding:6px 16px; font-weight:700; display:inline-block;
               font-size:1.1rem; }

.alert-pass {
    background: linear-gradient(135deg, #0d2b1a, #1a4a2e);
    border: 2px solid #3fb950;
    border-radius: 12px;
    padding: 20px 24px;
    margin: 12px 0;
    animation: pulseGreen 1.5s ease-in-out;
}
.alert-fail {
    background: linear-gradient(135deg, #2b0d0d, #4a1a1a);
    border: 2px solid #f85149;
    border-radius: 12px;
    padding: 20px 24px;
    margin: 12px 0;
    animation: pulseRed 1.5s ease-in-out;
}
.alert-title  { font-size: 1.4rem; font-weight: 700; margin-bottom: 6px; }
.alert-detail { font-size: 0.9rem; color: #8b949e; }

/* ── Suspicion / threat banner ──────────────────────────────────────────── */
.suspicion-banner {
    background: linear-gradient(135deg, #3a0a0a 0%, #5c1010 50%, #3a0a0a 100%);
    border: 2px solid #f85149;
    border-radius: 14px;
    padding: 22px 28px;
    margin: 0 0 18px 0;
    display: flex;
    align-items: center;
    gap: 18px;
    animation: threatPulse 1.0s ease-in-out infinite;
    box-shadow: 0 0 40px rgba(248,81,73,0.35);
}
.suspicion-icon { font-size: 2.8rem; flex-shrink: 0; }
.suspicion-body { flex: 1; }
.suspicion-title {
    font-size: 1.45rem;
    font-weight: 800;
    color: #ff6b6b;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.suspicion-sub {
    font-size: 0.92rem;
    color: #e6a89e;
    line-height: 1.5;
}
.camera-activating {
    background: linear-gradient(135deg, #1a2a3a 0%, #0d1a2b 100%);
    border: 1.5px solid #58a6ff;
    border-radius: 10px;
    padding: 12px 18px;
    margin-top: 10px;
    font-size: 0.88rem;
    color: #79c0ff;
    display: flex;
    align-items: center;
    gap: 8px;
    animation: blink 1.2s step-start infinite;
}
.score-high {
    background: linear-gradient(135deg, #2b1510, #3d1a10);
    border: 1.5px solid #f0883e;
    border-radius: 10px;
    padding: 12px 16px;
    margin-top: 8px;
    animation: warnPulse 2s ease-in-out infinite;
}
@keyframes threatPulse {
    0%   { box-shadow: 0 0 20px rgba(248,81,73,0.25); border-color: #f85149; }
    50%  { box-shadow: 0 0 55px rgba(248,81,73,0.65); border-color: #ff8080; }
    100% { box-shadow: 0 0 20px rgba(248,81,73,0.25); border-color: #f85149; }
}
@keyframes warnPulse {
    0%, 100% { box-shadow: 0 0 8px rgba(240,136,62,0.2); }
    50%       { box-shadow: 0 0 22px rgba(240,136,62,0.55); }
}
@keyframes blink {
    0%, 100% { opacity: 1.0; }
    50%       { opacity: 0.5; }
}

@keyframes pulseGreen {
    0%   { box-shadow: 0 0 0 0 rgba(63,185,80,0.6); }
    70%  { box-shadow: 0 0 0 16px rgba(63,185,80,0); }
    100% { box-shadow: 0 0 0 0 rgba(63,185,80,0); }
}
@keyframes pulseRed {
    0%   { box-shadow: 0 0 0 0 rgba(248,81,73,0.6); }
    70%  { box-shadow: 0 0 0 16px rgba(248,81,73,0); }
    100% { box-shadow: 0 0 0 0 rgba(248,81,73,0); }
}

.typing-box {
    background: linear-gradient(135deg, #0d1117, #161b22);
    border: 1px solid #30363d;
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 16px;
}
.typing-prompt {
    font-size: 1.05rem;
    color: #58a6ff;
    font-style: italic;
    line-height: 1.7;
    margin-bottom: 16px;
    padding: 12px 16px;
    background: #0d1117;
    border-left: 3px solid #58a6ff;
    border-radius: 6px;
}
.score-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 12px 16px;
    margin-top: 8px;
}
.score-label { color: #8b949e; font-size: 0.72rem; text-transform: uppercase;
               letter-spacing: 0.06em; margin-bottom: 4px; }
.score-value-ok   { color: #3fb950; font-size: 1.9rem; font-weight: 700; }
.score-value-warn { color: #f0883e; font-size: 1.9rem; font-weight: 700; }
.score-value-bad  { color: #f85149; font-size: 1.9rem; font-weight: 700; }
.score-thr        { color: #8b949e; font-size: 0.8rem; margin-top: 2px; }
.anomaly-badge    { color: #f85149; font-weight: 700; margin-top: 6px; font-size: 0.95rem; }

h2, h3 { color: #e6edf3 !important; }
section[data-testid="stSidebar"] { background: #161b22; }
section[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
.stPlotlyChart { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Persistent session log helpers
# ══════════════════════════════════════════════════════════════════════════════

_LOG_LOCK = threading.Lock()   # protects file writes from the background thread

_LOG_SCHEMA = {
    "ksd_scores": [],
    "cv_results": [],
    "trigger_count": 0,
    "total_windows": 0,
    "ksd_triggered_events": [],
}

MAX_STORED_SCORES = 500


def _load_session_log() -> dict:
    """Load the persisted session log from disk, or return defaults."""
    if not os.path.exists(SESSION_LOG):
        return dict(_LOG_SCHEMA)
    try:
        with open(SESSION_LOG, "r", encoding="utf-8") as f:
            data = json.load(f)
        for k, v in _LOG_SCHEMA.items():
            data.setdefault(k, v)
        return data
    except Exception as exc:
        log.warning("Could not read session_log.json: %s", exc)
        return dict(_LOG_SCHEMA)


def _save_session_log(log_data: dict) -> None:
    """Persist the session log to disk (main Streamlit thread only)."""
    try:
        log_data["ksd_scores"] = log_data["ksd_scores"][-MAX_STORED_SCORES:]
        with open(SESSION_LOG, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2)
    except Exception as exc:
        log.warning("Could not write session_log.json: %s", exc)


def _clear_session_log() -> None:
    """Reset the persistent log to defaults."""
    _save_session_log(dict(_LOG_SCHEMA))


def _bg_append_ksd(score: float, triggered: bool, timestamp: float) -> None:
    """
    Thread-safe session log update called from the KSD background thread.
    Uses _LOG_LOCK so it doesn't corrupt concurrent writes from the main thread.
    """
    with _LOG_LOCK:
        try:
            data = _load_session_log()
            data["ksd_scores"].append(round(score, 6))
            data["ksd_scores"] = data["ksd_scores"][-MAX_STORED_SCORES:]
            data["total_windows"] = data.get("total_windows", 0) + 1
            if triggered:
                data["trigger_count"] = data.get("trigger_count", 0) + 1
                data.setdefault("ksd_triggered_events", [])
                data["ksd_triggered_events"].append(round(timestamp, 3))
                data["ksd_triggered_events"] = (
                    data["ksd_triggered_events"][-MAX_STORED_SCORES:]
                )
            with open(SESSION_LOG, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as exc:
            log.debug("bg_append_ksd error: %s", exc)


# ══════════════════════════════════════════════════════════════════════════════
# Camera capture & CV verification
# ══════════════════════════════════════════════════════════════════════════════

def _capture_frame(camera_index: int = 0) -> np.ndarray | None:
    """Grab a single frame from the webcam. Returns BGR array or None."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return None
    try:
        for _ in range(5):
            cap.read()
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def _frame_to_b64(frame: np.ndarray) -> str:
    """Encode a BGR frame as a base64 JPEG string for HTML embedding."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    buf = io.BytesIO()
    from PIL import Image
    Image.fromarray(rgb).save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _run_cv_verification() -> dict:
    """
    Capture a short webcam session, extract v2 MediaPipe gaze vectors, and compare
    them against the enrolled cascaded_auth gaze profile.
    Returns a result dict (including a JPEG snapshot as base64).
    """
    try:
        import joblib
        from cascaded_auth.cv_check import (
            MATCH_THRESHOLD,
            PROFILE_FILE,
            SESSION_SECS,
            collect_gaze_vectors,
        )
    except ImportError as exc:
        return {
            "error": f"Import failed: {exc}", "match": False,
            "combined": None, "gaze_dist": None, "geom_dist": None,
            "timestamp": time.time(), "snapshot_b64": None,
        }

    # Grab a quick snapshot for the dashboard display
    snapshot_b64 = None
    raw_frame = _capture_frame()
    if raw_frame is not None:
        try:
            snapshot_b64 = _frame_to_b64(raw_frame)
        except Exception:
            pass

    if not os.path.exists(PROFILE_FILE):
        return {
            "error": "No v2 gaze profile found", "match": False,
            "combined": None, "gaze_dist": None, "geom_dist": None,
            "threshold": MATCH_THRESHOLD,
            "timestamp": time.time(), "snapshot_b64": snapshot_b64,
        }

    try:
        profile = joblib.load(PROFILE_FILE)
    except Exception as exc:
        return {
            "error": f"Failed to load v2 gaze profile: {exc}", "match": False,
            "combined": None, "gaze_dist": None, "geom_dist": None,
            "threshold": MATCH_THRESHOLD,
            "timestamp": time.time(), "snapshot_b64": snapshot_b64,
        }

    vectors = collect_gaze_vectors(SESSION_SECS)
    if not vectors:
        return {
            "error": "No face detected in camera", "match": False,
            "combined": None, "gaze_dist": None, "geom_dist": None,
            "threshold": MATCH_THRESHOLD,
            "timestamp": time.time(), "snapshot_b64": snapshot_b64,
        }

    try:
        session_vec = np.mean(np.array(vectors, dtype=np.float64), axis=0)
        mean_vec = np.asarray(profile["mean"], dtype=np.float64)
        cov = np.asarray(profile["cov"], dtype=np.float64)
        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(cov)
    except Exception as exc:
        return {
            "error": f"Invalid v2 gaze profile: {exc}", "match": False,
            "combined": None, "gaze_dist": None, "geom_dist": None,
            "threshold": MATCH_THRESHOLD,
            "timestamp": time.time(), "snapshot_b64": snapshot_b64,
        }

    try:
        from scipy.spatial.distance import mahalanobis
        dist = float(mahalanobis(session_vec, mean_vec, cov_inv))
    except Exception:
        diff = session_vec - mean_vec
        dist = float(np.sqrt(diff @ diff))

    match = dist < MATCH_THRESHOLD

    return {
        "match":        match,
        "combined":     round(dist, 4),
        "gaze_dist":    round(dist, 4),
        "geom_dist":    None,
        "threshold":    MATCH_THRESHOLD,
        "n_frames":     len(vectors),
        "timestamp":    time.time(),
        "snapshot_b64": snapshot_b64,
        "error":        None,
    }


# ══════════════════════════════════════════════════════════════════════════════
# KSD Background Thread — singleton via @st.cache_resource
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def _start_ksd_background() -> dict:
    """
    Launch the cascaded_auth v2 KSD listener + scorer as a background daemon thread.
    """
    from cascaded_auth.ksd import KeystrokeBuffer, extract_features, AnomalyScorer
    from pynput import keyboard

    # Thread-safe communication channels
    result_deque = collections.deque(maxlen=200)   # KSD results → main thread
    raw_q        = queue.Queue()
    stop_event   = threading.Event()

    live = {
        "press_count":         0,
        "last_score":          None,
        "last_thr":            None,
        "last_triggered":      False,
        "total_windows":       0,
        "last_typing_speed":   None,
        "last_backspace_rate": None,
        "last_digraph_var":    None,
        "last_dwell_mean":     None,
        "last_flight_mean":    None,
        "listener_active":      False,
        "listener_error":       None,
        "enrol_active":         False,
        "enrol_vectors":        [],
    }

    scorer = AnomalyScorer()
    try:
        scorer.load()
    except FileNotFoundError:
        pass

    live["_model_ref"] = scorer

    def _worker():
        buffer = KeystrokeBuffer()

        def on_press(key):
            try:
                buffer.on_press(key)
                live["press_count"] = len(buffer._buffer)
            except Exception as exc:
                live["listener_error"] = str(exc)
                log.debug("[KSD-BG] on_press failed: %s", exc)

        def on_release(key):
            try:
                buffer.on_release(key)
                live["press_count"] = len(buffer._buffer)
                if buffer.ready():
                    events = buffer.flush()
                    live["press_count"] = 0
                    vec = extract_features(events)
                    if live.get("enrol_active") and vec[0] > 0.01:
                        live.setdefault("enrol_vectors", []).append(vec)
                    raw_q.put(vec)
            except Exception as exc:
                live["listener_error"] = str(exc)
                log.debug("[KSD-BG] on_release failed: %s", exc)

        listener = None
        try:
            listener = keyboard.Listener(on_press=on_press, on_release=on_release)
            listener.start()
            live["listener_active"] = True
            live["listener_error"] = None
        except Exception as exc:
            live["listener_active"] = False
            live["listener_error"] = str(exc)
            log.exception("[KSD-BG] failed to start keyboard listener")
            return

        try:
            while not stop_event.is_set():
                if live.get("_reload_flag"):
                    live["_reload_flag"] = False
                    try:
                        scorer.load()
                        log.info("[KSD-BG] v2 model hot-reloaded from disk.")
                    except FileNotFoundError:
                        pass

                try:
                    vec = raw_q.get(timeout=0.5)
                except queue.Empty:
                    continue

                try:
                    score     = scorer.score(vec)
                    thresh    = scorer.threshold()
                    triggered = scorer.is_anomaly(score)
                    ts        = time.time()

                    live["last_score"]     = round(score, 4)
                    live["last_thr"]       = round(thresh, 4)
                    live["last_triggered"] = triggered
                    live["total_windows"]  = live["total_windows"] + 1

                    if len(vec) >= 12:
                        live["last_typing_speed"]   = round(float(vec[10]), 3)
                        live["last_backspace_rate"] = 0.0
                        live["last_digraph_var"]    = round(float(vec[11]), 2) # Burst ratio
                        live["last_dwell_mean"]     = round(float(vec[0]),  2)
                        live["last_flight_mean"]    = round(float(vec[5]),  2)

                    result_deque.append({
                        "score":     round(score, 4),
                        "threshold": round(thresh, 4),
                        "triggered": triggered,
                        "timestamp": ts,
                    })

                    _bg_append_ksd(score, triggered, ts)

                    log.info(
                        "[KSD-BG] score=%.4f  threshold=%.4f  triggered=%s  dim=%d",
                        score, thresh, triggered, len(vec),
                    )

                except Exception as _win_exc:
                    log.warning(
                        "[KSD-BG] window scoring failed (vec dim=%d): %s  "
                        "→ Go to Enrolment page to retrain the model.",
                        len(vec), _win_exc,
                    )
                    live["last_score"] = None

        finally:
            live["listener_active"] = False
            if listener is not None:
                listener.stop()
            log.info("[KSD-BG] background thread stopped.")

    t = threading.Thread(target=_worker, name="KSD-BG", daemon=True)
    t.start()
    log.info("[KSD-BG] background thread started (pid=%d).", os.getpid())

    return {
        "result_deque": result_deque,
        "live":         live,
        "stop_event":   stop_event,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Session-state initialisation (runs once per browser session)
# ══════════════════════════════════════════════════════════════════════════════

MAX_HISTORY = 60   # scores to keep in the in-memory chart deque

if "initialised" not in st.session_state:
    log_data = _load_session_log()
    recent   = log_data["ksd_scores"][-MAX_HISTORY:]
    st.session_state.ksd_scores        = collections.deque(recent, maxlen=MAX_HISTORY)
    st.session_state.ksd_thresholds    = collections.deque(maxlen=MAX_HISTORY)
    st.session_state.cv_results        = log_data["cv_results"][-20:]
    st.session_state.last_cv           = (log_data["cv_results"][-1]
                                          if log_data["cv_results"] else None)
    st.session_state.trigger_count     = log_data["trigger_count"]
    st.session_state.total_windows     = log_data["total_windows"]
    st.session_state.cv_running        = False
    st.session_state.prompt_idx        = 0
    st.session_state.suspicion_active  = False   # True when KSD anomaly raised
    st.session_state.last_suspicion_ts = 0.0     # epoch of last suspicion raise
    st.session_state.last_cv_run_ts    = 0.0     # epoch of last CV run completion
    st.session_state.initialised       = True

# ── Start (or retrieve cached) background KSD thread ──────────────────────────
ksd_bg = _start_ksd_background()


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🔐 Auth Monitor")
    st.markdown("---")

    st.markdown("### Navigation")
    app_mode = st.radio("Go to", ["Monitor", "Enrolment"], label_visibility="collapsed")
    st.markdown("---")

    model_ok   = os.path.exists(V2_MODEL_PATH)
    profile_ok = os.path.exists(V2_GAZE_PATH)

    st.markdown("### System Readiness")
    st.markdown(
        f"{'✅' if model_ok   else '❌'} **KSD Model** "
        f"({'enrolled' if model_ok else 'use Enrolment'})"
    )
    st.markdown(
        f"{'✅' if profile_ok else '❌'} **Gaze Profile** "
        f"({'enrolled' if profile_ok else 'use Enrolment'})"
    )
    if not (model_ok and profile_ok):
        st.warning("Complete Enrolment before monitoring.")

    st.markdown("---")
    st.markdown("### Controls")
    auto_refresh = st.checkbox("Auto-refresh (2 s)", value=True)
    refresh_btn  = st.button("🔄 Refresh Now")

    st.markdown("---")
    st.markdown("### Camera")
    manual_trigger = st.button("📷 Trigger CV Check Now")
    st.caption("CV fires automatically on every real KSD anomaly.")

    st.markdown("---")
    st.markdown("### Typing Prompt")
    if st.button("🔀 New Prompt"):
        st.session_state.prompt_idx = (
            (st.session_state.prompt_idx + 1) % len(TYPING_PROMPTS)
        )

    st.markdown("---")
    st.markdown("### Session")
    if st.button("🗑 Clear History"):
        _clear_session_log()
        st.session_state.ksd_scores     = collections.deque(maxlen=MAX_HISTORY)
        st.session_state.ksd_thresholds = collections.deque(maxlen=MAX_HISTORY)
        st.session_state.cv_results     = []
        st.session_state.last_cv        = None
        st.session_state.trigger_count  = 0
        st.session_state.total_windows  = 0
        st.success("History cleared.")

    st.markdown("---")
    st.markdown("### Legend")
    st.markdown("🟢 **PASS** — identity confirmed")
    st.markdown("🔴 **FAIL** — possible impostor")
    st.markdown("🚨 **ANOMALY** — CV triggered")
    st.markdown("---")
    st.caption("Two-Tier Auth Prototype · KSD v2")





# ══════════════════════════════════════════════════════════════════════════════
# Enrolment Page (v2 — Cascaded Auth)
# ══════════════════════════════════════════════════════════════════════════════
if app_mode == "Enrolment":

    # ── v2 window size (must match cascaded_auth/ksd.py) ─────────────────────
    from cascaded_auth.ksd import WINDOW as _V2_WINDOW
    _V2_MIN_WINS = 10    # minimum windows for a reliable IsolationForest baseline
    _V2_TARGET   = _V2_WINDOW * _V2_MIN_WINS  # = 250 keystrokes

    st.markdown("# 📝 Enrolment — Cascaded Auth")
    st.markdown(
        "Build the **v2 keystroke baseline** (12-D feature vector, "
        "K=2.5, FLOOR=0.60, consecutive-anomaly buffer=3) and the **MediaPipe gaze profile** "
        "used by `cascaded_auth/monitor.py`."
    )

    # ── Model Health Card ─────────────────────────────────────────────────────
    # Shown whenever a v2 model + enrol_vectors file are both on disk.
    # Probes the saved model against the enrolment vectors so the user
    # can immediately see whether the IsolationForest is well-trained.
    if os.path.exists(V2_MODEL_PATH) and os.path.exists(V2_ENROL_VEC):
        try:
            import joblib as _jl_h
            import numpy as _np_h
            from cascaded_auth.ksd import AnomalyScorer as _HSc
            _hsc = _HSc()
            _hsc.load()
            _hvecs = _jl_h.load(V2_ENROL_VEC)          # shape (N, 12)
            _hraw  = _hsc.model.score_samples(_hvecs)   # raw IF scores
            _hoffset = float(_hsc.model.offset_)
            _hnorm = [float(1.0 / (1.0 + _np_h.exp(10.0 * (r - _hoffset)))) for r in _hraw]
            _hmean = float(_np_h.mean(_hnorm))
            _hmax  = float(_np_h.max(_hnorm))
            _hn    = len(_hvecs)

            # Colour: green ≤ 0.25 (good), amber ≤ 0.50, red > 0.50
            if _hmean <= 0.25:
                _hc, _hi, _ht = "#3fb950", "✅", "Good — model trained on sufficient data"
            elif _hmean <= 0.50:
                _hc, _hi, _ht = "#f0883e", "⚠️", "Acceptable — more windows will improve accuracy"
            else:
                _hc, _hi, _ht = "#f85149", "❌", "Undertrained — enrol more windows before running evaluate.py"

            st.markdown(
                f'<div style="background:#161b22;border:2px solid {_hc};border-radius:12px;'
                f'padding:16px 22px;margin-bottom:16px;">'
                f'<div style="font-size:1.1rem;font-weight:700;color:{_hc};margin-bottom:6px;">'
                f'{_hi} Model Health · {_hn} enrolment windows</div>'
                f'<div style="color:#c9d1d9;font-size:0.88rem;line-height:1.8;">'
                f'Normalised score on own typing — '
                f'<strong style="color:{_hc};">mean {_hmean:.3f}</strong> · '
                f'<strong>max {_hmax:.3f}</strong><br>'
                f'<span style="color:#8b949e;font-size:0.80rem;">'
                f'Ideal: mean ≤ 0.25. Scores near 1.0 mean the model cannot distinguish '
                f'your typing from anomalies — re-enrol with 50+ windows.</span><br>'
                f'<span style="color:{_hc};font-size:0.82rem;font-weight:600;">{_ht}</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )
        except Exception:
            pass
    st.markdown("---")

    # ── Readiness badges ──────────────────────────────────────────────────────
    _v2r1, _v2r2, _v2r3 = st.columns(3)
    _v2_ksd_ok  = os.path.exists(V2_MODEL_PATH)
    _v2_gaze_ok = os.path.exists(V2_GAZE_PATH)

    for _col, _ok, _label, _detail in [
        (_v2r1, _v2_ksd_ok,  "KSD Model",    "model.joblib"),
        (_v2r2, _v2_gaze_ok, "Gaze Profile",  "gaze_profile.joblib"),
        (_v2r3, _v2_ksd_ok and _v2_gaze_ok, "System", "Ready" if (_v2_ksd_ok and _v2_gaze_ok) else "Incomplete"),
    ]:
        _c = "#3fb950" if _ok else "#f85149"
        _i = "✅" if _ok else "❌"
        _col.markdown(
            f'<div style="background:#161b22;border:1px solid {_c};border-radius:10px;'
            f'padding:14px 18px;text-align:center;">'
            f'<div style="font-size:1.6rem;">{_i}</div>'
            f'<div style="color:{_c};font-weight:700;font-size:0.9rem;margin-top:4px;">{_label}</div>'
            f'<div style="color:#8b949e;font-size:0.78rem;">{_detail}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.markdown("---")

    _v2_col_gaze, _v2_col_ksd = st.columns(2, gap="large")

    # ─────────────────────────────────────────────────────────────────────────
    # Left column — Gaze Profile (MediaPipe)
    # ─────────────────────────────────────────────────────────────────────────
    with _v2_col_gaze:
        st.markdown("### 👁 Gaze Profile (MediaPipe)")
        st.markdown(
            "Sit naturally in front of the webcam. The system captures "
            "**30 seconds** of MediaPipe iris gaze vectors and saves a "
            "Mahalanobis distance profile used by `cv_check.py`."
        )
        st.markdown(
            '<ol style="color:#8b949e;font-size:0.88rem;line-height:1.9;padding-left:18px;margin-top:8px;">'
            '<li>Position your face in the webcam</li>'
            '<li>Click <strong style="color:#e6edf3;">Start 30 s Gaze Capture</strong></li>'
            '<li>Look naturally at the screen for 30 seconds</li>'
            '<li>Done — profile saved automatically</li>'
            '</ol>',
            unsafe_allow_html=True,
        )

        if _v2_gaze_ok:
            try:
                import joblib as _jl_g
                _gp = _jl_g.load(V2_GAZE_PATH)
                st.markdown(
                    f'<div style="background:#0d2b1a;border:1px solid #3fb950;border-radius:8px;'
                    f'padding:10px 14px;font-size:0.83rem;color:#7ee2a8;margin-bottom:10px;">'
                    f'✅ Gaze profile on disk · {_gp.get("n", "?")} frames</div>',
                    unsafe_allow_html=True,
                )
            except Exception:
                pass

        if st.button("👁 Start 30 s Gaze Capture", use_container_width=True, key="v2_gaze_btn"):
            try:
                from cascaded_auth.cv_check import collect_gaze_vectors, save_gaze_profile
                _gaze_prog = st.progress(0.0, text="Opening webcam…")
                _gaze_info = st.info("Look at the screen naturally for 30 seconds…")

                import cv2 as _cv2_g
                import mediapipe as _mp_g

                _mp_face_g = _mp_g.solutions.face_mesh
                _cap_g     = _cv2_g.VideoCapture(0)
                _vecs_g    = []
                _end_g     = time.time() + 30.0
                _ph_g      = st.empty()

                with _mp_face_g.FaceMesh(refine_landmarks=True, max_num_faces=1) as _mesh_g:
                    while time.time() < _end_g:
                        _ok_g, _fr_g = _cap_g.read()
                        if not _ok_g:
                            continue
                        _h_g, _w_g = _fr_g.shape[:2]
                        _rgb_g = _cv2_g.cvtColor(_fr_g, _cv2_g.COLOR_BGR2RGB)
                        _res_g = _mesh_g.process(_rgb_g)
                        if _res_g.multi_face_landmarks:
                            from cascaded_auth.cv_check import get_gaze_vector
                            _vv = get_gaze_vector(_res_g.multi_face_landmarks[0], _w_g, _h_g)
                            _vecs_g.append(_vv)
                        _rem_g = max(0, _end_g - time.time())
                        _pct_g = min(1.0, 1.0 - _rem_g / 30.0)
                        _gaze_prog.progress(_pct_g, text=f"{len(_vecs_g)} frames | {_rem_g:.0f}s remaining")
                        _ph_g.image(_fr_g, channels="BGR")

                _cap_g.release()
                _ph_g.empty()
                _gaze_prog.empty()
                _gaze_info.empty()

                if not _vecs_g:
                    st.error("❌ No face detected — check webcam and try again.")
                else:
                    save_gaze_profile(_vecs_g)
                    st.success(f"✅ Gaze profile saved! · {len(_vecs_g)} frames")
                    st.rerun()

            except ImportError as _gimp:
                st.error(f"❌ Import error: {_gimp}. Ensure mediapipe is installed.")
            except Exception as _ge:
                st.error(f"❌ Gaze capture failed: {_ge}")

    # ─────────────────────────────────────────────────────────────────────────
    # Right column — KSD Enrolment
    # ─────────────────────────────────────────────────────────────────────────
    with _v2_col_ksd:
        st.markdown("### ⌨️ KSD Enrolment (12-D)")
        st.markdown(
            f"Type naturally. The system needs **at least {_V2_TARGET:,} key presses** "
            f"({_V2_MIN_WINS}+ windows of {_V2_WINDOW} keystrokes each) to train a "
            "reliable IsolationForest baseline. More windows → lower false-alarm rate."
        )
        # ── Point 1: why 50 windows are required ─────────────────────────────
        st.info(
            f"📌 **Why {_V2_MIN_WINS} windows?** The IsolationForest needs at least "
            f"30–50 samples to learn your normal typing distribution. With fewer samples "
            "it scores *everything* as anomalous (normalised score ≈ 1.0), causing the "
            "dynamic threshold to saturate and preventing intruder detection."
        )
        st.markdown(
            '<ol style="color:#8b949e;font-size:0.88rem;line-height:1.9;padding-left:18px;margin-top:8px;">'
            '<li>Click <strong style="color:#e6edf3;">▶️ Start Enrolment</strong></li>'
            '<li>Type naturally in the pad below (mix fast, slow, code, prose)</li>'
            '<li>Keep typing until the bar reaches 100%</li>'
            '<li>Click <strong style="color:#e6edf3;">⏹️ Stop &amp; Save Model</strong></li>'
            '</ol>',
            unsafe_allow_html=True,
        )

        # ── State init ────────────────────────────────────────────────────────
        if "enrol_v2_state" not in st.session_state:
            st.session_state.enrol_v2_state = {"active": False, "events": []}

        _vs = st.session_state.enrol_v2_state

        # ── Existing model badge ───────────────────────────────────────────────
        if _v2_ksd_ok and not _vs["active"]:
            try:
                import joblib as _jl_k
                _vm = _jl_k.load(V2_MODEL_PATH)
                st.markdown(
                    f'<div style="background:#0d2b1a;border:1px solid #3fb950;border-radius:8px;'
                    f'padding:10px 14px;font-size:0.83rem;color:#7ee2a8;margin-bottom:10px;">'
                    f'✅ KSD model on disk · {getattr(_vm, "n_estimators", "?")} estimators · '
                    f'Re-enrol to update.</div>',
                    unsafe_allow_html=True,
                )
            except Exception:
                pass

        if not _vs["active"]:
            if st.button("▶️ Start Enrolment", use_container_width=True, key="v2_ksd_start"):
                ksd_bg["live"]["enrol_vectors"] = []
                ksd_bg["live"]["enrol_active"] = True
                _vs["active"] = True
                st.rerun()

        else:
            _v2_wins = len(ksd_bg["live"].get("enrol_vectors", []))

            _v2_btns = st.columns([1, 1])
            with _v2_btns[0]:
                if st.button(
                    "⏹️ Stop & Save Model",
                    type="primary",
                    use_container_width=True,
                    key="v2_ksd_stop",
                ):
                    _vs["active"] = False
                    ksd_bg["live"]["enrol_active"] = False

                    import numpy as _np2
                    from cascaded_auth.ksd import AnomalyScorer as _V2Scorer

                    _v2_wins_list = list(ksd_bg["live"].get("enrol_vectors", []))

                    if len(_v2_wins_list) < _V2_MIN_WINS:
                        st.error(
                            f"❌ Only {len(_v2_wins_list)} valid windows extracted — "
                            f"need ≥{_V2_MIN_WINS}. Type more and try again."
                        )
                    else:
                        _v2_X = _np2.array(_v2_wins_list, dtype=_np2.float64)
                        with st.spinner(
                            f"Training v2 IsolationForest on {len(_v2_wins_list)} windows "
                            f"(dim={_v2_X.shape[1]})…"
                        ):
                            _v2sc = _V2Scorer()
                            _v2sc.fit(_v2_X, save_vectors=True)
                            ksd_bg["live"]["_reload_flag"] = True

                        # ── Point 2: model health check immediately after save ──
                        _raw_chk  = _v2sc.model.score_samples(_v2_X)
                        _v2_offset = float(_v2sc.model.offset_)
                        _norm_chk = [float(1.0 / (1.0 + _np2.exp(10.0 * (r - _v2_offset)))) for r in _raw_chk]
                        _mean_chk = float(_np2.mean(_norm_chk))
                        _max_chk  = float(_np2.max(_norm_chk))

                        if _mean_chk <= 0.25:
                            _health_msg = f"✅ Model health: **GOOD** (mean score {_mean_chk:.3f} ≤ 0.25)"
                            _health_fn  = st.success
                        elif _mean_chk <= 0.50:
                            _health_msg = (
                                f"⚠️ Model health: **ACCEPTABLE** (mean score {_mean_chk:.3f}). "
                                "Consider re-enrolling with more windows for better accuracy."
                            )
                            _health_fn = st.warning
                        else:
                            _health_msg = (
                                f"❌ Model health: **POOR** (mean score {_mean_chk:.3f} > 0.50). "
                                "The IsolationForest is still undertrained. "
                                f"Discard and re-enrol with 100+ windows for reliable detection."
                            )
                            _health_fn = st.error

                        st.success(
                            f"✅ KSD model saved! · {len(_v2_wins_list)} windows · "
                            f"feature dim={_v2_X.shape[1]}"
                        )
                        _health_fn(_health_msg)
                        st.caption(
                            f"Normalised score on own enrolment data — "
                            f"mean: {_mean_chk:.4f} · max: {_max_chk:.4f} · "
                            f"(ideal: mean ≤ 0.25, max ≤ 0.50)"
                        )
                        st.info("Now run `python cascaded_auth/monitor.py` to start monitoring.")
                        st.rerun()

            with _v2_btns[1]:
                if st.button("🗑 Discard", use_container_width=True, key="v2_ksd_discard"):
                    _vs["active"] = False
                    ksd_bg["live"]["enrol_active"] = False
                    ksd_bg["live"]["enrol_vectors"] = []
                    st.rerun()

        # Live typing pad — JS tracks keystrokes in real time
        import streamlit.components.v1 as _enrol_components
        _border_color   = '#f85149' if _vs['active'] else '#30363d'
        _label_color    = '#f85149' if _vs['active'] else '#8b949e'
        _blink_css      = 'animation: blink 1s step-end infinite;' if _vs['active'] else ''
        _label_text     = '\U0001f534 Recording\u2026 type anything below' if _vs['active'] else '\u23f8\ufe0f Paused \u2014 click \u25b6\ufe0f Start Enrolment above first'
        _placeholder    = 'Type naturally here\u2026 fast, slow, code, prose \u2014 anything.' if _vs['active'] else 'Start enrolment first, then type here\u2026'
        _autofocus_attr = 'autofocus' if _vs['active'] else ''
        _enrol_components.html(
            f"""
            <!DOCTYPE html><html><head><meta charset="utf-8">
            <style>
              * {{ box-sizing: border-box; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
              body {{ margin: 0; padding: 0; background: transparent; color: #e6edf3; }}
              .outer {{
                background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
                border: 1px solid {_border_color};
                border-radius: 12px; padding: 16px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.3);
              }}
              .label {{
                font-size: 0.85rem; font-weight: 700;
                color: {_label_color};
                margin-bottom: 8px;
                {_blink_css}
              }}
              @keyframes blink {{ 50%{{ opacity:0.4 }} }}
              textarea {{
                width: 100%; height: 110px;
                background: #0d1117; border: 1px solid #30363d; border-radius: 8px;
                color: #e6edf3; padding: 12px; font-size: 15px; line-height: 1.5;
                resize: none; outline: none; transition: border-color 0.2s;
              }}
              textarea:focus {{ border-color: #58a6ff; }}
              .total-header {{
                display: flex; justify-content: space-between; align-items: center;
                font-size: 12px; font-weight: 700; margin: 14px 0 5px;
              }}
              .total-lbl {{ color: #c9d1d9; }}
              .total-cnt {{ color: #e6edf3; font-family: monospace; font-size: 13px; }}
              .total-bg  {{
                height: 14px; background: #21262d; border-radius: 7px; overflow: hidden;
              }}
              .total-fill {{
                height: 100%; width: 0%;
                background: linear-gradient(90deg, #58a6ff, #3fb950);
                border-radius: 7px; transition: width 0.1s ease-out;
              }}
              .total-status {{ margin: 5px 0 12px; font-size: 12px; color: #8b949e; min-height: 16px; }}
              .meter-header {{
                display: flex; justify-content: space-between; align-items: center;
                font-size: 11px; font-weight: 600; margin: 6px 0 4px;
              }}
              .meter-lbl {{ color: #8b949e; }}
              .meter-cnt {{ color: #58a6ff; font-family: monospace; }}
              .meter-bg  {{
                height: 8px; background: #21262d; border-radius: 4px; overflow: hidden;
              }}
              .meter-fill {{
                height: 100%; width: 0%;
                background: linear-gradient(90deg, #58a6ff, #3fb950);
                border-radius: 4px; transition: width 0.06s ease-out;
              }}
              .meter-status {{ margin-top: 4px; font-size: 11px; color: #8b949e; font-style: italic; min-height: 14px; }}
              .badge {{ color: #3fb950; font-weight: bold; }}
            </style></head><body>
            <div class="outer">
              <div class="label">{_label_text}</div>
              <textarea id="ep" placeholder="{_placeholder}" {_autofocus_attr}></textarea>

              <div class="total-header">
                <span class="total-lbl">&#128202; Total progress</span>
                <span class="total-cnt" id="tc">0 / {_V2_MIN_WINS} windows</span>
              </div>
              <div class="total-bg"><div class="total-fill" id="tf"></div></div>
              <div class="total-status" id="ts">&#9203; Start typing to begin enrolment&hellip;</div>

              <div class="meter-header">
                <span class="meter-lbl">&#9889; Current window</span>
                <span class="meter-cnt" id="cnt">0 / {_V2_WINDOW}</span>
              </div>
              <div class="meter-bg"><div class="meter-fill" id="mf"></div></div>
              <div class="meter-status" id="wst"></div>
            </div>
            <script>
              const pad = document.getElementById('ep');
              const mf  = document.getElementById('mf');
              const cnt = document.getElementById('cnt');
              const wst = document.getElementById('wst');
              const tf  = document.getElementById('tf');
              const tc  = document.getElementById('tc');
              const ts  = document.getElementById('ts');
              const W        = {_V2_WINDOW};
              const MIN_WINS = {_V2_MIN_WINS};
              let totalKeys = 0;

              pad.addEventListener('keydown', e => {{
                if (['Shift','Control','Alt','Meta','CapsLock'].includes(e.key)) return;
                totalKeys++;

                const totalWins = Math.floor(totalKeys / W);
                const winPct    = Math.min((totalWins / MIN_WINS) * 100, 100);
                tf.style.width  = winPct + '%';

                if (winPct >= 100) {{
                  tf.style.background = 'linear-gradient(90deg,#3fb950,#2ea043)';
                  tc.textContent      = totalWins + ' / ' + MIN_WINS + ' windows \u2705';
                  ts.innerHTML        = '<span style="color:#3fb950;font-weight:700">\u2705 Minimum reached! Click \u23f9\ufe0f Stop & Save Model.</span>';
                }} else if (totalWins > 0) {{
                  tf.style.background = 'linear-gradient(90deg,#58a6ff,#d29922)';
                  tc.textContent      = totalWins + ' / ' + MIN_WINS + ' windows';
                  ts.textContent      = '🟧 ' + (MIN_WINS - totalWins) + ' more windows to go\u2026';
                }} else {{
                  tf.style.background = 'linear-gradient(90deg,#58a6ff,#79c0ff)';
                  tc.textContent      = '0 / ' + MIN_WINS + ' windows';
                  ts.textContent      = '🟨 Accumulating\u2026 ' + totalKeys + ' keys so far';
                }}

                const inWin = totalKeys % W || W;
                const pct   = (inWin / W) * 100;
                mf.style.width  = pct + '%';
                cnt.textContent = inWin + ' / ' + W;

                if (pct < 40)       {{ mf.style.background='linear-gradient(90deg,#58a6ff,#79c0ff)'; wst.textContent='\U0001F7E1 Accumulating\u2026'; }}
                else if (pct < 80)  {{ mf.style.background='linear-gradient(90deg,#58a6ff,#d29922)'; wst.textContent='\U0001F7E0 Getting there \u2014 ' + (W - inWin) + ' keys left'; }}
                else if (pct < 100) {{ mf.style.background='linear-gradient(90deg,#d29922,#3fb950)'; wst.textContent='\U0001F525 Almost there \u2014 ' + (W - inWin) + ' more!'; }}
                else                {{ mf.style.background='linear-gradient(90deg,#3fb950,#2ea043)'; wst.innerHTML='<span class="badge">\u2705 Window ' + totalWins + ' captured!</span>'; }}
              }});
            </script></body></html>
            """,
            height=380,
        )

    st.stop()  # Do not render Monitor below


# ══════════════════════════════════════════════════════════════════════════════
# Drain background KSD result deque into session state
# ══════════════════════════════════════════════════════════════════════════════

new_ksd_triggered = False
auto_triggered    = False

pending = []
dq = ksd_bg["result_deque"]
while dq:
    try:
        pending.append(dq.popleft())
    except IndexError:
        break

for item in pending:
    st.session_state.ksd_scores.append(item["score"])
    if "threshold" in item and item["threshold"] is not None:
        st.session_state.ksd_thresholds.append(item["threshold"])
    st.session_state.total_windows += 1
    if item["triggered"]:
        st.session_state.trigger_count += 1
        new_ksd_triggered = True

scores = np.array(st.session_state.ksd_scores)
thr = (
    float(list(st.session_state.ksd_thresholds)[-1])
    if st.session_state.ksd_thresholds
    else None
)

auto_triggered = new_ksd_triggered
cv_triggered   = new_ksd_triggered or manual_trigger

# ── Debounce CV Activation ────────────────────────────────────────────────────
# Prevent the camera from spamming if a barrage of anomalies occurs (or if ghost
# keystrokes from background apps trigger KSD constantly).
# Reduced to 8 s to match the faster 25-key window cadence.
_CV_DEBOUNCE = 8.0
_time_since_last_cv = time.time() - st.session_state.last_cv_run_ts
if cv_triggered and not manual_trigger and _time_since_last_cv < _CV_DEBOUNCE:
    cv_triggered   = False
    auto_triggered = False

# Live state from background thread (display only — NOT used to fire CV)
live          = ksd_bg["live"]
press_now     = live["press_count"]
last_bg_score = live["last_score"]
last_bg_thr   = live["last_thr"]
last_bg_trig  = live["last_triggered"]
listener_active = live.get("listener_active", False)
listener_error  = live.get("listener_error")

# NOTE: We intentionally do NOT re-trigger cv_triggered from last_bg_trig here.
# CV is only triggered by a brand-new anomaly item arriving in the result deque
# (new_ksd_triggered) or by the manual button. Re-reading last_bg_trig on every
# rerun caused the camera to spam after the first anomaly was caught.

# ── Suspicion state management ────────────────────────────────────────────────
# Suspicion activates on a new KSD anomaly and clears either:
#   a) after CV has run (pass or fail) — checked via last_cv_run_ts advancing, or
#   b) after a 60-second hard cooldown with no new triggers
_SUSPICION_COOLDOWN = 60.0
_now = time.time()

if cv_triggered:
    st.session_state.suspicion_active  = True
    st.session_state.last_suspicion_ts = _now
elif st.session_state.suspicion_active:
    elapsed = _now - st.session_state.last_suspicion_ts
    # Clear suspicion once CV has actually run after this trigger
    cv_ran_after = st.session_state.last_cv_run_ts > st.session_state.last_suspicion_ts
    if cv_ran_after or elapsed > _SUSPICION_COOLDOWN:
        st.session_state.suspicion_active = False


# ══════════════════════════════════════════════════════════════════════════════
# CV Camera Trigger & Verification
# ══════════════════════════════════════════════════════════════════════════════

cv_result_this_tick = None

if cv_triggered and not st.session_state.cv_running:
    if profile_ok:
        st.session_state.cv_running = True
        with st.spinner("📷 Suspicious keystroke pattern — activating camera for identity check…"):
            cv_result_this_tick = _run_cv_verification()
        st.session_state.cv_running = False
        st.session_state.last_cv_run_ts = time.time()
    else:
        st.warning("⚠️ CV skipped — complete Enrolment to create a v2 gaze profile.")

    if cv_result_this_tick:
        with _LOG_LOCK:
            disk = _load_session_log()
            entry = {k: v for k, v in cv_result_this_tick.items()
                     if k != "snapshot_b64"}
            disk["cv_results"].append(entry)
            _save_session_log(disk)
        st.session_state.cv_results.append(cv_result_this_tick)
        st.session_state.last_cv = cv_result_this_tick

# Sync cv_results from disk if no new result this tick (picks up external writes)
if not cv_result_this_tick:
    try:
        disk_cv = _load_session_log().get("cv_results", [])
        if disk_cv:
            last_disk = disk_cv[-1]
            last_mem  = st.session_state.last_cv
            if (last_mem is None or
                    last_disk.get("timestamp", 0) > last_mem.get("timestamp", 0)):
                st.session_state.last_cv    = last_disk
                st.session_state.cv_results = disk_cv[-20:]
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# ── UI ────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("# 🔐 Two-Tier Continuous Authentication")
st.markdown("*Real-time keystroke dynamics + computer vision verification*")
st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# 🚨 Suspicion Alert Banner — shown when KSD anomaly is active
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.suspicion_active:
    _elapsed_susp = _now - st.session_state.last_suspicion_ts
    _last_cv_res  = st.session_state.last_cv
    _cv_after     = (
        _last_cv_res is not None
        and _last_cv_res.get("timestamp", 0) > st.session_state.last_suspicion_ts
    )

    if _cv_after and _last_cv_res.get("match"):
        # CV ran and passed — show green cleared banner
        st.markdown("""
        <div style="background:linear-gradient(135deg,#0d2b1a,#1a4a2e);border:2px solid #3fb950;
                    border-radius:14px;padding:18px 24px;margin-bottom:16px;
                    display:flex;align-items:center;gap:16px;
                    box-shadow:0 0 30px rgba(63,185,80,0.3);">
            <span style="font-size:2.4rem;">✅</span>
            <div>
                <div style="font-size:1.3rem;font-weight:800;color:#3fb950;
                            text-transform:uppercase;letter-spacing:0.05em;">Identity Confirmed</div>
                <div style="font-size:0.9rem;color:#7ee2a8;margin-top:4px;">
                    Camera verification passed — owner identity confirmed. Monitoring continues.</div>
            </div>
        </div>""", unsafe_allow_html=True)
    elif _cv_after and not _last_cv_res.get("match"):
        # CV ran and FAILED — intruder alert
        st.markdown("""
        <div class="suspicion-banner">
            <span class="suspicion-icon">🔴</span>
            <div class="suspicion-body">
                <div class="suspicion-title">🚨 INTRUDER ALERT — IDENTITY NOT VERIFIED</div>
                <div class="suspicion-sub">
                    Keystroke anomaly triggered camera check. <strong>Face verification FAILED.</strong><br>
                    The person at the keyboard does not match the enrolled owner.
                </div>
            </div>
        </div>""", unsafe_allow_html=True)
    elif st.session_state.cv_running:
        # Camera is actively running
        st.markdown("""
        <div class="suspicion-banner">
            <span class="suspicion-icon">📷</span>
            <div class="suspicion-body">
                <div class="suspicion-title">Verifying Identity…</div>
                <div class="suspicion-sub">
                    Suspicious typing pattern detected. Camera is active — analysing your face now.
                </div>
                <div class="camera-activating">🔴 CAMERA ACTIVE — Identity check in progress…</div>
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        # Suspicion raised, CV will fire next rerun (or is queued)
        st.markdown(f"""
        <div class="suspicion-banner">
            <span class="suspicion-icon">⚠️</span>
            <div class="suspicion-body">
                <div class="suspicion-title">Suspicious Keystroke Pattern Detected</div>
                <div class="suspicion-sub">
                    Anomalous typing rhythm observed (score: <strong>{last_bg_score:.4f}</strong>).
                    The camera will activate to verify the owner's identity.
                </div>
                <div class="camera-activating">📷 ACTIVATING CAMERA for identity check…</div>
            </div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ⌨️  Typing Pad
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("## ⌨️ Typing Pad")
st.markdown(
    "Type naturally in the box below. "
    "The system silently analyses your keystroke rhythm every "
    f"**{KSD_WINDOW_SIZE} keystrokes** and computes an anomaly score."
)

prompt = TYPING_PROMPTS[st.session_state.prompt_idx % len(TYPING_PROMPTS)]

pad_col, stat_col = st.columns([3, 1], gap="large")

with pad_col:
    # Prompt display
    st.markdown(
        f'<div class="typing-prompt">📝 {prompt}</div>',
        unsafe_allow_html=True,
    )
    
    # Real-time Live Typing Pad with instant keystroke counter & progress bar
    import streamlit.components.v1 as components
    components.html(
        f"""
        <!DOCTYPE html>
        <html>
        <head>
        <meta charset="utf-8">
        <style>
            * {{ box-sizing: border-box; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
            body {{ margin: 0; padding: 0; background: transparent; color: #e6edf3; }}
            .container {{
                background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
                border: 1px solid #30363d;
                border-radius: 12px;
                padding: 16px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            }}
            textarea {{
                width: 100%;
                height: 120px;
                background: #0d1117;
                border: 1px solid #30363d;
                border-radius: 8px;
                color: #e6edf3;
                padding: 12px;
                font-size: 15px;
                line-height: 1.5;
                resize: vertical;
                outline: none;
                transition: border-color 0.2s, box-shadow 0.2s;
            }}
            textarea:focus {{
                border-color: #58a6ff;
                box-shadow: 0 0 0 3px rgba(88, 166, 255, 0.2);
            }}
            .meter-wrapper {{
                margin-top: 14px;
            }}
            .meter-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                font-size: 13px;
                font-weight: 600;
                margin-bottom: 6px;
            }}
            .meter-label {{ color: #8b949e; }}
            .meter-count {{ color: #58a6ff; font-family: monospace; font-size: 14px; }}
            .meter-bg {{
                height: 10px;
                background: #21262d;
                border-radius: 5px;
                overflow: hidden;
                position: relative;
            }}
            .meter-fill {{
                height: 100%;
                width: 0%;
                background: linear-gradient(90deg, #58a6ff, #3fb950);
                border-radius: 5px;
                transition: width 0.08s ease-out, background 0.2s;
            }}
            .meter-status {{
                margin-top: 6px;
                font-size: 12px;
                color: #8b949e;
                font-style: italic;
            }}
            .badge-complete {{
                color: #3fb950;
                font-weight: bold;
                animation: pulse 1s infinite alternate;
            }}
            @keyframes pulse {{
                from {{ opacity: 0.7; transform: scale(1); }}
                to {{ opacity: 1; transform: scale(1.02); }}
            }}
        </style>
        </head>
        <body>
        <div class="container">
            <textarea id="livePad" placeholder="Type here — the meter fills up live with every keypress..."></textarea>
            <div class="meter-wrapper">
                <div class="meter-header">
                    <span class="meter-label">⚡ Live Keystroke Meter (Window: {KSD_WINDOW_SIZE})</span>
                    <span class="meter-count" id="countDisplay">0 / {KSD_WINDOW_SIZE} keys</span>
                </div>
                <div class="meter-bg">
                    <div class="meter-fill" id="meterFill"></div>
                </div>
                <div class="meter-status" id="statusDisplay">⏳ Waiting for keystrokes…</div>
            </div>
        </div>

        <script>
            const pad = document.getElementById('livePad');
            const fill = document.getElementById('meterFill');
            const countDisplay = document.getElementById('countDisplay');
            const statusDisplay = document.getElementById('statusDisplay');
            const windowSize = {KSD_WINDOW_SIZE};
            let totalKeyStrokes = 0;

            pad.addEventListener('keydown', function(e) {{
                // Ignore modifier-only keys
                if (['Shift', 'Control', 'Alt', 'Meta', 'CapsLock'].includes(e.key)) return;
                
                totalKeyStrokes++;
                const currentInWindow = totalKeyStrokes % windowSize;
                const completedWindows = Math.floor(totalKeyStrokes / windowSize);
                const displayCount = (currentInWindow === 0 && totalKeyStrokes > 0) ? windowSize : currentInWindow;
                const pct = (displayCount / windowSize) * 100;
                
                fill.style.width = pct + '%';
                countDisplay.textContent = displayCount + ' / ' + windowSize + ' keys (Total: ' + totalKeyStrokes + ')';
                
                if (pct < 40) {{
                    fill.style.background = 'linear-gradient(90deg, #58a6ff, #79c0ff)';
                    statusDisplay.innerHTML = '🟡 Accumulating keystrokes...';
                }} else if (pct < 80) {{
                    fill.style.background = 'linear-gradient(90deg, #58a6ff, #d29922)';
                    statusDisplay.innerHTML = '🟠 Good pace — almost at ' + windowSize + '!';
                }} else if (pct < 100) {{
                    fill.style.background = 'linear-gradient(90deg, #d29922, #3fb950)';
                    statusDisplay.innerHTML = '🔥 ' + (windowSize - displayCount) + ' keys left to score window!';
                }} else {{
                    fill.style.background = 'linear-gradient(90deg, #3fb950, #2ea043)';
                    statusDisplay.innerHTML = '<span class="badge-complete">✅ Window of ' + windowSize + ' completed & analysed!</span>';
                }}
            }});
        </script>
        </body>
        </html>
        """,
        height=250,
    )
    st.caption(
        "💡 Tip: Type in the box above or anywhere on your PC. "
        f"The meter fills up dynamically with every keypress to show your progress toward {KSD_WINDOW_SIZE} keys."
    )

with stat_col:
    # ── Live progress ───────────────────────────────────────────────────────
    st.markdown("#### 🔍 Background Window")
    if listener_error:
        st.error(f"Keyboard listener error: {listener_error}")
    elif not listener_active:
        st.warning("Keyboard listener is starting…")

    pct  = min(press_now / KSD_WINDOW_SIZE, 1.0)
    label_txt = f"{press_now} / {KSD_WINDOW_SIZE} keystrokes"
    st.progress(pct, text=label_txt)

    if press_now == 0:
        st.markdown("*⏳ Waiting for keystrokes…*")
    elif press_now < KSD_WINDOW_SIZE // 2:
        st.markdown("🟡 *Accumulating…*")
    else:
        st.markdown("🟠 *Almost there — keep typing!*")

    # ── Last window result ──────────────────────────────────────────────────
    if last_bg_score is not None:
        st.markdown("#### 📊 Last Window")
        if last_bg_trig:
            cls       = "score-value-bad"
            card_cls  = "score-high"
        elif last_bg_score > 0.55:
            cls       = "score-value-warn"
            card_cls  = "score-card"
        else:
            cls       = "score-value-ok"
            card_cls  = "score-card"

        if last_bg_trig:
            anomaly_html = '<div class="anomaly-badge">🚨 ANOMALY — Camera activating</div>'
        elif last_bg_score > 0.55:
            anomaly_html = '<div style="color:#f0883e;font-size:0.85rem;margin-top:6px;">⚠️ Elevated — monitoring…</div>'
        else:
            anomaly_html = ""

        thr_txt = f"thr: {last_bg_thr}" if last_bg_thr else "thr: —"
        st.markdown(
            f'<div class="{card_cls}">'
            f'  <div class="score-label">Anomaly Score</div>'
            f'  <div class="{cls}">{last_bg_score:.4f}</div>'
            f'  <div class="score-thr">{thr_txt}</div>'
            f'  {anomaly_html}'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# Top KPIs
# ══════════════════════════════════════════════════════════════════════════════

col1, col2, col3, col4, col5 = st.columns(5)

display_score = last_bg_score if last_bg_score is not None else (
    float(list(st.session_state.ksd_scores)[-1]) if st.session_state.ksd_scores else 0.0
)
display_thr = last_bg_thr if last_bg_thr is not None else (thr if thr else None)

with col1:
    st.metric("Latest KSD Score", f"{display_score:.4f}")
with col2:
    st.metric("Dynamic Threshold", f"{display_thr:.4f}" if display_thr else "—")
with col3:
    st.metric("CV Triggers",
              st.session_state.trigger_count,
              delta=1 if auto_triggered else None)
with col4:
    st.metric("Windows Processed", st.session_state.total_windows)
with col5:
    last = st.session_state.last_cv
    if last:
        badge = "PASS ✅" if last.get("match") else "FAIL ❌"
        st.metric("Last CV Result", badge,
                  delta=f"combined={last.get('combined', '—')}")
    else:
        st.metric("Last CV Result", "—")

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# KSD Score Chart
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("## 📊 Tier 1 — Keystroke Dynamics Monitor")

if len(st.session_state.ksd_scores) > 0:
    try:
        import plotly.graph_objects as go

        s_arr = list(st.session_state.ksd_scores)
        t_arr = list(st.session_state.ksd_thresholds)
        x_s   = list(range(len(s_arr)))
        x_t   = list(range(len(s_arr) - len(t_arr), len(s_arr)))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_s, y=s_arr,
            mode="lines+markers",
            name="Anomaly Score",
            line=dict(color="#58a6ff", width=2),
            marker=dict(size=5),
            fill="tozeroy",
            fillcolor="rgba(88,166,255,0.08)",
        ))
        if t_arr:
            fig.add_trace(go.Scatter(
                x=x_t, y=t_arr,
                mode="lines",
                name="Dynamic Threshold",
                line=dict(color="#f0883e", width=2, dash="dash"),
            ))
            breach_pairs = [(x_t[i], s) for i, (s, th) in
                            enumerate(zip(s_arr[-len(t_arr):], t_arr)) if s > th]
            breach_x = [p[0] for p in breach_pairs]
            breach_y = [p[1] for p in breach_pairs]
            if breach_x:
                fig.add_trace(go.Scatter(
                    x=breach_x, y=breach_y,
                    mode="markers",
                    name="Threshold Breach 🚨",
                    marker=dict(color="#f85149", size=12, symbol="x"),
                ))

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#161b22",
            plot_bgcolor="#0d1117",
            font=dict(family="Inter", color="#c9d1d9"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=10, r=10, t=10, b=10),
            height=300,
            yaxis=dict(title="Anomaly Score", rangemode="tozero"),
            xaxis=dict(title="Window Index"),
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        df_scores = pd.DataFrame({"Anomaly Score": list(st.session_state.ksd_scores)})
        st.line_chart(df_scores, color="#58a6ff")


# ══════════════════════════════════════════════════════════════════════════════
# CV Camera Alert Panel
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("## 👁 Tier 2 — Computer Vision Verification")

if cv_result_this_tick:
    err     = cv_result_this_tick.get("error")
    matched = cv_result_this_tick.get("match", False)

    if err:
        st.markdown(f"""
        <div class="alert-fail">
            <div class="alert-title">⚠️ CV CAPTURE ERROR</div>
            <div class="alert-detail">{err}</div>
        </div>""", unsafe_allow_html=True)
    elif matched:
        st.markdown(f"""
        <div class="alert-pass">
            <div class="alert-title">✅ IDENTITY CONFIRMED</div>
            <div class="alert-detail">
                Combined: {cv_result_this_tick['combined']} &nbsp;|&nbsp;
                Gaze dist: {cv_result_this_tick['gaze_dist']} &nbsp;|&nbsp;
                Geom dist: {cv_result_this_tick['geom_dist']} &nbsp;|&nbsp;
                Threshold: {cv_result_this_tick.get('threshold', 3.5)}
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="alert-fail">
            <div class="alert-title">🚨 VERIFICATION FAILED — POSSIBLE IMPOSTOR</div>
            <div class="alert-detail">
                Combined: {cv_result_this_tick['combined']} &nbsp;|&nbsp;
                Gaze dist: {cv_result_this_tick['gaze_dist']} &nbsp;|&nbsp;
                Geom dist: {cv_result_this_tick['geom_dist']} &nbsp;|&nbsp;
                Threshold: {cv_result_this_tick.get('threshold', 3.5)}
            </div>
        </div>""", unsafe_allow_html=True)

cv_col1, cv_col2 = st.columns([2, 1])

with cv_col1:
    if st.session_state.cv_results:
        df_cv = pd.DataFrame(st.session_state.cv_results[-20:])
        df_cv["time"]   = pd.to_datetime(df_cv["timestamp"], unit="s")
        df_cv["status"] = df_cv["match"].map({True: "✅ PASS", False: "❌ FAIL"})
        cols_want  = ["time", "status", "gaze_dist", "geom_dist", "combined"]
        cols_avail = [c for c in cols_want if c in df_cv.columns]
        st.dataframe(
            df_cv[cols_avail].rename(columns={
                "time":      "Timestamp",
                "status":    "Decision",
                "gaze_dist": "Gaze Dist",
                "geom_dist": "Geom Dist",
                "combined":  "Combined",
            }),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No CV events yet. CV fires automatically when KSD exceeds its threshold.")

with cv_col2:
    st.markdown("### Last Verification")
    if last := st.session_state.last_cv:
        badge_cls = "badge-ok" if last.get("match") else "badge-fail"
        badge_txt = "IDENTITY CONFIRMED" if last.get("match") else "VERIFICATION FAILED"
        st.markdown(f'<div class="{badge_cls}">{badge_txt}</div>',
                    unsafe_allow_html=True)
        st.markdown("")
        for label, key in [
            ("Gaze Distance", "gaze_dist"),
            ("Geom Distance", "geom_dist"),
            ("Combined",      "combined"),
        ]:
            if last.get(key) is not None:
                st.markdown(f"**{label}:** `{last[key]}`")
        st.markdown(f"**Threshold:** `{last.get('threshold', 3.5)}`")
        if ts := last.get("timestamp"):
            st.markdown(f"**Time:** {datetime.fromtimestamp(ts).strftime('%H:%M:%S')}")
        if snap := last.get("snapshot_b64"):
            st.markdown("**Captured Frame:**")
            st.markdown(
                f'<img src="data:image/jpeg;base64,{snap}" '
                f'style="width:100%;border-radius:8px;'
                f'border:1px solid #30363d;margin-top:8px;" />',
                unsafe_allow_html=True,
            )
    else:
        st.markdown('<div class="badge-warn">AWAITING TRIGGER</div>',
                    unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# KSD Statistics
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("## 📈 KSD Score Statistics")

if len(scores) >= 2:
    stat_cols = st.columns(4)
    stat_cols[0].metric("Mean Score", f"{scores.mean():.4f}")
    stat_cols[1].metric("Std Dev",    f"{scores.std():.4f}")
    stat_cols[2].metric("Min Score",  f"{scores.min():.4f}")
    stat_cols[3].metric("Max Score",  f"{scores.max():.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# KSD Feature Breakdown  (v2 12-D features)
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("## ⌨️ KSD Feature Breakdown")
st.markdown(
    "Live per-window breakdown of the **12-D feature vector** scored by the IsolationForest. "
    "Values update after every 50-keystroke window."
)

_fb = ksd_bg["live"]   # re-read the live dict
_typing_speed   = _fb.get("last_typing_speed")
_backspace_rate = _fb.get("last_backspace_rate")
_digraph_var    = _fb.get("last_digraph_var")
_dwell_mean     = _fb.get("last_dwell_mean")
_flight_mean    = _fb.get("last_flight_mean")

def _feat_val(v, fmt="{}", suffix=""):
    return f"{fmt.format(v)}{suffix}" if v is not None else "—"

fb_col1, fb_col2, fb_col3, fb_col4, fb_col5 = st.columns(5)

with fb_col1:
    st.markdown("""
    <div style="background:linear-gradient(135deg,#161b22,#21262d);border:1px solid #30363d;
                border-radius:12px;padding:16px 18px;text-align:center;">
        <div style="color:#8b949e;font-size:0.72rem;text-transform:uppercase;
                    letter-spacing:0.06em;margin-bottom:6px;">Dwell Time (mean)</div>
        <div style="color:#58a6ff;font-size:1.8rem;font-weight:700;">"""
    + _feat_val(_dwell_mean, "{:.1f}", " ms") +
    """</div>
        <div style="color:#8b949e;font-size:0.75rem;margin-top:4px;">press → release</div>
    </div>""", unsafe_allow_html=True)

with fb_col2:
    st.markdown("""
    <div style="background:linear-gradient(135deg,#161b22,#21262d);border:1px solid #30363d;
                border-radius:12px;padding:16px 18px;text-align:center;">
        <div style="color:#8b949e;font-size:0.72rem;text-transform:uppercase;
                    letter-spacing:0.06em;margin-bottom:6px;">Flight Time (mean)</div>
        <div style="color:#a5d6a7;font-size:1.8rem;font-weight:700;">"""
    + _feat_val(_flight_mean, "{:.1f}", " ms") +
    """</div>
        <div style="color:#8b949e;font-size:0.75rem;margin-top:4px;">release → next press</div>
    </div>""", unsafe_allow_html=True)

with fb_col3:
    st.markdown("""
    <div style="background:linear-gradient(135deg,#161b22,#21262d);border:1px solid #30363d;
                border-radius:12px;padding:16px 18px;text-align:center;">
        <div style="color:#8b949e;font-size:0.72rem;text-transform:uppercase;
                    letter-spacing:0.06em;margin-bottom:6px;">Typing Speed</div>
        <div style="color:#f0883e;font-size:1.8rem;font-weight:700;">"""
    + _feat_val(_typing_speed, "{:.1f}", " kps") +
    """</div>
        <div style="color:#8b949e;font-size:0.75rem;margin-top:4px;">keystrokes / sec</div>
    </div>""", unsafe_allow_html=True)

with fb_col4:
    _bs_pct = f"{_backspace_rate * 100:.1f}%" if _backspace_rate is not None else "—"
    _bs_color = "#f85149" if (_backspace_rate or 0) > 0.10 else "#e6edf3"
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#161b22,#21262d);border:1px solid #30363d;
                border-radius:12px;padding:16px 18px;text-align:center;">
        <div style="color:#8b949e;font-size:0.72rem;text-transform:uppercase;
                    letter-spacing:0.06em;margin-bottom:6px;">Backspace Rate</div>
        <div style="color:{_bs_color};font-size:1.8rem;font-weight:700;">{_bs_pct}</div>
        <div style="color:#8b949e;font-size:0.75rem;margin-top:4px;">removed in v2</div>
    </div>""", unsafe_allow_html=True)

with fb_col5:
    st.markdown("""
    <div style="background:linear-gradient(135deg,#161b22,#21262d);border:1px solid #30363d;
                border-radius:12px;padding:16px 18px;text-align:center;">
        <div style="color:#8b949e;font-size:0.72rem;text-transform:uppercase;
                    letter-spacing:0.06em;margin-bottom:6px;">Burst Ratio</div>
        <div style="color:#d2a8ff;font-size:1.8rem;font-weight:700;">"""
    + _feat_val(_digraph_var, "{:.2f}") +
    """</div>
        <div style="color:#8b949e;font-size:0.75rem;margin-top:4px;">flight std / mean</div>
    </div>""", unsafe_allow_html=True)

st.markdown("")
st.caption(
    "**Dwell** = time key is held down &nbsp;|&nbsp; "
    "**Flight** = gap between release and next press &nbsp;|&nbsp; "
    "**Speed** = keystrokes per second over the 50-key window &nbsp;|&nbsp; "
    "**Backspace Rate** = removed from v2 and shown as 0.0% &nbsp;|&nbsp; "
    "**Burst Ratio** = flight-time standard deviation divided by mean"
)


# ══════════════════════════════════════════════════════════════════════════════
# Persistent log info
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
with st.expander("💾 Persistent Session Log"):
    st.markdown(f"Log path: `{SESSION_LOG}`")
    loaded = _load_session_log()
    st.markdown(f"- **Stored KSD windows:** {len(loaded['ksd_scores'])}")
    st.markdown(f"- **Stored CV events:** {len(loaded['cv_results'])}")
    st.markdown(f"- **Total CV triggers:** {loaded['trigger_count']}")
    st.caption(
        "KSD scores and CV results survive dashboard restarts. "
        "Snapshots are kept in-memory only (not persisted to disk)."
    )

# ══════════════════════════════════════════════════════════════════════════════
# Auto-refresh
# ══════════════════════════════════════════════════════════════════════════════

if auto_refresh or refresh_btn:
    time.sleep(2.0)
    st.rerun()
