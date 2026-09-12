"""
enrol.py — One-time enrolment script.

Runs two concurrent tasks:
  1. Keystroke capture → trains IsolationForest + One-Class SVM → saves model.joblib + ocsvm.joblib
  2. Webcam FaceMesh capture (30 s) → saves gaze_profile.npy
"""

import os
import time
import queue
import threading
import logging

import numpy as np
import joblib
import cv2
from sklearn.svm import OneClassSVM
import mediapipe as mp
from pynput import keyboard
from sklearn.ensemble import IsolationForest

from ksd.features import compute_features_from_raw

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR      = os.path.join(os.path.dirname(__file__), "data")
MODEL_PATH    = os.path.join(DATA_DIR, "model.joblib")
OCSVM_PATH    = os.path.join(DATA_DIR, "ocsvm.joblib")
PROFILE_PATH  = os.path.join(DATA_DIR, "gaze_profile.npy")

os.makedirs(DATA_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("enrol")

# ── Keystroke enrolment ────────────────────────────────────────────────────────
WINDOW_SIZE = 80          # keystrokes per training window (must match KSD_WINDOW_SIZE in main.py)
MIN_WINDOWS = 15          # minimum windows required for training

def _enrol_keystrokes():
    """Capture keystrokes (press + release), build feature windows, train IsolationForest."""
    raw_events: list[tuple[str, str, float]] = []  # (event_type, key_char, timestamp)
    feature_windows: list[np.ndarray] = []
    done = threading.Event()

    def on_press(key):
        try:
            ch = key.char if hasattr(key, "char") and key.char else str(key)
        except Exception:
            ch = str(key)
        raw_events.append(("press", ch, time.perf_counter()))

    def on_release(key):
        try:
            ch = key.char if hasattr(key, "char") and key.char else str(key)
        except Exception:
            ch = str(key)
        raw_events.append(("release", ch, time.perf_counter()))
        if key == keyboard.Key.esc:
            done.set()
            return False        # stop listener

    log.info("⌨  Keystroke enrolment started. "
             "Type naturally (≥ %d keystrokes). Press Esc when done.",
             WINDOW_SIZE * MIN_WINDOWS)

    with keyboard.Listener(on_press=on_press, on_release=on_release):
        done.wait()

    # Slice raw_events into non-overlapping windows of WINDOW_SIZE presses each
    press_indices = [i for i, ev in enumerate(raw_events) if ev[0] == "press"]
    for w_start in range(0, len(press_indices) - WINDOW_SIZE + 1, WINDOW_SIZE):
        # Take all raw events between the first and last press of this window
        idx_first = press_indices[w_start]
        idx_last  = press_indices[w_start + WINDOW_SIZE - 1]
        window_events = raw_events[idx_first : idx_last + 1]
        vec = compute_features_from_raw(window_events)
        if vec is not None:
            feature_windows.append(vec)

    if len(feature_windows) < MIN_WINDOWS:
        log.warning("Only %d windows captured; need %d. "
                    "Type more next time for a better model.",
                    len(feature_windows), MIN_WINDOWS)
    else:
        X = np.array(feature_windows)

        # ── IsolationForest ───────────────────────────────────────────────────
        clf = IsolationForest(n_estimators=200, contamination=0.05,
                              random_state=42)
        clf.fit(X)
        joblib.dump(clf, MODEL_PATH)
        log.info("✅ IsolationForest saved → %s  (%d windows)", MODEL_PATH,
                 len(feature_windows))

        # ── One-Class SVM ─────────────────────────────────────────────────────
        ocsvm = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
        ocsvm.fit(X)
        joblib.dump(ocsvm, OCSVM_PATH)
        log.info("✅ One-Class SVM saved → %s  (%d windows)", OCSVM_PATH,
                 len(feature_windows))



# ── Face enrolment ─────────────────────────────────────────────────────────────
ENROL_SECONDS = 30
ENROL_FPS     = 5   # slower capture — embedding extraction is expensive


def _enrol_face():
    """
    Capture ENROL_SECONDS of webcam video, extract Facenet512 embeddings for
    each frame using DeepFace, and save the mean embedding to gaze_profile.npy.

    The webcam window shows a live countdown.  Press Esc to abort early.
    """
    from cv.face import extract_embedding, FACE_MODEL

    log.info("📷 Face enrolment starting (%d s @ %d fps, model=%s).",
             ENROL_SECONDS, ENROL_FPS, FACE_MODEL)
    log.info("   Look naturally at the screen.  Press Esc to abort early.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        log.error("Cannot open webcam — face enrolment skipped.")
        return

    embeddings = []
    deadline       = time.time() + ENROL_SECONDS
    frame_interval = 1.0 / ENROL_FPS

    try:
        next_frame = time.time()
        while time.time() < deadline:
            now = time.time()
            if now < next_frame:
                time.sleep(max(0.0, next_frame - now))
            next_frame += frame_interval

            ok, frame = cap.read()
            if not ok:
                continue

            # Extract embedding (runs DeepFace internally)
            emb = extract_embedding(frame)
            if emb is not None:
                embeddings.append(emb)

            # Live preview with countdown and embedding count
            remaining = max(0, int(deadline - time.time()))
            label = (f"Enrolling face: {remaining}s  "
                     f"({len(embeddings)} frames captured)")
            cv2.putText(frame, label, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
            cv2.imshow("Face Enrolment", frame)
            if cv2.waitKey(1) & 0xFF == 27:   # Esc aborts
                log.info("Face enrolment aborted by user.")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if len(embeddings) < 5:
        log.error("Too few face embeddings (%d) — check that the webcam can see "
                  "your face and run enrol.py again.", len(embeddings))
        return

    emb_arr  = np.array(embeddings, dtype=np.float64)   # (N, 512)
    emb_mean = emb_arr.mean(axis=0)                     # (512,)
    emb_std  = emb_arr.std(axis=0)

    log.info("✅ Face embedding mean computed over %d frames.", len(embeddings))
    log.info("   Embedding norm: %.4f", float(np.linalg.norm(emb_mean)))

    profile = {
        "face_mean":       emb_mean,
        "face_std":        emb_std,
        "face_embeddings": emb_arr,   # keep all frames for future offline analysis
        "model":           FACE_MODEL,
        "n_frames":        len(embeddings),
    }
    np.save(PROFILE_PATH, profile)
    log.info("✅ Face profile saved → %s  (%d embeddings, model=%s)",
             PROFILE_PATH, len(embeddings), FACE_MODEL)


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    log.info("=== Two-Tier Auth — Enrolment ===")
    log.info("Running keystroke and face enrolment CONCURRENTLY.")
    log.info("• Face  : runs for %d seconds automatically (webcam window).", ENROL_SECONDS)
    log.info("• Keyboard: type naturally then press Esc when finished.")

    t_face = threading.Thread(target=_enrol_face, name="face-enrol", daemon=False)
    t_keys = threading.Thread(target=_enrol_keystrokes, name="key-enrol", daemon=False)

    t_face.start()
    t_keys.start()

    t_face.join()
    t_keys.join()

    log.info("=== Enrolment complete. Run main.py to start authentication. ===")


if __name__ == "__main__":
    main()

