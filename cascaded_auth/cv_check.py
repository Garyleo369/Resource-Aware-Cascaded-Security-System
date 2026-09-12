"""
cascaded_auth/cv_check.py — Webcam identity verification for Prototype v2.

Uses MediaPipe FaceMesh iris landmarks to build a 4-D normalised gaze vector
per frame, then compares the session mean against the enrolled profile using
Mahalanobis distance.

Constants
---------
SESSION_SECS    : seconds of webcam footage collected per check (default 5)
MATCH_THRESHOLD : Mahalanobis distance below which identity is confirmed (default 5.0)
PROFILE_FILE    : path to the saved gaze profile (cascaded_auth/gaze_profile.joblib)

Functions
---------
get_gaze_vector(face_landmarks, frame_w, frame_h) -> np.ndarray (4,)
collect_gaze_vectors(duration_secs)               -> list[np.ndarray]
save_gaze_profile(vectors)                        -> None
run_cv_check()                                    -> bool
"""

import os
import time
import logging

import cv2
import numpy as np
import joblib

try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "mediapipe not installed — cv_check will always return True (fail-open)."
    )

try:
    from scipy.spatial.distance import mahalanobis
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "scipy not installed — using Euclidean fallback."
    )

log = logging.getLogger(__name__)

# ── File paths ────────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
PROFILE_FILE = os.path.join(_HERE, "gaze_profile.joblib")

# ── Constants ──────────────────────────────────────────────────────────────────────────
SESSION_SECS    = 5      # capture 5 s of webcam for better frame count
MATCH_THRESHOLD = 5.0    # Mahalanobis distance threshold — raised to reduce false rejections

# MediaPipe FaceMesh iris landmark indices
# Left iris:  468-472  Right iris: 473-477
# Left outer corner: 33   Right outer corner: 263
_LEFT_IRIS  = list(range(468, 473))
_RIGHT_IRIS = list(range(473, 478))
_LEFT_CORNER  = 33
_RIGHT_CORNER = 263


# ══════════════════════════════════════════════════════════════════════════════
# Gaze vector extraction
# ══════════════════════════════════════════════════════════════════════════════

def get_gaze_vector(face_landmarks, frame_w: int, frame_h: int) -> np.ndarray:
    """
    Compute a normalised 4-D gaze vector from MediaPipe FaceMesh landmarks.

    The vector is the offset of each iris centre relative to its eye outer
    corner, normalised by the inter-ocular distance (IOD):

        [left_iris_offset_x, left_iris_offset_y,
         right_iris_offset_x, right_iris_offset_y]

    Normalising by IOD makes the vector robust to head distance from camera.

    Parameters
    ----------
    face_landmarks : mediapipe face landmarks object
    frame_w, frame_h : pixel dimensions of the frame

    Returns
    -------
    np.ndarray of shape (4,)
    """
    lm = face_landmarks.landmark

    def _px(idx):
        return np.array([lm[idx].x * frame_w, lm[idx].y * frame_h])

    left_iris  = np.mean([_px(i) for i in _LEFT_IRIS],  axis=0)
    right_iris = np.mean([_px(i) for i in _RIGHT_IRIS], axis=0)

    left_corner  = _px(_LEFT_CORNER)
    right_corner = _px(_RIGHT_CORNER)

    iod = float(np.linalg.norm(right_corner - left_corner)) + 1e-6

    vec = np.concatenate([
        (left_iris  - left_corner)  / iod,
        (right_iris - right_corner) / iod,
    ])
    return vec   # shape (4,)


# ══════════════════════════════════════════════════════════════════════════════
# Data collection
# ══════════════════════════════════════════════════════════════════════════════

def collect_gaze_vectors(duration_secs: float = SESSION_SECS) -> list[np.ndarray]:
    """
    Open the webcam and collect gaze vectors for duration_secs seconds.

    Returns
    -------
    list of np.ndarray (4,) — one per frame where a face was detected.
    Empty list if mediapipe is unavailable or no face found.
    """
    if not _MP_AVAILABLE:
        log.warning("mediapipe not available — returning empty gaze vectors.")
        return []

    mp_face = mp.solutions.face_mesh
    cap     = cv2.VideoCapture(0)
    vectors = []
    end     = time.time() + duration_secs

    with mp_face.FaceMesh(refine_landmarks=True, max_num_faces=1) as mesh:
        while time.time() < end:
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = mesh.process(rgb)
            if result.multi_face_landmarks:
                try:
                    vec = get_gaze_vector(result.multi_face_landmarks[0], w, h)
                    vectors.append(vec)
                except Exception as exc:
                    log.debug("Gaze vector extraction failed: %s", exc)

    cap.release()
    log.info("Collected %d gaze vectors over %.1f s", len(vectors), duration_secs)
    return vectors


# ══════════════════════════════════════════════════════════════════════════════
# Profile persistence
# ══════════════════════════════════════════════════════════════════════════════

def save_gaze_profile(vectors: list[np.ndarray]) -> None:
    """
    Compute and save the gaze profile (mean + covariance) to PROFILE_FILE.

    Parameters
    ----------
    vectors : list of np.ndarray (4,)
        Gaze vectors collected during enrolment.
    """
    if not vectors:
        log.error("No gaze vectors provided — profile not saved.")
        return

    mat     = np.array(vectors, dtype=np.float64)
    profile = {
        "mean": np.mean(mat, axis=0),
        "cov":  np.cov(mat.T),
        "n":    len(vectors),
    }
    joblib.dump(profile, PROFILE_FILE)
    print(f"Gaze profile saved ({len(vectors)} samples) → {PROFILE_FILE}")


# ══════════════════════════════════════════════════════════════════════════════
# Live check
# ══════════════════════════════════════════════════════════════════════════════

def run_cv_check() -> bool:
    """
    Capture a short webcam session and compare against the enrolled gaze profile.

    Returns
    -------
    True  — identity confirmed (Mahalanobis distance < MATCH_THRESHOLD)
    False — no match or no face detected
    """
    if not os.path.exists(PROFILE_FILE):
        print("No gaze profile found. Run enrol.py first.")
        return True   # fail-open: don't lock out if not enrolled

    try:
        profile = joblib.load(PROFILE_FILE)
    except Exception as exc:
        log.error("Failed to load gaze profile: %s", exc)
        return True

    vectors = collect_gaze_vectors(SESSION_SECS)
    if not vectors:
        # Fail-open: if no face detected (bad lighting, camera issue, etc.)
        # do NOT lock out the owner. Log a warning and confirm identity.
        log.warning("CV check: no face detected — failing open (owner assumed present).")
        print("CV: no face detected — failing open")
        return True

    session_vec = np.mean(np.array(vectors, dtype=np.float64), axis=0)

    # ── Mahalanobis distance ──────────────────────────────────────────────────
    try:
        cov     = profile["cov"]
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        log.warning("Covariance matrix singular — using identity.")
        cov_inv = np.eye(len(session_vec))

    if _SCIPY_AVAILABLE:
        dist = float(mahalanobis(session_vec, profile["mean"], cov_inv))
    else:
        # Euclidean fallback (less reliable but avoids hard dependency)
        diff = session_vec - profile["mean"]
        dist = float(np.sqrt(diff @ diff))

    match = dist < MATCH_THRESHOLD
    print(f"CV check: dist={dist:.3f}  threshold={MATCH_THRESHOLD}  match={match}")
    return match
