"""
cascaded_auth/cv_check.py — Deep Face Identity Verification for Prototype v2.

Upgraded from iris-gaze tracking to deep facial recognition using InsightFace
ArcFace (512-dimensional identity embeddings).

How it works:
-------------
1. On verification trigger, the webcam captures frames across a short session (3 s).
2. For each frame, InsightFace detects the face and extracts a 512-D ArcFace embedding.
3. The session embeddings are averaged to form a session centroid.
4. Cosine distance between session centroid and enrolled face embedding is computed:
     distance < MATCH_THRESHOLD (0.50)  → Identity Confirmed (PASS)
     distance >= MATCH_THRESHOLD (0.50) → Impostor Detected (FAIL)
5. If no face is detected (e.g. camera covered or user stepped away), verification
   fails closed to prevent unauthorized access.

Backward Compatibility:
-----------------------
Maintains all legacy function aliases (collect_gaze_vectors, save_gaze_profile,
get_gaze_vector, PROFILE_FILE) so existing scripts and callers continue to work.
"""

from __future__ import annotations

import os
import io
import time
import base64
import logging
from typing import Optional, List, Tuple

import cv2
import numpy as np
import joblib

log = logging.getLogger(__name__)

# ── File paths ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
FACE_PROFILE_FILE   = os.path.join(_HERE, "face_profile.joblib")
GAZE_PROFILE_FILE   = os.path.join(_HERE, "gaze_profile.joblib")
DATA_PROFILE_FILE   = os.path.join(os.path.dirname(_HERE), "data", "gaze_profile.npy")

# Legacy symbol for existing callers
PROFILE_FILE        = FACE_PROFILE_FILE

# ── Verification Constants ───────────────────────────────────────────────────
SESSION_SECS: float    = 3.0    # Duration of webcam check
MATCH_THRESHOLD: float = 0.50   # ArcFace Cosine distance threshold (lower = stricter)
FACE_MODEL: str        = "ArcFace (InsightFace buffalo_sc 512-D)"

# ── Lazy InsightFace loader ───────────────────────────────────────────────────
_app = None


def _get_insightface_app():
    global _app
    if _app is None:
        try:
            import insightface
            app = insightface.app.FaceAnalysis(
                name="buffalo_sc",
                providers=["CPUExecutionProvider"],
            )
            app.prepare(ctx_id=0, det_size=(640, 640))
            _app = app
            log.info("InsightFace FaceAnalysis ready (%s).", FACE_MODEL)
        except Exception as exc:
            log.error("Failed to initialize InsightFace: %s", exc)
            _app = None
    return _app


# ── Core Face Embedding Extraction ───────────────────────────────────────────

def extract_face_embedding(frame_bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    Detect the most prominent face in a BGR frame and return its 512-D ArcFace embedding.
    Returns None if no face is detected or if an error occurs.
    """
    app = _get_insightface_app()
    if app is None:
        return None
    try:
        faces = app.get(frame_bgr)
        if not faces:
            return None
        # Choose the face with the largest bounding-box area
        best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        emb = np.array(best.embedding, dtype=np.float64)
        if np.linalg.norm(emb) < 1e-6:
            return None
        return emb
    except Exception as exc:
        log.debug("extract_face_embedding error: %s", exc)
        return None


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Cosine distance in [0, 2].
    0.0 = identical direction vectors, 1.0 = orthogonal, 2.0 = opposite.
    """
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 2.0
    return float(1.0 - np.dot(a / na, b / nb))


def _frame_to_b64(frame: np.ndarray) -> Optional[str]:
    """Convert BGR frame to Base64 JPEG string for Streamlit dashboard rendering."""
    if frame is None:
        return None
    try:
        from PIL import Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return None


# ── Face Capture & Session Verification ──────────────────────────────────────

def collect_face_embeddings(
    duration_secs: float = SESSION_SECS,
    fps: float = 6.0,
    camera_index: int = 0,
) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
    """
    Open the webcam and capture face embeddings over `duration_secs`.
    
    Returns
    -------
    (embeddings, best_frame):
      - embeddings: list of 512-D numpy vectors
      - best_frame: raw BGR frame from the first detected face (for dashboard snapshot)
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        log.error("Cannot open webcam index %d", camera_index)
        return [], None

    embeddings: List[np.ndarray] = []
    best_frame: Optional[np.ndarray] = None
    deadline = time.time() + duration_secs
    frame_interval = 1.0 / max(fps, 1.0)
    next_frame_time = time.time()

    try:
        # Warmup sensor
        for _ in range(3):
            cap.read()

        while time.time() < deadline:
            now = time.time()
            if now < next_frame_time:
                time.sleep(max(0.0, next_frame_time - now))
            next_frame_time += frame_interval

            ok, frame = cap.read()
            if not ok:
                continue

            emb = extract_face_embedding(frame)
            if emb is not None:
                embeddings.append(emb)
                if best_frame is None:
                    best_frame = frame.copy()

    finally:
        cap.release()

    log.info("Collected %d face embeddings over %.1f s", len(embeddings), duration_secs)
    return embeddings, best_frame


# ── Profile Persistence ──────────────────────────────────────────────────────

def save_face_profile(embeddings: List[np.ndarray]) -> bool:
    """
    Compute and save the enrolled face profile to FACE_PROFILE_FILE and GAZE_PROFILE_FILE.
    """
    if not embeddings:
        log.error("No face embeddings provided — profile not saved.")
        return False

    mat = np.array(embeddings, dtype=np.float64)
    face_mean = np.mean(mat, axis=0)
    face_std  = np.std(mat, axis=0)

    profile = {
        "face_mean":       face_mean,
        "face_std":        face_std,
        "face_embeddings": embeddings,
        "model":           FACE_MODEL,
        "n_frames":        len(embeddings),
        "threshold":       MATCH_THRESHOLD,
        "timestamp":       time.time(),
    }

    # Save to primary v2 face profile
    joblib.dump(profile, FACE_PROFILE_FILE)
    log.info("Face profile saved (%d frames) → %s", len(embeddings), FACE_PROFILE_FILE)

    # Also sync to GAZE_PROFILE_FILE so older checks find an enrolled profile
    try:
        joblib.dump(profile, GAZE_PROFILE_FILE)
    except Exception:
        pass

    return True


def load_enrolled_profile() -> Tuple[Optional[dict], str]:
    """
    Load enrolled profile from disk.
    Checks FACE_PROFILE_FILE, then GAZE_PROFILE_FILE, then DATA_PROFILE_FILE.
    
    Returns (profile_dict, mode) where mode is 'face' or 'legacy_gaze'.
    """
    for path in (FACE_PROFILE_FILE, GAZE_PROFILE_FILE):
        if os.path.exists(path):
            try:
                data = joblib.load(path)
                if isinstance(data, dict):
                    if "face_mean" in data:
                        return data, "face"
                    elif "mean" in data and "cov" in data:
                        return data, "legacy_gaze"
            except Exception as exc:
                log.warning("Could not read profile at %s: %s", path, exc)

    if os.path.exists(DATA_PROFILE_FILE):
        try:
            data = np.load(DATA_PROFILE_FILE, allow_pickle=True).item()
            if isinstance(data, dict) and "face_mean" in data:
                return data, "face"
        except Exception as exc:
            log.warning("Could not read %s: %s", DATA_PROFILE_FILE, exc)

    return None, "none"


# ── Complete Verification Session ────────────────────────────────────────────

def verify_face_session(
    duration_secs: float = SESSION_SECS,
    camera_index: int = 0,
) -> dict:
    """
    Run a complete identity check session:
      1. Load enrolled face profile
      2. Capture face embeddings from webcam
      3. Compare cosine distance to enrolled centroid
      4. Return detailed verification result dict
    """
    profile, mode = load_enrolled_profile()
    if profile is None:
        log.warning("No face profile found on disk — please enrol first.")
        return {
            "error": "No face profile found. Complete Enrolment first.",
            "match": False,
            "combined": None,
            "face_dist": None,
            "gaze_dist": None,
            "geom_dist": None,
            "threshold": MATCH_THRESHOLD,
            "n_frames": 0,
            "timestamp": time.time(),
            "snapshot_b64": None,
        }

    # Open webcam and collect embeddings
    embeddings, best_frame = collect_face_embeddings(
        duration_secs=duration_secs,
        fps=6.0,
        camera_index=camera_index,
    )
    snapshot_b64 = _frame_to_b64(best_frame) if best_frame is not None else None

    # Fail closed if no face detected in camera
    if not embeddings:
        log.warning("CV check: no face detected — failing closed (intruder or camera blocked).")
        return {
            "error": "No face detected in camera",
            "match": False,
            "combined": 2.0,
            "face_dist": 2.0,
            "gaze_dist": 2.0,
            "geom_dist": None,
            "threshold": MATCH_THRESHOLD,
            "n_frames": 0,
            "timestamp": time.time(),
            "snapshot_b64": snapshot_b64,
        }

    if mode == "face":
        session_mean = np.mean(embeddings, axis=0)
        enrolled_mean = np.asarray(profile["face_mean"], dtype=np.float64)
        dist = cosine_distance(session_mean, enrolled_mean)
        thr = float(profile.get("threshold", MATCH_THRESHOLD))
        match = dist < thr

        log.info("Face check: dist=%.4f (thr=%.2f) match=%s [%d frames]", dist, thr, match, len(embeddings))
        return {
            "match":        bool(match),
            "combined":     round(dist, 4),
            "face_dist":    round(dist, 4),
            "gaze_dist":    round(dist, 4),   # alias for backward-compatible table views
            "geom_dist":    None,
            "threshold":    thr,
            "n_frames":     len(embeddings),
            "timestamp":    time.time(),
            "snapshot_b64": snapshot_b64,
            "error":        None,
        }

    else:
        # Fallback for legacy gaze profiles with 'mean' and 'cov'
        log.warning("Using legacy gaze profile fallback.")
        return {
            "match": False,
            "combined": 2.0,
            "face_dist": 2.0,
            "gaze_dist": 2.0,
            "geom_dist": None,
            "threshold": MATCH_THRESHOLD,
            "n_frames": len(embeddings),
            "timestamp": time.time(),
            "snapshot_b64": snapshot_b64,
            "error": "Profile format requires re-enrolment with Face Capture",
        }


def run_cv_check() -> bool:
    """
    Capture a short webcam session and compare against the enrolled face profile.
    Used by cascaded_auth/monitor.py.

    Returns
    -------
    True  — identity confirmed
    False — impostor / mismatch or no face detected
    """
    result = verify_face_session(duration_secs=SESSION_SECS)
    return bool(result.get("match", False))


# ── Backward Compatibility Aliases ───────────────────────────────────────────

def collect_gaze_vectors(duration_secs: float = SESSION_SECS) -> list:
    """Legacy alias for collect_face_embeddings."""
    embs, _ = collect_face_embeddings(duration_secs=duration_secs)
    return embs


def save_gaze_profile(vectors: list) -> None:
    """Legacy alias for save_face_profile."""
    save_face_profile(vectors)


def get_gaze_vector(face_landmarks, frame_w: int, frame_h: int) -> np.ndarray:
    """Legacy stub in case an external caller references get_gaze_vector."""
    return np.zeros(4, dtype=np.float64)
