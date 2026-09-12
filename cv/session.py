"""
cv/session.py — Webcam capture session for Tier-2 CV verification.

Captures N seconds of frames from the webcam at a target FPS, runs
Mediapipe FaceMesh with refine_landmarks=True, and returns the raw
landmark sequences for downstream gaze and geometry extraction.
"""

import time
import logging

import cv2
import mediapipe as mp
import numpy as np

log = logging.getLogger(__name__)

# ── Iris / eye landmark indices (478-point FaceMesh) ──────────────────────────
LEFT_IRIS   = [474, 475, 476, 477]
RIGHT_IRIS  = [469, 470, 471, 472]

# Eye corner indices for inter-ocular distance normalisation
LEFT_EYE_INNER  = 133
LEFT_EYE_OUTER  = 33
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263

# Nose tip and chin for geometry
NOSE_TIP = 1
CHIN     = 199

# Lip landmarks
UPPER_LIP = 13
LOWER_LIP = 14
LEFT_LIP  = 61
RIGHT_LIP = 291


def capture_verification_session(
    duration_s: float = 3.0,
    fps: float = 15.0,
    camera_index: int = 0,
) -> dict | None:
    """
    Open the webcam, run FaceMesh for ``duration_s`` seconds, and collect:

      - gaze vectors  (N, 3)  — [gx, gy, iod_ratio]
      - geometry vecs (N, 3)  — [iod_norm, nose_chin_ratio, lip_curve]

    Returns
    -------
    dict with keys "gaze" and "geometry" (each np.ndarray), or None on error.
    """
    mp_face = mp.solutions.face_mesh
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        log.error("Cannot open camera index %d", camera_index)
        return None

    gaze_vecs: list[np.ndarray]     = []
    geom_vecs: list[np.ndarray]     = []

    deadline       = time.time() + duration_s
    frame_interval = 1.0 / fps

    try:
        with mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as face_mesh:

            next_frame = time.time()

            while time.time() < deadline:
                now = time.time()
                if now < next_frame:
                    time.sleep(max(0.0, next_frame - now))
                next_frame += frame_interval

                ok, frame = cap.read()
                if not ok:
                    log.warning("Frame grab failed — skipping.")
                    continue

                h, w = frame.shape[:2]
                rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)

                if not results.multi_face_landmarks:
                    continue

                lms = results.multi_face_landmarks[0].landmark

                # ── Gaze vector ────────────────────────────────────────────────
                gv = _gaze_vector(lms, w, h)
                if gv is not None:
                    gaze_vecs.append(gv)

                # ── Geometry vector ────────────────────────────────────────────
                gm = _geometry_vector(lms, w, h)
                if gm is not None:
                    geom_vecs.append(gm)

    finally:
        cap.release()

    if not gaze_vecs or not geom_vecs:
        log.error("No valid face frames captured during verification.")
        return None

    log.info("CV session: %d gaze frames, %d geometry frames captured.",
             len(gaze_vecs), len(geom_vecs))

    return {
        "gaze":     np.array(gaze_vecs,    dtype=np.float32),
        "geometry": np.array(geom_vecs,    dtype=np.float32),
    }


# ── Helper functions ───────────────────────────────────────────────────────────

def _lm_xy(lms, idx: int, w: int, h: int) -> np.ndarray:
    return np.array([lms[idx].x * w, lms[idx].y * h])


def _centre(lms, idx_list: list[int], w: int, h: int) -> np.ndarray:
    pts = np.array([_lm_xy(lms, i, w, h) for i in idx_list])
    return pts.mean(axis=0)


def _gaze_vector(lms, w: int, h: int) -> np.ndarray | None:
    """Return [gx, gy, iod_ratio] or None."""
    try:
        lc = _centre(lms, LEFT_IRIS,  w, h)
        rc = _centre(lms, RIGHT_IRIS, w, h)

        p_li = _lm_xy(lms, LEFT_EYE_INNER,  w, h)
        p_lo = _lm_xy(lms, LEFT_EYE_OUTER,  w, h)
        p_ri = _lm_xy(lms, RIGHT_EYE_INNER, w, h)
        p_ro = _lm_xy(lms, RIGHT_EYE_OUTER, w, h)

        iod = np.linalg.norm(lc - rc)
        if iod < 1e-6:
            return None

        l_mid = (p_li + p_lo) / 2
        r_mid = (p_ri + p_ro) / 2

        gx = ((lc[0] - l_mid[0]) + (rc[0] - r_mid[0])) / (2 * iod)
        gy = ((lc[1] - l_mid[1]) + (rc[1] - r_mid[1])) / (2 * iod)

        left_width  = np.linalg.norm(p_lo - p_li)
        right_width = np.linalg.norm(p_ro - p_ri)
        iod_ratio   = iod / ((left_width + right_width) / 2 + 1e-6)

        return np.array([gx, gy, iod_ratio], dtype=np.float32)
    except Exception as exc:
        log.debug("Gaze vector error: %s", exc)
        return None


def _geometry_vector(lms, w: int, h: int) -> np.ndarray | None:
    """Return [iod_norm, nose_chin_ratio, lip_curvature] or None."""
    try:
        lc = _centre(lms, LEFT_IRIS,  w, h)
        rc = _centre(lms, RIGHT_IRIS, w, h)
        iod = np.linalg.norm(lc - rc)
        if iod < 1e-6:
            return None

        nose = _lm_xy(lms, NOSE_TIP, w, h)
        chin = _lm_xy(lms, CHIN,     w, h)
        nose_chin = np.linalg.norm(nose - chin) / iod

        ul = _lm_xy(lms, UPPER_LIP, w, h)
        ll = _lm_xy(lms, LOWER_LIP, w, h)
        rl = _lm_xy(lms, LEFT_LIP,  w, h)
        rr = _lm_xy(lms, RIGHT_LIP, w, h)
        lip_height = np.linalg.norm(ul - ll)
        lip_width  = np.linalg.norm(rl - rr)
        lip_curve  = lip_height / (lip_width + 1e-6)

        # IOD normalised by image diagonal for scale invariance
        diag      = np.sqrt(w**2 + h**2)
        iod_norm  = iod / diag

        return np.array([iod_norm, nose_chin, lip_curve], dtype=np.float32)
    except Exception as exc:
        log.debug("Geometry vector error: %s", exc)
        return None
