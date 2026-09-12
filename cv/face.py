"""
cv/face.py — Deep face recognition using InsightFace (ArcFace).

InsightFace ships pre-built ONNX models — no C++ compiler or TensorFlow needed.
It produces 512-dimensional ArcFace embeddings with state-of-the-art accuracy.

Cosine similarity thresholds for ArcFace (buffalo_sc model):
    > 0.28  → same person (matches InsightFace's default)
    i.e. cosine distance < 0.72 (= 1 - 0.28)

We express the threshold in cosine DISTANCE (lower = more similar):
    < COSINE_THRESHOLD  → same person  (PASS)
    >= COSINE_THRESHOLD → different    (FAIL)
"""

import logging
import time

import cv2
import numpy as np

log = logging.getLogger(__name__)

FACE_MODEL = "ArcFace (InsightFace buffalo_sc)"

# InsightFace ArcFace cosine-distance threshold.
# buffalo_sc default similarity threshold is 0.28 → distance = 1 - 0.28 = 0.72
# We use 0.50 (a bit stricter) to reduce false positives.
COSINE_THRESHOLD = 0.50

# ── Lazy globals ───────────────────────────────────────────────────────────────
_app = None   # insightface FaceAnalysis app


def _get_app():
    global _app
    if _app is None:
        import insightface  # noqa: PLC0415
        app = insightface.app.FaceAnalysis(
            name="buffalo_sc",
            providers=["CPUExecutionProvider"],
        )
        app.prepare(ctx_id=0, det_size=(640, 640))
        _app = app
        log.info("InsightFace FaceAnalysis ready (model=buffalo_sc / ArcFace).")
    return _app


# ── Core functions ─────────────────────────────────────────────────────────────

def extract_embedding(frame_bgr: np.ndarray) -> np.ndarray | None:
    """
    Detect the largest face in a BGR frame and return its 512-D ArcFace embedding.
    Returns None if no face is detected.
    """
    try:
        app = _get_app()
        faces = app.get(frame_bgr)
        if not faces:
            return None
        # Use the face with the largest bounding-box area (most prominent / closest)
        best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        emb = np.array(best.embedding, dtype=np.float64)
        if np.linalg.norm(emb) < 1e-6:
            return None
        return emb
    except Exception as exc:
        log.debug("extract_embedding error: %s", exc)
    return None


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance in [0, 2].  0 = identical vectors."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 2.0
    return float(1.0 - np.dot(a / na, b / nb))


def capture_face_embeddings(
    duration_s: float = 3.0,
    fps: float = 5.0,
    camera_index: int = 0,
) -> list[np.ndarray]:
    """
    Open the webcam for ``duration_s`` seconds, extract ArcFace embeddings at
    up to ``fps`` frames per second, and return all valid embeddings.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        log.error("Cannot open camera index %d", camera_index)
        return []

    embeddings: list[np.ndarray] = []
    deadline       = time.time() + duration_s
    frame_interval = 1.0 / fps

    log.info("📷 Face capture started: %.0f s @ %.0f fps", duration_s, fps)

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

            emb = extract_embedding(frame)
            if emb is not None:
                embeddings.append(emb)
                log.debug("  Embedding captured — running total: %d", len(embeddings))

    finally:
        cap.release()

    log.info("Face capture complete: %d valid embeddings in %.1f s",
             len(embeddings), duration_s)
    return embeddings


def verify_against_enrolled(
    session_embeddings: list[np.ndarray],
    enrolled_mean: np.ndarray,
) -> dict:
    """
    Average all session embeddings and compare to the enrolled mean via cosine distance.

    Returns
    -------
    dict: match, distance, threshold, n_frames
    """
    valid = [e for e in session_embeddings if e is not None]
    if not valid:
        log.warning("No valid face embeddings in session — cannot verify.")
        return {
            "match": False,
            "distance": 2.0,
            "threshold": COSINE_THRESHOLD,
            "n_frames": 0,
        }

    session_mean = np.mean(valid, axis=0)
    dist = cosine_distance(session_mean, enrolled_mean)
    matched = dist < COSINE_THRESHOLD

    if matched:
        log.info("✅ Face MATCH   dist=%.4f < thr=%.2f  n_frames=%d",
                 dist, COSINE_THRESHOLD, len(valid))
    else:
        log.warning("❌ Face MISMATCH  dist=%.4f >= thr=%.2f  n_frames=%d",
                    dist, COSINE_THRESHOLD, len(valid))

    return {
        "match":     matched,
        "distance":  round(dist, 4),
        "threshold": COSINE_THRESHOLD,
        "n_frames":  len(valid),
    }
