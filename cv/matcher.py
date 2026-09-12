"""
cv/matcher.py — Face recognition matching for CV verification.

Loads the enrolled Facenet512 embedding and compares it to the session
mean embedding using cosine distance:

    distance < COSINE_THRESHOLD  → owner confirmed (PASS)
    distance >= COSINE_THRESHOLD → possible impostor (FAIL)

COSINE_THRESHOLD = 0.30 is DeepFace's validated threshold for Facenet512.
"""

import os
import logging

import numpy as np

from cv.face import (
    COSINE_THRESHOLD,
    FACE_MODEL,
    cosine_distance,
    verify_against_enrolled,
)

log = logging.getLogger(__name__)

DATA_DIR     = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
PROFILE_PATH = os.path.join(DATA_DIR, "gaze_profile.npy")

# Re-export for dashboard / other callers that read MATCH_THRESHOLD
MATCH_THRESHOLD = COSINE_THRESHOLD


class CVMatcher:
    """
    Loads the enrolled face embedding profile and verifies session embeddings
    using Facenet512 cosine distance.
    """

    def __init__(self):
        self._profile: dict | None = None
        self._enrolled_mean: np.ndarray | None = None
        self._load_profile()

    def _load_profile(self):
        if not os.path.exists(PROFILE_PATH):
            log.warning("Profile not found at %s — run enrol.py first.", PROFILE_PATH)
            return
        try:
            data = np.load(PROFILE_PATH, allow_pickle=True).item()
            self._profile = data
            if "face_mean" in data:
                self._enrolled_mean = data["face_mean"]
                log.info("Face profile loaded (model=%s, dim=%d).",
                         data.get("model", "?"), len(self._enrolled_mean))
            else:
                log.warning("Profile missing face_mean — run enrol.py to re-enrol.")
        except Exception as exc:
            log.error("Failed to load profile: %s", exc)

    @property
    def ready(self) -> bool:
        return self._enrolled_mean is not None

    # ── Primary interface: accept session embeddings ───────────────────────────

    def match_embeddings(self, session_embeddings: list[np.ndarray]) -> dict:
        """
        Compare session face embeddings to the enrolled profile.

        Parameters
        ----------
        session_embeddings : list of np.ndarray (512,)

        Returns
        -------
        dict: match, combined (cosine dist), gaze_dist, geom_dist, threshold
        """
        if not self.ready:
            log.error("Enrolled face embedding not available — run enrol.py.")
            return {
                "match": False, "combined": 2.0,
                "gaze_dist": 2.0, "geom_dist": 2.0,
                "threshold": MATCH_THRESHOLD,
            }

        result = verify_against_enrolled(session_embeddings, self._enrolled_mean)
        dist = result["distance"]

        # Keep legacy keys (gaze_dist, geom_dist, combined) for dashboard compat
        return {
            "match":     result["match"],
            "combined":  dist,
            "gaze_dist": dist,          # same value — no longer split
            "geom_dist": dist,
            "threshold": MATCH_THRESHOLD,
            "n_frames":  result["n_frames"],
        }

    # ── Legacy interface kept for backward compatibility ───────────────────────

    def match(
        self,
        session_gaze_mean: np.ndarray | None = None,
        session_geom_mean: np.ndarray | None = None,
        session_embeddings: list | None = None,
    ) -> dict:
        """Unified match entry point — prefers embeddings over legacy means."""
        if session_embeddings is not None:
            return self.match_embeddings(session_embeddings)
        # No embeddings provided and no face profile → fail safely
        log.warning("No face embeddings provided and no legacy path. "
                    "Re-run enrol.py and restart the dashboard.")
        return {
            "match": False, "combined": 2.0,
            "gaze_dist": 2.0, "geom_dist": 2.0,
            "threshold": MATCH_THRESHOLD,
        }
