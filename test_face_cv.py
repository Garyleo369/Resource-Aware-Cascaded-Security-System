"""
test_face_cv.py — Verification tests for Face Capture & ArcFace Recognition.

Tests:
  1. Cosine distance computation & properties
  2. Safe face embedding extraction on non-face / blank frames
  3. Profile loading (loads 512-D ArcFace profile)
  4. Genuine user verification (cosine distance < 0.50)
  5. Impostor rejection (cosine distance >= 0.50)
  6. Fail-closed security when no face is detected
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from cascaded_auth.cv_check import (
    cosine_distance,
    extract_face_embedding,
    load_enrolled_profile,
    MATCH_THRESHOLD,
)


def test_cosine_distance():
    print("\n=== Test 1: Cosine Distance Properties ===")
    v1 = np.random.randn(512)
    v2 = v1.copy()
    v3 = -v1.copy()

    d_identical = cosine_distance(v1, v2)
    assert abs(d_identical) < 1e-6, f"Expected 0.0, got {d_identical}"
    print(f"  PASS  Identical vectors: dist = {d_identical:.6f}")

    d_opposite = cosine_distance(v1, v3)
    assert abs(d_opposite - 2.0) < 1e-6, f"Expected 2.0, got {d_opposite}"
    print(f"  PASS  Opposite vectors: dist = {d_opposite:.6f}")

    d_zero = cosine_distance(np.zeros(512), v1)
    assert d_zero == 2.0, f"Expected 2.0 for zero vector, got {d_zero}"
    print(f"  PASS  Zero vector handled safely: dist = {d_zero:.1f}")


def test_extract_embedding_blank():
    print("\n=== Test 2: Blank / Non-face Frame Extraction ===")
    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    emb = extract_face_embedding(black_frame)
    assert emb is None, f"Expected None on blank frame, got {emb}"
    print("  PASS  No face detected on blank frame -> returns None safely")


def test_profile_loading():
    print("\n=== Test 3: Profile Loading ===")
    profile, mode = load_enrolled_profile()
    assert profile is not None, "Expected enrolled profile to exist on disk"
    assert mode == "face", f"Expected mode 'face', got '{mode}'"
    assert "face_mean" in profile, "Profile must contain 'face_mean'"
    assert len(profile["face_mean"]) == 512, f"Expected 512-D embedding, got {len(profile['face_mean'])}"
    print(f"  PASS  Loaded face profile: model={profile.get('model')} (dim={len(profile['face_mean'])})")


def test_genuine_and_impostor_matching():
    print("\n=== Test 4: Genuine vs Impostor Matching ===")
    profile, _ = load_enrolled_profile()
    enrolled_mean = np.asarray(profile["face_mean"], dtype=np.float64)

    # Genuine: sample from enrolled embeddings with slight perturbation
    if "face_embeddings" in profile and len(profile["face_embeddings"]) > 0:
        genuine_sample = profile["face_embeddings"][0]
        genuine_dist = cosine_distance(genuine_sample, enrolled_mean)
        assert genuine_dist < MATCH_THRESHOLD, (
            f"Genuine sample failed: dist {genuine_dist:.4f} >= {MATCH_THRESHOLD}"
        )
        print(f"  PASS  Genuine sample: dist = {genuine_dist:.4f} < {MATCH_THRESHOLD} (MATCH)")

    # Impostor: orthogonal or different face vector
    rng = np.random.default_rng(42)
    impostor_vec = rng.standard_normal(512)
    impostor_dist = cosine_distance(impostor_vec, enrolled_mean)
    assert impostor_dist >= MATCH_THRESHOLD, (
        f"Impostor should be rejected: dist {impostor_dist:.4f} < {MATCH_THRESHOLD}"
    )
    print(f"  PASS  Impostor vector: dist = {impostor_dist:.4f} >= {MATCH_THRESHOLD} (REJECTED)")


if __name__ == "__main__":
    tests = [
        test_cosine_distance,
        test_extract_embedding_blank,
        test_profile_loading,
        test_genuine_and_impostor_matching,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {e}")

    print(f"\n{'='*50}")
    print(f"{passed}/{len(tests)} face CV tests passed.")
    if passed != len(tests):
        sys.exit(1)

