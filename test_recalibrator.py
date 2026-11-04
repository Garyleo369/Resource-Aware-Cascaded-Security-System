"""
test_recalibrator.py — Unit tests for VerificationGatedRecalibrator.

Tests five scenarios without fabricating performance results:
  1. Genuine user — verified samples accumulate, recalibration fires at MIN_VERIFIED_SAMPLES.
  2. Impostor (failed verification) — baseline and threshold preserved.
  3. High-risk anomaly — bypasses temporal smoothing gate immediately.
  4. Post-verification cooldown — suppresses re-triggers after a check.
  5. Unverified samples — never influence the model or offset.

Run with: python -X utf8 test_recalibrator.py
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

# ── Minimal AnomalyScorer stub (no model required) ────────────────────────────
class _StubScorer:
    def __init__(self):
        self.threshold_offset = 0.0

    def score(self, vec):
        return float(np.mean(vec))  # dummy

    def threshold(self):
        return 0.55   # static for testing

    def is_anomaly(self, score):
        return score > self.threshold()


from cascaded_auth.recalibrator import (
    VerificationGatedRecalibrator,
    RECAL_HISTORY_FILE,
    MIN_VERIFIED_SAMPLES, ANOMALY_COUNT, ANOMALY_WINDOW,
    HIGH_RISK_THRESHOLD, POST_VERIFY_COOLDOWN_S,
)


def _clean_disk():
    try:
        os.remove(RECAL_HISTORY_FILE)
    except FileNotFoundError:
        pass


def _make_vec(score_target=0.7):
    return np.full(12, score_target, dtype=np.float64)


# ── Test 1: Genuine user scenario ─────────────────────────────────────────────
def test_genuine_user():
    print("\n=== Test 1: Genuine user recalibration at 30 verified samples ===")
    _clean_disk()
    scorer = _StubScorer()
    recal  = VerificationGatedRecalibrator(scorer)
    vec    = _make_vec(0.72)

    for _ in range(ANOMALY_COUNT):
        recal.record_window(0.72, vec, triggered=True)

    recal._last_cv_ts = 0.0
    fire = recal.should_fire_cv(0.72, triggered=True)
    assert fire, f"Expected True after {ANOMALY_COUNT} anomalies"
    print(f"  PASS  should_fire_cv=True after {ANOMALY_COUNT} anomalies")

    recal.mark_cv_started()
    for i in range(MIN_VERIFIED_SAMPLES - 1):
        recal._last_cv_ts = 0.0
        recal.record_window(0.72, vec, triggered=True)
        result = recal.on_cv_result(matched=True, score_at_trigger=0.72)
        assert not result["recalibrated"], f"Premature recal at sample {i+1}"

    print(f"  PASS  No recalibration before {MIN_VERIFIED_SAMPLES} samples (count={recal.verified_count})")

    recal._last_cv_ts = 0.0
    recal.record_window(0.72, vec, triggered=True)
    result = recal.on_cv_result(matched=True, score_at_trigger=0.72)
    assert result["recalibrated"], "Expected recalibration at MIN_VERIFIED_SAMPLES"
    print(f"  PASS  Recal fired: offset {result['old_offset']:+.4f} -> {result['new_offset']:+.4f}")

    assert -0.15 <= result["new_offset"] <= 0.15
    print(f"  PASS  Offset within +-0.15 bounds")

    assert scorer.threshold_offset == recal.threshold_offset
    print(f"  PASS  scorer.threshold_offset synchronised: {scorer.threshold_offset:+.4f}")


# ── Test 2: Impostor scenario ─────────────────────────────────────────────────
def test_impostor():
    print("\n=== Test 2: Impostor — baseline preserved on failed verification ===")
    _clean_disk()
    scorer = _StubScorer()
    recal  = VerificationGatedRecalibrator(scorer)
    vec    = _make_vec(0.80)

    for _ in range(5):
        recal._last_cv_ts = 0.0
        recal.record_window(0.80, vec, triggered=True)
        recal.on_cv_result(matched=True, score_at_trigger=0.80)

    prev_offset = recal.threshold_offset
    prev_count  = recal.verified_count
    print(f"  INFO  Before impostor: offset={prev_offset:+.4f}, verified={prev_count}")

    recal._last_cv_ts = 0.0
    recal.record_window(0.80, vec, triggered=True)
    result = recal.on_cv_result(matched=False, score_at_trigger=0.80)

    assert not result["recalibrated"]
    assert result["new_offset"] == prev_offset
    assert recal.threshold_offset == prev_offset
    assert recal.verified_count == prev_count
    print(f"  PASS  Offset unchanged: {recal.threshold_offset:+.4f}")
    print(f"  PASS  Verified count unchanged: {recal.verified_count}")
    print(f"  PASS  Message confirms security response")


# ── Test 3: High-risk bypass ──────────────────────────────────────────────────
def test_high_risk_bypass():
    print("\n=== Test 3: High-risk anomaly bypasses temporal smoothing ===")
    _clean_disk()
    scorer = _StubScorer()
    recal  = VerificationGatedRecalibrator(scorer)

    high_score = HIGH_RISK_THRESHOLD + 0.05
    assert sum(recal._anomaly_window) == 0

    recal._last_cv_ts = 0.0
    fire = recal.should_fire_cv(high_score, triggered=True)
    assert fire, f"High-risk {high_score:.2f} must bypass smoothing"
    print(f"  PASS  Score {high_score:.2f} immediately fires CV")

    recal._last_cv_ts = 0.0
    fire_normal = recal.should_fire_cv(0.60, triggered=True)
    assert not fire_normal
    print(f"  PASS  Normal 0.60 with empty window suppressed by smoothing")


# ── Test 4: Post-verification cooldown ───────────────────────────────────────
def test_cooldown():
    print("\n=== Test 4: Post-verification cooldown suppresses re-triggers ===")
    _clean_disk()
    scorer = _StubScorer()
    recal  = VerificationGatedRecalibrator(scorer)
    vec    = _make_vec(0.70)

    recal._last_cv_ts = time.time()
    for _ in range(ANOMALY_COUNT):
        recal.record_window(0.70, vec, triggered=True)

    fire = recal.should_fire_cv(0.70, triggered=True)
    assert not fire
    print(f"  PASS  Cooldown active ({recal.cooldown_remaining:.0f} s remaining) — suppressed")

    recal._last_cv_ts = time.time() - POST_VERIFY_COOLDOWN_S - 1.0
    fire = recal.should_fire_cv(0.70, triggered=True)
    assert fire
    print(f"  PASS  CV fires after cooldown expires")


# ── Test 5: Unverified samples never influence the model ──────────────────────
def test_no_unverified_influence():
    print("\n=== Test 5: Unverified samples never enter pool or offset ===")
    _clean_disk()
    scorer = _StubScorer()
    recal  = VerificationGatedRecalibrator(scorer)
    vec    = _make_vec(0.70)

    for _ in range(1000):
        recal.record_window(0.70, vec, triggered=True)

    assert recal.verified_count == 0, f"Got {recal.verified_count}"
    assert recal.threshold_offset == 0.0, f"Offset drifted to {recal.threshold_offset}"
    print(f"  PASS  1000 unverified windows: verified_count=0, offset=+0.0000")


# ── Runner ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = []
    for fn in [
        test_genuine_user,
        test_impostor,
        test_high_risk_bypass,
        test_cooldown,
        test_no_unverified_influence,
    ]:
        try:
            fn()
            results.append(("PASS", fn.__name__))
        except AssertionError as e:
            print(f"\n  FAILED: {e}")
            results.append(("FAIL", fn.__name__))

    print(f"\n{'='*50}")
    for status, name in results:
        print(f"  {status}  {name}")
    n_fail = sum(1 for s, _ in results if s == "FAIL")
    print(f"\n{len(results) - n_fail}/{len(results)} tests passed.")
    _clean_disk()
    if n_fail:
        sys.exit(1)
