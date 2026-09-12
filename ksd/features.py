"""
ksd/features.py — Keystroke feature extraction.

Produces a 17-dimensional feature vector from a list of keystroke events.

Vector layout
─────────────
Indices  0-4  : dwell-time stats   [mean, std, p10, p50, p90]   (milliseconds)
Indices  5-9  : flight-time stats  [mean, std, p10, p50, p90]   (milliseconds)
Indices 10-13 : top-4 digraph flight-time means                 (milliseconds)
Index   14    : typing speed                                     (keystrokes/sec)
Index   15    : backspace frequency                              (fraction 0-1)
Index   16    : digraph latency variance (std of all digraphs)   (milliseconds)

Both dwell and flight times are in milliseconds.

NOTE: Changing this vector layout invalidates any previously saved model.joblib.
      Re-run enrol.py (or the dashboard Keystroke Enrolment) after updates.
"""

import logging
from collections import Counter

import numpy as np

log = logging.getLogger(__name__)

# Number of digraph (bigram) features included in the vector
N_DIGRAPH = 4

# Key strings emitted by pynput for the Backspace key (covers all platforms)
_BACKSPACE_KEYS = frozenset({
    "Key.backspace",
    "\x08",          # raw backspace char (some OS / keyboard combos)
    "backspace",
})

# ── Feature vector dimension ───────────────────────────────────────────────────
FEATURE_DIM = 5 + 5 + N_DIGRAPH + 3   # = 17


def _is_backspace(ch: str) -> bool:
    """Return True if the key character string represents a backspace press."""
    return ch in _BACKSPACE_KEYS or ch.lower() in _BACKSPACE_KEYS


def _stats(arr: np.ndarray) -> list[float]:
    """Return [mean, std, p10, p50, p90] for a 1-D array; zeros if empty."""
    if len(arr) == 0:
        return [0.0] * 5
    return [
        float(np.mean(arr)),
        float(np.std(arr)),
        float(np.percentile(arr, 10)),
        float(np.percentile(arr, 50)),
        float(np.percentile(arr, 90)),
    ]


def compute_features(
    events: list[tuple[str, float]]
) -> np.ndarray | None:
    """
    Build a 17-D feature vector from a window of keystroke events.

    Parameters
    ----------
    events : list of (key_char, timestamp)
        Raw press events, ordered by time.  Each element is
        ``(character: str, press_time: float)`` where the float is from
        ``time.perf_counter()``.

    Returns
    -------
    np.ndarray of shape (17,), or None if too few events.
    """
    if len(events) < 2:
        return None

    keys  = [e[0] for e in events]
    times = [e[1] for e in events]

    # ── Dwell & flight times ───────────────────────────────────────────────────
    # We estimate dwell time as the inter-press gap when only press events are
    # available; a reasonable approximation for most keyboards.
    gaps_ms = np.array([(times[i + 1] - times[i]) * 1000
                        for i in range(len(times) - 1)])
    gaps_ms = np.clip(gaps_ms, 0, 1000)          # remove spurious outliers

    # Split heuristically: gaps < 150 ms → dwell-like, longer → flight-like
    dwell  = gaps_ms[gaps_ms <  150]
    flight = gaps_ms[gaps_ms >= 150]

    if len(dwell)  < 2:
        dwell  = gaps_ms[:max(1, len(gaps_ms) // 2)]
    if len(flight) < 2:
        flight = gaps_ms[max(1, len(gaps_ms) // 2):]

    dwell_feats  = _stats(dwell)
    flight_feats = _stats(flight)

    # ── Digraph features ───────────────────────────────────────────────────────
    digraphs = [(keys[i] + keys[i + 1], gaps_ms[i])
                for i in range(len(keys) - 1)]

    counter     = Counter(d[0] for d in digraphs)
    top_bigrams = [bg for bg, _ in counter.most_common(N_DIGRAPH)]

    digraph_feats: list[float] = []
    all_digraph_latencies: list[float] = []
    for d_pair, d_lat in digraphs:
        all_digraph_latencies.append(d_lat)
    for bg in top_bigrams:
        times_bg = [t for g, t in digraphs if g == bg]
        digraph_feats.append(float(np.mean(times_bg)))

    # Pad to N_DIGRAPH if fewer bigrams exist
    while len(digraph_feats) < N_DIGRAPH:
        digraph_feats.append(0.0)

    # ── Typing speed ───────────────────────────────────────────────────────────
    # Keystrokes per second over the window duration.
    duration_s = times[-1] - times[0]
    if duration_s > 0:
        typing_speed = len(keys) / duration_s
    else:
        typing_speed = 0.0

    # ── Backspace frequency ────────────────────────────────────────────────────
    n_backspace = sum(1 for k in keys if _is_backspace(k))
    backspace_freq = n_backspace / len(keys) if keys else 0.0

    # ── Digraph latency variance ───────────────────────────────────────────────
    if all_digraph_latencies:
        digraph_var = float(np.std(all_digraph_latencies))
    else:
        digraph_var = 0.0

    # ── Assemble vector ────────────────────────────────────────────────────────
    vec = np.array(
        dwell_feats + flight_feats + digraph_feats
        + [typing_speed, backspace_freq, digraph_var],
        dtype=np.float32,
    )
    assert vec.shape == (FEATURE_DIM,), f"Unexpected shape: {vec.shape}"
    return vec


def compute_features_from_raw(
    raw_events: list[tuple[str, str, float]]
) -> np.ndarray | None:
    """
    Build a 17-D feature vector from raw (event_type, key_char, timestamp)
    triples emitted by KeystrokeListener.

    Calculates true dwell time (press→release) and flight time
    (release_n → press_{n+1}).

    Parameters
    ----------
    raw_events : list of (\"press\"|\"release\", key_char, timestamp)

    Returns
    -------
    np.ndarray of shape (17,) or None.
    """
    press_times:   dict[str, float] = {}
    release_times: dict[str, float] = {}
    dwell_ms:  list[float] = []
    flight_ms: list[float] = []
    key_order: list[str]   = []
    digraph_times: list[tuple[str, float]] = []

    last_key:          str | None   = None
    last_release_time: float | None = None

    # Track timing for typing-speed calculation
    first_press_time: float | None = None
    last_press_time:  float | None = None
    total_presses = 0
    backspace_presses = 0

    for ev_type, ch, t in raw_events:
        if ev_type == "press":
            total_presses += 1
            if _is_backspace(ch):
                backspace_presses += 1

            press_times[ch] = t
            if first_press_time is None:
                first_press_time = t
            last_press_time = t

            if last_release_time is not None and last_key is not None:
                ft = (t - last_release_time) * 1000
                if 0 < ft < 2000:
                    flight_ms.append(ft)
                    digraph_times.append((last_key + ch, ft))
            key_order.append(ch)
            last_key = ch

        elif ev_type == "release":
            release_times[ch] = t
            if ch in press_times:
                dwell = (t - press_times[ch]) * 1000
                if 0 < dwell < 500:
                    dwell_ms.append(dwell)
                last_release_time = t

    if len(dwell_ms) < 2 or len(flight_ms) < 2:
        return None

    dwell_feats  = _stats(np.array(dwell_ms))
    flight_feats = _stats(np.array(flight_ms))

    # ── Digraph features ───────────────────────────────────────────────────────
    counter     = Counter(d[0] for d in digraph_times)
    top_bigrams = [bg for bg, _ in counter.most_common(N_DIGRAPH)]
    digraph_feats: list[float] = []
    for bg in top_bigrams:
        times_bg = [t for g, t in digraph_times if g == bg]
        digraph_feats.append(float(np.mean(times_bg)))
    while len(digraph_feats) < N_DIGRAPH:
        digraph_feats.append(0.0)

    # ── Typing speed ───────────────────────────────────────────────────────────
    if (first_press_time is not None and last_press_time is not None
            and last_press_time > first_press_time):
        duration_s = last_press_time - first_press_time
        typing_speed = total_presses / duration_s
    else:
        typing_speed = 0.0

    # ── Backspace frequency ────────────────────────────────────────────────────
    backspace_freq = backspace_presses / total_presses if total_presses > 0 else 0.0

    # ── Digraph latency variance ───────────────────────────────────────────────
    all_lats = [lat for _, lat in digraph_times]
    digraph_var = float(np.std(all_lats)) if all_lats else 0.0

    # ── Assemble vector ────────────────────────────────────────────────────────
    vec = np.array(
        dwell_feats + flight_feats + digraph_feats
        + [typing_speed, backspace_freq, digraph_var],
        dtype=np.float32,
    )
    assert vec.shape == (FEATURE_DIM,), f"Unexpected shape: {vec.shape}"
    return vec
