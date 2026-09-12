# Two-Tier Cascaded Continuous Authentication — Prototype

A feasibility prototype for a **continuous biometric authentication** system that combines:

| Tier | Modality | Technology | Trigger |
|------|----------|-----------|---------|
| **Tier 1** | Keystroke Dynamics (KSD) | IsolationForest anomaly detection | Always-on |
| **Tier 2** | Computer Vision (CV) | Mediapipe FaceMesh + Mahalanobis distance | On-demand (fired by Tier 1) |

---

## Project Structure

```
prototype/
├── main.py           # Entry point — launches all threads
├── enrol.py          # One-time setup script
├── dashboard.py      # Streamlit live monitor
├── requirements.txt
│
├── ksd/
│   ├── listener.py   # pynput keyboard event logging
│   ├── features.py   # 14-D keystroke feature vector extraction
│   ├── model.py      # IsolationForest scoring wrapper
│   └── threshold.py  # Rolling dynamic threshold (mean + 1.5σ)
│
├── cv/
│   ├── session.py    # Webcam + FaceMesh capture session
│   ├── gaze.py       # Iris tracking + session statistics
│   ├── geometry.py   # Facial ratio extraction + statistics
│   └── matcher.py    # Mahalanobis distance matcher
│
└── data/             # Auto-generated
    ├── model.joblib  # Trained IsolationForest
    └── gaze_profile.npy  # Enrolled gaze mean + covariance
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Enrol (one-time)

```bash
python enrol.py
```

This runs two concurrent tasks:

- **Keyboard**: Type naturally in any application. Press `Esc` when done.
  Requires ≥ 500 keystrokes (10 × 50-key windows) for a good model.
- **Webcam**: Automatically captures 30 seconds of gaze data. Look naturally
  at the screen. A preview window will open.

Artefacts saved to `data/`:
- `data/model.joblib` — trained IsolationForest
- `data/gaze_profile.npy` — gaze mean vector + covariance matrix

---

## Running

### Authentication system (headless)

```bash
python main.py
```

Press **Ctrl+C** to stop.

### Live dashboard

```bash
streamlit run dashboard.py
```

Opens at `http://localhost:8501`.

---

## Architecture Details

### Tier 1 — Keystroke Dynamics

```
pynput Listener → raw event queue
    ↓
Feature extraction (14-D vector):
  • Dwell time  stats [mean, std, p10, p50, p90]
  • Flight time stats [mean, std, p10, p50, p90]
  • Top-4 digraph mean flight times
    ↓
IsolationForest.score_samples() → normalise to [0,1]
    ↓
Rolling window (n=20) → threshold = mean + 1.5σ
    ↓
score > threshold → SET cv_trigger Event
```

### Tier 2 — Computer Vision

```
WAIT on cv_trigger Event (sleep state)
    ↓ triggered
Capture 3 s × 15 fps webcam frames
    ↓
Mediapipe FaceMesh (refine_landmarks=True)
    ↓
Per-frame extraction:
  • Gaze vector    [gx, gy, iod_ratio]       (iris landmarks)
  • Geometry vector [iod_norm, nose_chin, lip_curve]
    ↓
Session mean gaze + geometry
    ↓
Mahalanobis distance vs enrolled profile
    ↓
Combined = 0.65 × gaze_dist + 0.35 × geom_dist
    ↓
match = combined < 3.5
```

### Thread Management

```
main.py
  ├── threading.Event()   cv_trigger
  ├── queue.Queue()       result_queue
  ├── threading.Event()   stop_event
  │
  ├── Thread("KSD-Tier1", daemon=True)  → ksd_thread_fn()
  ├── Thread("CV-Tier2",  daemon=True)  → cv_thread_fn()
  └── Main thread → consume_results()  [blocks until Ctrl-C]
```

---

## Feature Vector Reference

| Index | Feature | Unit |
|-------|---------|------|
| 0 | Dwell mean | ms |
| 1 | Dwell std | ms |
| 2 | Dwell p10 | ms |
| 3 | Dwell p50 | ms |
| 4 | Dwell p90 | ms |
| 5 | Flight mean | ms |
| 6 | Flight std | ms |
| 7 | Flight p10 | ms |
| 8 | Flight p50 | ms |
| 9 | Flight p90 | ms |
| 10-13 | Top-4 digraph mean flight times | ms |

---

## Notes & Known Limitations

- **Webcam index**: defaults to `0`. Change `camera_index` in `main.py` if
  your webcam is on a different index.
- **Geometry matching**: The CV geometry component uses a proxy Euclidean
  distance since geometry is not separately enrolled. For production, enrol
  geometry ratios alongside gaze vectors.
- **Permissions**: `pynput` requires elevated permissions or Accessibility
  access on macOS; on Windows it works without elevation.
- **Threshold cold start**: The dynamic threshold requires ≥ 2 KSD windows
  before it becomes active (returns `None` until then).
