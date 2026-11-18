# Resource-Aware Cascaded Continuous Authentication

**A two-tier behavioral authentication research prototype**

A prototype that continuously analyzes keystroke dynamics and activates a webcam-based verification stage only when typing behavior appears anomalous. The goal is to explore whether a lightweight always-on signal can reduce unnecessary computer-vision processing.

**Stack:** Python · scikit-learn · MediaPipe · OpenCV · Streamlit · pynput

## Problem

A one-time login cannot establish that the same person remains at a workstation. Continuous camera-based verification may consume resources and raise usability concerns. This project investigates a staged approach that escalates verification only when the first signal indicates unusual behavior.

## System overview

```text
Keyboard events
     |
     v
Dwell and flight time features
     |
     v
Isolation Forest anomaly scoring
     |
     +---- normal behavior ---> continue monitoring
     |
     +---- anomaly ----------> trigger camera session
                                  |
                                  v
                           MediaPipe FaceMesh
                                  |
                                  v
                        Gaze / facial geometry
                                  |
                                  v
                         Profile comparison
                                  |
                                  v
                           Verification result
```

## Key components

| Path | Responsibility |
| --- | --- |
| `enrol.py` | Capture baseline keyboard and webcam data |
| `main.py` | Run the authentication pipeline |
| `dashboard.py` | Streamlit monitoring interface |
| `ksd/listener.py` | Capture keyboard timing events |
| `ksd/features.py` | Extract dwell, flight and digraph features |
| `ksd/model.py` | Score typing patterns with Isolation Forest |
| `ksd/threshold.py` | Manage rolling anomaly thresholds |
| `cv/session.py` | Capture triggered camera sessions |
| `cv/gaze.py` | Extract gaze-related signals |
| `cv/geometry.py` | Extract facial geometry features |
| `cv/matcher.py` | Compare captured features with the enrolled profile |

## Technical approach

### Tier 1 — Keystroke dynamics

The system builds a 14-dimensional feature representation from typing windows: summary statistics of dwell and flight times, plus selected digraph timing features. Isolation Forest provides anomaly scores without requiring a labeled dataset of every possible impostor. A rolling threshold determines when to escalate.

### Tier 2 — Camera verification

A triggered camera session uses MediaPipe facial landmarks to derive gaze and facial geometry features. The matcher compares session-level features with the enrolled baseline. The two tiers are intentionally separate so the more expensive stage does not need to run continuously.

## Run locally

**Prerequisites:** Python, a keyboard and a webcam.

```bash
pip install -r requirements.txt
python enrol.py
python main.py
```

For the monitoring dashboard, run in a separate terminal:

```bash
streamlit run dashboard.py
```

The original setup expects a substantial keyboard enrollment sample and a short webcam baseline session. Consult the code and on-screen prompts for current enrollment requirements.

## Suggested interview demo

1. Enroll a baseline user.
2. Start monitoring and show normal typing scores.
3. Introduce a deliberate change in typing rhythm.
4. Observe whether the anomaly threshold is crossed.
5. Show the triggered webcam verification stage and resulting status.

> Add an actual screenshot from `screenshots/` and a short screen recording here after checking that no sensitive biometric data or private keystrokes are exposed.

## Evaluation and limitations

See [PROJECT_REPORT.md](PROJECT_REPORT.md) and [evaluation_results.json](evaluation_results.json) for the project's evaluation artifacts. Prototype results must be interpreted in the context of the dataset, threshold selection and test protocol; they are not evidence of production-grade security.

The repository's earlier technical notes also identify limitations including geometry matching that uses a proxy distance rather than a fully enrolled geometry distribution. Camera availability, environmental conditions, enrollment drift and false positives remain important concerns.

## Responsible use

This is an experimental authentication system, not a replacement for established authentication controls. Keyboard and facial measurements are sensitive. Obtain informed consent, minimize retention and avoid uploading raw personal biometric samples to public repositories.
