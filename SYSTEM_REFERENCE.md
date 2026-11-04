# Two-Tier Cascaded Continuous Biometric Authentication System
## Comprehensive System Reference & Architectural Handbook

---

**Project Name:** Resource-Aware Cascaded Continuous Authentication Prototype  
**Current Pipeline Version:** v2 (`cascaded_auth/` & `cv/`)  
**Document Scope:** Complete end-to-end technical explanation, from core fundamentals to runtime mechanics, mathematics, and operational instructions.

---

## Table of Contents
1. [Core Concept & The Fundamental Problem](#1-core-concept--the-fundamental-problem)
2. [How the System Solves It (The Cascaded Solution)](#2-how-the-system-solves-it-the-cascaded-solution)
3. [The Two Tiers in Detail](#3-the-two-tiers-in-detail)
   - [Tier 1: Keystroke Dynamics (KSD)](#tier-1-keystroke-dynamics-ksd)
   - [Tier 2: Computer Vision & Gaze Tracking (CV)](#tier-2-computer-vision--gaze-tracking-cv)
4. [Mathematical & Algorithmic Foundations](#4-mathematical--algorithmic-foundations)
5. [Complete Repository & File Structure](#5-complete-repository--file-structure)
6. [Data Flow, Threading, and Inter-Process Communication](#6-data-flow-threading-and-inter-process-communication)
7. [Stored Data, Artifacts & File Schemas](#7-stored-data-artifacts--file-schemas)
8. [Setup, Enrolment, and How to Run Everything](#8-setup-enrolment-and-how-to-run-everything)
9. [Experimental Evaluation & Verified Benchmarks](#9-experimental-evaluation--verified-benchmarks)
10. [Troubleshooting & Known Practical Details](#10-troubleshooting--known-practical-details)

---

## 1. Core Concept & The Fundamental Problem

### What is Static Authentication?
Traditional security relies on **static entry-point authentication**: you type a password, enter a PIN, tap a fingerprint scanner, or look at Windows Hello when you first turn on your PC. Once that check passes, the computer assumes you remain the same person for the entire session.

### The Security Problem (Session Hijacking & Physical Takeover)
If you step away from your desk to grab water or use the restroom without locking your screen:
* An unauthorized coworker or malicious actor can walk up to your workstation.
* They have full access to your unlocked email, source code, internal databases, and private credentials.
* The operating system has no mechanism to detect that the hands on the keyboard no longer belong to you.

### Why Not Just Use Continuous Webcams?
If we leave a webcam running continuously:
1. **Excessive Resource Consumption:** Video capture and facial landmarking consume significant CPU, GPU, and memory, rapidly draining laptop batteries.
2. **Severe Privacy Concerns:** Users strongly object to having an active camera monitoring and recording them all day long.
3. **Camera Wear & Occlusion:** Continuous polling causes sensor heating and fails when the user leans back or looks slightly away.

### Why Not Just Use Keystroke Biometrics Alone?
Keystroke Dynamics (KSD) observes typing rhythms silently with near-zero CPU usage. However, **human typing rhythm naturally drifts**:
* You get tired at the end of the day.
* You type while holding a cup or phone.
* You feel stressed or distracted.
If a keystroke-only system is tuned strictly to catch every impostor, it will constantly lock legitimate users out when their typing cadence naturally drifts.

---

## 2. How the System Solves It (The Cascaded Solution)

This project implements a **two-tier event-driven cascade** that combines the strengths of both modalities while eliminating their drawbacks:

```
[ Normal Typing ]
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│ TIER 1: Keystroke Dynamics (KSD)                        │
│ - Always on in the background                           │
│ - Ultra-lightweight (sub-12 ms, < 2% CPU)               │
│ - No camera running                                     │
└──────────────────────────┬──────────────────────────────┘
                           │
                 Typing Rhythm Changes?
                /                      \
         NO (Normal)                YES (Anomaly Detected)
              │                                 │
              ▼                                 ▼
       [ Keep Working ]              ┌─────────────────────────────────────┐
                                     │ TIER 2: Computer Vision (CV)        │
                                     │ - Wakes from sleep on demand        │
                                     │ - Captures a brief 3-second burst   │
                                     │ - Verifies Face Mesh & Gaze         │
                                     └──────────────────┬──────────────────┘
                                                        │
                                            Face & Gaze Match Profile?
                                           /                          \
                                    YES (Genuine)               NO (Impostor)
                                         │                            │
                                         ▼                            ▼
                                [ Rescue Session ]           [ Lock Workstation ]
                             (Silent False Alarm Clear)    (Instant Security Action)
```

1. **Tier 1 acts as a silent sentinel:** It continuously evaluates typing cadence without powering on the camera.
2. **Tier 2 is an on-demand validator:** It sleeps until Tier 1 flags a divergence.
3. **Self-Healing False Alarms:** If Tier 1 raises an alarm simply because you are tired, Tier 2 wakes, sees your face, clears the alarm, and returns to sleep—all within 3 seconds, without disrupting your workflow.
4. **Immediate Impostor Blocking:** If an impostor takes over your keyboard, Tier 1 detects the anomalous cadence, Tier 2 wakes up, discovers a mismatched face, and triggers a system lockout.

---

## 3. The Two Tiers in Detail

### Tier 1: Keystroke Dynamics (KSD)
* **How keystrokes are captured:** A low-level keyboard hook (`pynput.keyboard.Listener`) records two microsecond timestamps for every keystroke:
  * $t_{\text{press}}$: The exact moment the key went down.
  * $t_{\text{release}}$: The exact moment the key was released.
* **Privacy by Design:** The system **never records which key was pressed**. Characters, words, and passwords are not stored. Only relative time differences ($\Delta t$) are captured.
* **Feature Extraction (12-Dimensional Vector):**
  Keystrokes are buffered into a rolling window of **25 consecutive keys**. From these 25 keys, a 12-D summary vector is computed:
  1. **`dwell_mean`**: Average time keys remain pressed.
  2. **`dwell_std`**: Standard deviation of dwell durations.
  3. **`dwell_p10`**: 10th percentile dwell time (fast key presses).
  4. **`dwell_p50`**: Median dwell time.
  5. **`dwell_p90`**: 90th percentile dwell time (lingering key presses).
  6. **`flight_mean`**: Average transit time between releasing one key and pressing the next.
  7. **`flight_std`**: Standard deviation of flight times.
  8. **`flight_p10`**: 10th percentile flight time.
  9. **`flight_p50`**: Median flight time.
  10. **`flight_p90`**: 90th percentile flight time.
  11. **`typing_rate`**: Overall typing pace (keystrokes per second).
  12. **`burst_ratio`**: $\frac{\sigma_{\text{flight}}}{\mu_{\text{flight}}}$, measuring typing rhythm burstiness.
* **The Machine Learning Model (`IsolationForest`):**
  * An unsupervised ensemble of 200 isolation trees trained exclusively on your genuine enrolment keystrokes.
  * Measures how easily an input vector can be isolated: anomalous typing rhythms require fewer splits to isolate, resulting in higher anomaly scores.
  * Scores are normalized into $[0, 1]$, where values near $0.0$ represent normal typing and values approaching $1.0$ indicate anomalous cadence.
* **Dynamic Adaptive Threshold:**
  Instead of using an inflexible static threshold, the system maintains a rolling window of the last 20 scores and adapts dynamically:
  $$\text{Threshold} = \max\left(\text{Median}(S_{1..20}) + 1.2 \times \sigma(S_{1..20}),\ 0.45\right)$$

---

### Tier 2: Computer Vision & Gaze Tracking (CV)
* **State when not in use:** The camera thread sits completely idle in a blocked state (`cv_trigger.wait()`), consuming 0% CPU with the webcam hardware released.
* **Activation Trigger:** When Tier 1 scores breach the dynamic threshold, `cv_trigger.set()` fires, waking the Tier 2 thread immediately.
* **Frame Acquisition Burst:** Opens `cv2.VideoCapture`, capturing a concise **3.0-second burst at 15 fps (45 frames)**.
* **Landmark Extraction via Google MediaPipe FaceMesh:**
  Tracks 468 facial contour points and 10 refined iris contour landmarks:
  1. **Ocular Iris Gaze Vector:** Computes the relative horizontal and vertical offset of each pupil relative to the eyeball corners:
     $$g_x = \frac{x_{\text{iris}} - x_{\text{inner\_corner}}}{x_{\text{outer\_corner}} - x_{\text{inner\_corner}}}, \quad g_y = \frac{y_{\text{iris}} - y_{\text{upper\_lid}}}{y_{\text{lower\_lid}} - y_{\text{upper\_lid}}}$$
  2. **Scale-Invariant Facial Geometry:** Measures the Inter-Ocular Distance ($IOD$, distance between eye centers) normalized against the nose-to-chin distance and lip curvature.
* **Verification via Mahalanobis Distance:**
  Matches observed biometric measurements against the enrolled profile using covariance weighting:
  $$D_M(\vec{x}) = \sqrt{(\vec{x} - \vec{\mu}_{\text{enrol}})^T \mathbf{\Sigma}_{\text{enrol}}^{-1} (\vec{x} - \vec{\mu}_{\text{enrol}})}$$
  * $D_M \le 3.5 \implies$ **MATCH (Legitimate user rescued; alert cleared).**
  * $D_M > 3.5 \implies$ **REJECT (Impostor detected; workstation locked).**
* **Instant Dormancy:** As soon as the burst completes, `cap.release()` is called, freeing camera hardware and returning the thread to sleep.

---

## 4. Mathematical & Algorithmic Foundations

### 1. Dwell & Flight Timing
For a sequence of key events where $k$-th key has press time $P_k$ and release time $R_k$:
$$\text{Dwell}_k = R_k - P_k$$
$$\text{Flight}_k = P_{k+1} - R_k$$

### 2. Isolation Forest Anomaly Scoring
The anomaly score $s$ for sample $x$ across an ensemble of $n$ isolation trees with average path length $h(x)$ is defined as:
$$s(x, n) = 2^{-\frac{\mathbb{E}(h(x))}{c(n)}}$$
where $c(n) = 2\left(\ln(n - 1) + 0.5772156649\right) - \frac{2(n - 1)}{n}$ is the average path length of unsuccessful searches in a Binary Search Tree.

### 3. Covariance & Mahalanobis Distance
Given $N$ enrolment observations $\vec{x}_1, \dots, \vec{x}_N \in \mathbb{R}^d$, the sample mean vector $\vec{\mu}$ and covariance matrix $\mathbf{\Sigma}$ are computed as:
$$\vec{\mu} = \frac{1}{N}\sum_{i=1}^N \vec{x}_i, \qquad \mathbf{\Sigma} = \frac{1}{N-1}\sum_{i=1}^N (\vec{x}_i - \vec{\mu})(\vec{x}_i - \vec{\mu})^T$$
The Mahalanobis distance scales distances by variance across dimensions and accounts for feature correlations, ensuring robust verification even when user distance from the camera shifts slightly.

---

## 5. Complete Repository & File Structure

```
prototype/
├── main.py                     # Entry point: spawns and manages KSD, CV, and dispatcher threads
├── enrol.py                    # Guided enrolment script for keystrokes and webcam gaze baseline
├── dashboard.py                # Streamlit live telemetry dashboard with real-time charts
├── requirements.txt            # Python dependencies (opencv, mediapipe, scikit-learn, etc.)
├── README.md                   # Quick-start documentation
├── PROJECT_REPORT.md           # Formal academic research report
│
├── cascaded_auth/              # v2 Production Pipeline Modules
│   ├── ksd.py                  # v2 KSD extraction, IsolationForest loading, and rolling threshold
│   ├── cv_check.py             # v2 MediaPipe FaceMesh & Mahalanobis distance validation
│   ├── model.joblib            # Trained IsolationForest model weights
│   └── enrol_vectors.joblib    # Enrolled baseline keystroke feature vectors
│
├── ksd/                        # v1 Modular Keystroke Package
│   ├── listener.py             # pynput keyboard listener hooks
│   ├── features.py             # 12-D and 14-D feature vector extraction
│   ├── model.py                # IsolationForest scoring wrapper
│   └── threshold.py            # Rolling threshold computation
│
├── cv/                         # v1 Modular Computer Vision Package
│   ├── session.py              # Camera burst capture session
│   ├── gaze.py                 # Iris tracking and gaze coordinates
│   ├── geometry.py             # Facial geometry and landmark ratios
│   └── matcher.py              # Mahalanobis distance matcher
│
├── experiments/                # Automated Evaluation Test Suite
│   ├── run_all.py              # Main test runner executing all 4 experiments
│   ├── exp1_ksd_performance.py # ROC-AUC, EER, and score distribution analysis
│   ├── exp2_drift.py           # Temporal drift analysis across 10-day logs
│   ├── exp3_power.py           # Always-on vs. event-driven resource benchmarking
│   ├── exp4_fusion.py          # KSD vs. KSD+Face vs. KSD+Face+Gaze comparison
│   └── results/                # Output artifacts
│       ├── final_report.md     # Generated test results
│       └── *.png               # Generated ROC curves, confusion matrices, and charts
│
└── data/                       # Local State & Log Directory
    ├── session_log.json        # Historical audit log of every scoring event and CV challenge
    └── gaze_profile.npy        # Baseline gaze distribution array (mean μ and covariance Σ)
```

---

## 6. Data Flow, Threading, and Inter-Process Communication

The system coordinates three concurrent threads in `main.py`:

```
┌────────────────────────────────────────────────────────────────────────┐
│                              main.py                                   │
│                                                                        │
│  Thread 1: KSD Thread (Daemon)                                         │
│    - Listens to pynput events                                          │
│    - Maintains 25-key buffer                                           │
│    - Calculates 12-D vector                                            │
│    - Runs IsolationForest scoring                                      │
│    - If score > threshold: cv_trigger.set()                            │
│                                                                        │
│  Thread 2: CV Thread (Daemon)                                          │
│    - cv_trigger.wait() [Blocks, 0% CPU]                                │
│    - On trigger: opens camera, records 45 frames, runs MediaPipe       │
│    - Calculates Mahalanobis distance                                   │
│    - Puts result tuple into result_queue                               │
│    - cv_trigger.clear()                                                │
│                                                                        │
│  Thread 3: Main / Dispatcher Thread                                    │
│    - Reads from result_queue                                           │
│    - Appends event to data/session_log.json                            │
│    - If IMPOSTOR: executes lock_workstation()                          │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Stored Data, Artifacts & File Schemas

### 1. `data/session_log.json`
Every scoring evaluation and verification challenge appends an entry to this structured log:
```json
{
  "timestamp": "2026-09-21T09:42:15.120Z",
  "ksd_score": 0.5124,
  "threshold": 0.4500,
  "triggered": true,
  "cv_result": "MATCH",
  "mahalanobis_distance": 1.842,
  "latency_ms": 11.704,
  "action": "RESCUE_GENUINE"
}
```

### 2. `data/gaze_profile.npy`
A NumPy binary file storing:
* `mean`: 1D array of mean ocular displacement coordinates $(\mu_x, \mu_y)$.
* `cov`: 2D array representing the inverse covariance matrix $\mathbf{\Sigma}^{-1}$.

### 3. `cascaded_auth/model.joblib`
Serialized scikit-learn `IsolationForest` object containing all 200 trained decision trees and scoring offsets.

---

## 8. Setup, Enrolment, and How to Run Everything

### Step 1: Environment Installation
From the project directory:
```bash
pip install -r requirements.txt
```

### Step 2: One-Time Enrolment
Before running authentication, the system must learn your genuine biometric baseline:
```bash
python enrol.py
```
* **Keystroke Enrolment:** Type naturally in any application window. At least 500 keystrokes (~10–11 windows) are collected. Press `Esc` when finished.
* **Camera Enrolment:** A preview window will open for 30 seconds. Look naturally at the screen while it captures your baseline iris gaze distribution.
* **Generated Files:** Generates `cascaded_auth/model.joblib`, `cascaded_auth/enrol_vectors.joblib`, and `data/gaze_profile.npy`.

### Step 3: Run the Live Authentication Daemon
```bash
python main.py
```
* Runs silently in the terminal.
* Type normally—the camera remains dormant.
* Simulating an erratic typing rhythm will trigger the webcam briefly to verify your identity.
* Press `Ctrl+C` to cleanly exit.

### Step 4: Run the Live Streamlit Dashboard
```bash
python -m streamlit run dashboard.py
```
* Opens your browser at `http://localhost:8501`.
* Visualizes real-time anomaly scores, current threshold levels, trigger flags, and historical decision distributions.

### Step 5: Run Full Experimental Benchmarks
```bash
python experiments/run_all.py
```
* Executes Experiments 1 through 4 across 400 test windows.
* Generates charts and summary metrics in `experiments/results/`.

---

## 9. Experimental Evaluation & Verified Benchmarks

Empirical evaluations conducted across 400 test windows (200 genuine, 200 impostor) with random seed `42` demonstrated:

### 1. Keystroke Anomaly Separation (Tier 1 Alone)
* **Genuine User Score:** Median $= 0.1289$, Mean $= 0.1606 \pm 0.1216$.
* **Impostor Attack Score:** Median $= 0.3015$, Mean $= 0.3003 \pm 0.0116$.
* **ROC-AUC:** **0.92**, showing strong separation between genuine and anomalous typing cadences.
* **Equal Error Rate (EER):** **8.0%** at threshold $0.2897$.
* **Detection Latency:** **11.70 ms per window**.

### 2. Always-On vs. Cascaded Resource Comparison
| Metric | Always-On Camera | Cascaded Prototype | Absolute Delta |
|---|---|---|---|
| **Camera Activations** | 400 | 49 | **−87.8%** |
| **Active Camera Time** | 12,602.5 s | 1,543.8 s | **−11,058.7 s (−87.7%)** |
| **False Interruption Rate** | 100.0% (Constant poll) | 9.0% | **−91.0%** |
| **Average Decision Latency** | 31,518.2 ms | 3,871.3 ms | **8.1x faster** |

### 3. Cascaded Multi-Modal Performance
* **Single-Tier KSD:** FAR $= 9.0\%$, FRR $= 0.0\%$.
* **Cascaded KSD + Face CV:** **FAR $= 0.00\%$**, **FRR $= 0.00\%$**, **Impostor Rejection $= 100.0\%$**.
* **Self-Healing Rescues:** In 18 out of 18 instances where keystroke drift caused a Tier 1 false alarm, the Tier 2 camera check verified the user's face and rescued the session without locking the screen.

---

## 10. Troubleshooting & Known Practical Details

1. **Webcam Indexing:**  
   If you have multiple webcams (built-in vs. USB), change `camera_index=0` in `main.py` or `cascaded_auth/cv_check.py` to `camera_index=1`.
2. **OS Keyboard Permissions:**  
   * **Windows:** `pynput` works without elevation.
   * **macOS:** Requires Accessibility permissions enabled under *System Settings > Privacy & Security > Accessibility*.
   * **Linux:** Requires read permissions on `/dev/input/`.
3. **Threshold Cold Start:**  
   The rolling dynamic threshold requires at least 2 completed typing windows before activating, during which it relies on the safety floor ($0.45$).
4. **Streamlit Deprecation Warnings:**  
   In newer Streamlit releases, `st.components.v1.html` and `use_container_width=True` may display deprecation notices. These warnings are non-fatal; the dashboard remains fully functional.
