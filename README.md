 ObstacleAssist — Real-Time Edge AI for the Visually Impaired

> **M.Sc. Thesis · FOM University Cologne · Big Data & Business Analytics · Grade: 1.9**  
> Developed by Aleksej Weidemann · Supervised by Prof. Dr. Rüdiger Buchkremer & Dr. Thorsten Weber

[![Platform](https://img.shields.io/badge/Platform-Android%2013-3DDC84?style=flat-square&logo=android&logoColor=white)](https://android.com)
[![Model](https://img.shields.io/badge/Model-YOLOv8n%20%2B%20ONNX-00AEEF?style=flat-square)](https://github.com/ultralytics/ultralytics)
[![Edge AI](https://img.shields.io/badge/Inference-On--Device%20Only-orange?style=flat-square)]()
[![Language](https://img.shields.io/badge/Language-Kotlin-7F52FF?style=flat-square&logo=kotlin&logoColor=white)](https://kotlinlang.org)
[![Status](https://img.shields.io/badge/Status-Thesis%20Completed-success?style=flat-square)]()

---

## Problem

Over 1 billion people worldwide live with visual impairment. Existing assistive technologies are often cloud-dependent, expensive, or require specialist hardware. This project addresses the gap: **a fully on-device, real-time obstacle detection system running on a commodity Android smartphone — no internet connection required.**

---

## What it does

ObstacleAssist is an Android app that:
- Captures live camera frames
- Runs **YOLOv8n** inference via **ONNX Runtime** entirely on-device
- Detects 15 obstacle classes relevant to urban pedestrian navigation
- Prioritizes warnings using a custom **Dangerscore** (combining proximity, zone, class priority, and confidence)
- Delivers spoken German-language warnings via **Android TTS**
- Logs structured CSV telemetry per frame and per warning event for reproducible evaluation

The system was evaluated across **3 real urban scenarios in Düsseldorf** with 18 total runs and over 18,000 processed frames.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Android App (Kotlin)                     │
│                                                                 │
│  ┌──────────┐    ┌────────────────┐    ┌────────────────────┐  │
│  │  Camera  │───▶│ Pre-processing │───▶│  ONNX Runtime      │  │
│  │  Stream  │    │ (YUV → RGB)    │    │  YOLOv8n inference │  │
│  └──────────┘    │ ~46–53 ms      │    │  ~99–107 ms        │  │
│                  └────────────────┘    └────────────┬───────┘  │
│                                                     │           │
│  ┌──────────┐    ┌────────────────┐    ┌────────────▼───────┐  │
│  │ Android  │◀───│  Dangerscore   │◀───│  Post-processing   │  │
│  │   TTS    │    │  Warning Logic │    │  NMS + filtering   │  │
│  │ (German) │    │                │    │  ~3 ms             │  │
│  └──────────┘    └────────────────┘    └────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  CSV Logging: frame_metrics + warn_events (per run)      │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                        ↓ No cloud. No internet. No latency overhead.
```

---

## Key Design Decisions

**Why ONNX Runtime instead of TFLite?**  
ONNX Runtime provides a stable, framework-agnostic inference path from the YOLOv8 Ultralytics export. It avoids TFLite delegate instability on mid-range hardware and allows direct export of YOLOv8 without conversion artifacts.

**Why YOLOv8n (Nano) instead of larger variants?**  
YOLOv8n is the optimal point on the accuracy/latency trade-off for edge deployment on a Qualcomm Snapdragon 865. Larger variants (YOLOv8s/m) showed significantly higher inference latency without sufficient accuracy gains to justify the cost in a real-time assistive context.

**Why a custom Dangerscore instead of raw confidence?**  
A pure confidence-based approach would frequently prioritize distant, high-confidence detections over nearby, lower-confidence objects in the direct walking path. The Dangerscore weights four factors: `confidence × area_ratio × zone_weight × class_priority` — empirically validated against a Confidence-Only baseline with no measurable latency overhead (< 2 ms difference across all scenarios).

**Why no cloud processing?**  
Cloud inference introduces unpredictable latency, requires constant connectivity, and raises privacy concerns for a personal assistive device. Full on-device processing was a deliberate architectural constraint from the start.

---

## System Performance (Field Evaluation)

Tested on **Samsung Galaxy S20 FE** (Snapdragon 865, ~6 GB RAM, Android 13) — a mid-range, commercially available device. No GPU delegation.

### Pipeline Latency (Dangerscore Mode)

| Scenario | Avg FPS | Median E2E | P90 E2E | Stall Rate |
|---|---|---|---|---|
| S1 — Residential street | 6.50 | 148 ms | 161 ms | 0.00% |
| S2 — City centre (Berliner Allee) | 5.70 | 163 ms | 212 ms | 0.07% |
| S3 — Pedestrian zone (Schadowstraße) | 6.10 | 157 ms | 203 ms | 0.08% |

Pipeline breakdown (Scenario S1): Pre-processing ~46 ms · Inference ~99 ms · Post-processing ~3 ms

The **Dangerscore adds zero measurable latency overhead** vs. Confidence-Only (<2 ms difference, all scenarios).

### Warning Events (Dangerscore Mode, 9 runs)

- **400 warning events** · ~1,597 seconds total runtime · 9,195 processed frames
- Warning rate: 13.8 warnings/min (S1) → 17.8 warnings/min (S3, pedestrian zone)
- Top warning classes in field: Person 45.5% · Car 24.0% · Large vehicle 13.0% · Construction obstacle 9.2%
- Median warning interval: 3.2 s (S1/S2) · 2.5 s (S3)
- Zero warnings under 500 ms apart across all scenarios (cooldown mechanism working)

---

## 15 Obstacle Classes

`Person` · `Auto` · `Grosses_Fahrzeug` · `Motorrad_E-Scooter` · `Fahrrad` · `Hund` · `Baustellenhindernis` · `Laterne_Pfosten` · `Muelltonne` · `Sitzbank` · `Stuhl` · `Tisch` · `Zaun_Absperrung` · `Ampel` · `Zebrastreifen`

Class selection was motivated by real urban pedestrian hazard profiles. Vehicles are split by hazard severity (small vs. large), dogs are included due to their prevalence and unpredictability, and `Laterne_Pfosten` covers multiple vertical urban obstacles under one class.

---

## Optimization Techniques

Three key optimizations to minimize per-frame processing time on Android:

1. **Reusable buffers** — all intermediate pipeline buffers are pre-allocated once and reused per frame, eliminating Garbage Collection pressure
2. **Bulk YUV transfer** — camera data transferred via bulk array copy, reducing native JNI calls from hundreds of thousands to three per frame; integer arithmetic used for color conversion
3. **Temporal smoothing** — bounding boxes displayed only when detected in ≥2 of the last 3 frames with IoU ≥ 0.35, eliminating visual flicker without increasing latency

---

## Evaluation Design

- **Scenarios:** 3 real Düsseldorf locations with increasing complexity (residential → city centre → pedestrian zone)
- **Runs per scenario:** 6 (3× Dangerscore mode + 3× Confidence-Only baseline)
- **Ablation comparison:** Dangerscore vs. Confidence-Only — same pipeline, different warning selection logic
- **Telemetry:** Two structured CSV logs per run (`frame_metrics` + `warn_events`) with full reproducibility (run_id, device_model, conf_threshold, selection_mode logged per run)
- **Statistical reporting:** All latency metrics reported as mean, p50, p90, min/max — not just averages, since latency spikes matter more than means in safety-critical assistive systems

---

## Repository Structure

```
ObstacleAssist/
├── app/
│   └── src/main/
│       ├── java/com/example/obstacleassist/
│       │   ├── MainActivity.kt          # Camera + UI orchestration
│       │   ├── InferenceEngine.kt       # ONNX Runtime inference wrapper
│       │   ├── WarnLogic.kt             # Dangerscore + cooldown logic
│       │   ├── CsvLogger.kt             # frame_metrics + warn_events logging
│       │   └── TtsManager.kt            # Android TTS integration
│       └── assets/
│           └── model.onnx               # Fine-tuned YOLOv8n (Epoch 28)
├── analysis/
│   └── evaluation.ipynb                 # Full evaluation: plots + metrics
├── docs/
│   └── architecture.png
└── README.md
```

---

## Setup & Usage

**Prerequisites:** Android Studio · Android device with API level 30+ · ~200 MB storage

```bash
git clone https://github.com/AleksejWeidemann/ObstacleAssist.git
cd ObstacleAssist
# Open in Android Studio → Build → Run on device
```

The ONNX model is available on request.

**Logs** are written to the device's external storage as CSV files per run. Use `analysis/evaluation.ipynb` to reproduce all evaluation charts.


---

## Author

**Aleksej Weidemann**  
M.Sc. Big Data & Business Analytics · FOM University Cologne

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/aleksej-weidemann-879929194/)
[![GitHub Profile](https://img.shields.io/badge/GitHub-AleksejWeidemann-181717?style=flat-square&logo=github)](https://github.com/AleksejWeidemann)
[![Email](https://img.shields.io/badge/Email-aleksejweidemann%40outlook.com-D14836?style=flat-square)](mailto:aleksejweidemann@outlook.com)

