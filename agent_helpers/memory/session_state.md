# 💾 Session State Snapshot (`/remember save`)

**Timestamp:** 2026-09-01 15:35 UTC  
**Session Topic:** Two-Stage CCTV Cascade Architecture Integration & Academic Benchmarking  
**Active Personas:** Axon Self-Evolving Engine (Left Brain Orchestrator + Right Brain Evie)  
**State Machine Status:** `[STATE 5: MEMORY_CONSOLIDATION]` -> Ready for `[STATE 1: SENSORY_INGESTION]`

---

## 📌 1. Project Objective & Core Problem
- **Domain**: Pediatric patient detection and entry/exit counting in hospital CCTV feeds.
- **Key Challenge**: Severe occlusions — carried infants, babies on parents' backs/chests, strollers, and swaddles, plus sitting posture ambiguities.
- **Solution Strategy**: 
  1. **Two-Stage CCTV Cascade Architecture**:
     - *Stage 1 (Proposals)*: Base Pretrained YOLO26s (COCO `person` class 0) achieving ~100% human proposal recall.
     - *Stage 2 (Triage & Posture Calibration)*: Secondary fine-tuned/distilled crop evaluator with adult-default fallback, seated aspect ratio calibration ($w/h > 0.65$ or $h_{\text{ratio}} > 0.20 \implies \text{Adult}$), and nested infant recovery.
  2. **Knowledge Distillation**: Frozen **DINOv3 / DINOv2** Foundation Teacher -> **YOLO26s / YOLO11s** Student via `LightlyTrain`.
  3. **Occlusion Handling & Small Object Detection**: **SAHI** multi-scale slicing via `supervision.InferenceSlicer`.
  4. **Tracking & Counting**: **FastTracker (Kalman Rollback)** with `supervision` and directional continuous flow counters.
  5. **Interactive Visual Learning**: **Marimo** visual reactive lab with 4-Way Arena, model light switches, diagrams, and step-by-step intuition designed by Evie.

---

## 🏛️ 2. Key Decisions & Architectural Locks
| Category | Decision | Details / Rationale |
| :--- | :--- | :--- |
| **Cascade Pipeline** | `TwoStagePediatricCascadePipeline` | Stage 1 COCO human proposal recall + Stage 2 crop classification with posture aspect ratio calibration. |
| **Package Manager** | `uv` | High-speed virtual environment and dependency manager on Windows. |
| **Notebook Engine** | `Marimo` | Pure Python, reactive DAG execution, rich interactive UI (`mo.ui.slider`, 4-Way Quad Canvas). |
| **Teacher Model** | `dinov3/vitb16` | High-capacity dense visual representations. |
| **Student Model** | `yolo26s.pt`, `pediatric-model.pt`, `best.pt` | Fast edge-friendly real-time object detection (`child`, `adult`). |
| **Edge Acceleration** | `ONNX Runtime` | `best.onnx`, `pediatric-model.onnx` production edge inference engines. |
| **Tracking & Slicing** | `FastTracker` + `SAHI` | Kalman rollback on co-located parent-child tracks + multi-scale native slicing. |

---

## 📋 3. Next Session Actions
1. Run DINOv3 Distillation & Fine-Tuning job on MoLab GPU (`distillation_notebook/dinov3_yolo26s_distillation_train.py`).
2. Run full 8-Model Academic Ablation Suite in Marimo (`distillation_notebook/pediatric_vision_lab.py` Chapter 5) and export LaTeX tables into thesis draft.
3. Real-time Multi-zone Benchmarking on OPD waiting area corridor vs triage entry.
4. Execute Deep Profiling Roadmap (Tier 2 `torch.profiler`, Tier 3 `trtexec` TensorRT FP16/INT8, Tier 4 Jetson `jtop` power telemetry).

