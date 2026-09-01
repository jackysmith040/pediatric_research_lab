<<<<<<< HEAD
# pediatric_research_lab
=======
# 🔬 Pediatric Research Lab: Knowledge Distillation & Edge Vision for Hospital CCTV

An advanced computer vision and knowledge distillation research pipeline engineered for **pediatric patient detection, occlusion-robust tracking, and automated entry/exit counting** in high-density hospital outpatient department (OPD) CCTV feeds.

---

## 🌟 Key Innovations & Architecture

1. **Foundation Feature Distillation**:
   - **Teacher**: Dense visual representations from frozen **DINOv3 / DINOv2** (`dinov3/vitb16`, `dinov2/vitb14`).
   - **Student**: Ultra-lightweight edge detector **YOLO26s / YOLO11s** trained via `LightlyTrain` feature distillation + task-loss fine-tuning.
2. **Pediatric Occlusion & Small Object Recovery**:
   - Multi-scale **SAHI (Slicing Aided Hyper Inference)** via `supervision.InferenceSlicer` to recover heavily occluded infants (swaddles, baby carriers, strollers).
3. **Continuous Tracking & Directional Zone Telemetry**:
   - **ByteTrack** and **FastTracker** tracking algorithms integrated with `supervision.PolygonZone` and `LineZone` for directional flow counting and occupancy monitoring.
4. **Interactive Reactive Research Lab**:
   - Pure Python reactive interface built with **Marimo**, featuring live stream telemetry, patch zoom inspectors, attention heatmap simulators, and 6-model academic ablation matrix.
5. **Neuro-Symbolic Conscience Agent**:
   - Built-in **Axon Conscience OS** with dual-hemisphere personas (Left-Brain Orchestrator + Right-Brain Evie), episodic memory logs, and neocortex knowledge graphs.

---

## 📂 Repository Structure

```
.
├── .agents/                                # Antigravity & Agent skill definitions (19 Axon skills)
├── agent_helpers/                          # Axon Conscience OS (Memory, Neocortex, Personas, Log)
│   ├── CONSCIENCE.md                       # Universal bootloader & 19 Laws of Consciousness
│   ├── memory/                             # Episodic logs, KANBAN, session states & neurons
│   └── personas/                           # Orchestrator (Left brain) & Evie (Right brain)
├── distillation_notebook/                  # Core computer vision codebase & interactive labs
│   ├── pediatric_vision_lab.py             # Main interactive Marimo Visual Study Lab (Ch 1-6)
│   ├── dinov3_yolo26s_distillation_train.py# Cloud DINOv3 -> YOLO26s distillation pipeline
│   ├── traditional_yolo26s_finetune_train.py# Supervised YOLO baseline fine-tuning script
│   ├── evaluation_metrics.py               # COCO mAP@[50:95], occlusion metrics, LaTeX export
│   ├── test_evaluation_metrics.py          # Academic metric pytest suite (12/12 unit tests)
│   ├── computer_vision_model/              # Base, fine-tuned, distilled .pt & .onnx models
│   └── pyproject.toml                      # UV project configuration and dependencies
├── 07_EXPERIMENTS_BENCHMARKS_AND_RESULTS.md# Comprehensive thesis experimental benchmark tables
├── notebook_run.txt                        # Command cheatsheet for running all modules
└── notebook_run.bat                        # Windows one-click launcher
```

---

## 🚀 Quickstart & Usage

### 1. Installation via `uv`
```bash
# Navigate to notebook directory
cd distillation_notebook

# Install all dependencies into virtual environment
uv sync
```

### 2. Launch Interactive Marimo Lab
```bash
# Interactive Edit Mode (Sliders, video players, parameter tuners)
uv run marimo edit pediatric_vision_lab.py

# Read-Only Dashboard App Mode
uv run marimo run pediatric_vision_lab.py
```

### 3. Run Academic Unit Tests
```bash
# Execute evaluation suite
uv run pytest -v test_evaluation_metrics.py
```

### 4. Run Knowledge Distillation & Training
```bash
# DINOv3 -> YOLO26s Feature Distillation
uv run marimo edit dinov3_yolo26s_distillation_train.py

# Supervised Baseline Fine-Tuning
uv run marimo edit traditional_yolo26s_finetune_train.py
```

---

## 📊 Evaluation & Metrics

The system supports rigorous academic benchmarking across:
* **Standard Detection**: $\text{mAP}@50$, $\text{mAP}@[50:95]$, Precision, Recall, $F_1$-score.
* **Occlusion Robustness**: Performance breakdown across No-Occlusion, Moderate ($30-70\%$), and Severe ($>70\%$) occlusion slices.
* **Distillation Fidelity**: Latent representation cosine similarity between Teacher and Student embeddings.
* **Edge Latency & Throughput**: Millisecond inference timing and FPS on CUDA, CPU, and ONNX Runtime.

---

## 📜 License & Citation

Developed for academic thesis and clinical computer vision deployment in pediatric healthcare settings.
>>>>>>> 34314a8 (Initial commit: Pediatric Research Lab with Knowledge Distillation, SAHI Occlusion Slicing, and Interactive Marimo Visual Lab)
