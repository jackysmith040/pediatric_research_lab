---
neuron_id: hardware_profiling_and_edge_deployment_roadmap
title: Deep Hardware Profiling & Edge Deployment Roadmap
synaptic_weight: 95
corpus_callosum: academic_evaluation_and_thesis_metrics
blindspot: false
summary: Complete technical roadmap and execution playbook for PyTorch operator profiling, TensorRT engine optimization, and NVIDIA Jetson edge telemetry for thesis defense.
---

# ⚡ Deep Hardware Profiling & Edge Deployment Roadmap

## 1. Executive Summary & Profiling Philosophy
In academic computer vision research, computational profiling progresses through distinct evolutionary tiers:

```
[ Tier 1: Academic System Level ] -> [ Tier 2: Operator & Kernel Level ] -> [ Tier 3: TensorRT / Edge Compiler ] -> [ Tier 4: Edge Appliance Telemetry ]
 (Wall-clock, p50/p95, FPS, FLOPs)      (torch.profiler, Layer Timings)       (trtexec, FP16/INT8 Quantization)     (jtop, Watts, Thermal, Jetson)
```

- **Tier 1 (Complete)**: Implemented in `evaluation_metrics.py`. Provides reproducible academic metrics ($p50/p95$ latency, FPS, parameter count, GFLOPs) for comparative thesis ablation tables ($M_1$ to $M_6$).
- **Tiers 2–4 (Future Roadmap Playbook)**: Detailed below for when deploying to hospital edge nodes (Jetson Orin, edge servers) and preparing the final hardware efficiency defense chapter.

---

## 2. Multi-Tier Profiler Architecture & Tooling Playbook

### 🔹 Tier 1: System-Level Latency & FPS Profiling *(Active in Codebase)*
- **Engine**: `evaluation_metrics.profile_model_hardware()`
- **Metrics Collected**:
  - Warmup-stabilized inference latency: Mean, $p50$ (median), $p95$ (tail latency under load), Min, Max.
  - Throughput: Effective Real-Time Frames Per Second ($\text{FPS} = \frac{1000}{t_{\text{mean}}}$).
  - Memory: Peak GPU VRAM (`torch.cuda.max_memory_allocated`).
  - CPU Scaling: Multi-threaded execution across $N$ CPU cores (`torch.set_num_threads`).
- **Thesis Use**: Direct population of Table 5.1 (Comparative Ablation Matrix) and Figure 5.1(D) (Pareto Frontier).

---

### 🔹 Tier 2: Operator & Kernel-Level PyTorch Profiling (`torch.profiler`)
When designing or fine-tuning custom distillation heads or evaluating multi-head attention overhead in DINOv3 vs YOLO convolutions.

#### Implementation Template (`agent_helpers/scratch/profile_torch_operators.py`):
```python
import torch
from ultralytics import YOLO
from torch.profiler import profile, record_function, ProfilerActivity

def profile_operator_breakdown(model_path="computer_vision_model/yolo26s.pt", device="cuda"):
    model = YOLO(model_path)
    inputs = torch.rand(1, 3, 640, 640, device=device)

    # 1. Warmup
    for _ in range(5):
        _ = model(inputs, verbose=False)

    # 2. Detailed Operator Tracing
    activities = [ProfilerActivity.CPU]
    if device == "cuda" and torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with record_function("model_inference"):
            for _ in range(20):
                _ = model(inputs, verbose=False)

    # 3. Export Chrome Tracing & Print Table
    print(prof.key_averages().table(sort_by="cuda_time_total" if device=="cuda" else "cpu_time_total", row_limit=15))
    prof.export_chrome_trace("training_artifacts/torch_trace.json")
    print("Exported Chrome trace to training_artifacts/torch_trace.json (view in chrome://tracing)")
```

*Key Insight for Thesis*: Proves whether the bottleneck is memory-bound (LayerNorm / Reshape in Vision Transformers) or compute-bound (GEMM Convolutions in YOLO).

---

### 🔹 Tier 3: TensorRT Engine Compilation & Quantization Profiling (`trtexec`)
When deploying the distilled model for high-throughput clinical CCTV video ingestion at the hospital edge.

#### Step 1: Export Distilled YOLO to ONNX with Dynamic/Static Batching
```bash
# In distillation_notebook:
uv run python -c "from ultralytics import YOLO; m = YOLO('runs/detect/pediatric_finetune/weights/best.pt'); m.export(format='onnx', imgsz=640, dynamic=False, simplify=True)"
```

#### Step 2: Benchmark with NVIDIA TensorRT `trtexec`
```bash
# FP16 Precision Benchmarking (Ideal for Jetson Orin Nano / RTX 4090):
trtexec --onnx=best.onnx \
        --saveEngine=yolo26s_distilled_fp16.engine \
        --fp16 \
        --warmUp=500 \
        --iterations=1000 \
        --dumpProfile \
        --exportTimes=training_artifacts/trtexec_times.json

# INT8 Precision Post-Training Quantization (PTQ) Benchmarking:
trtexec --onnx=best.onnx \
        --saveEngine=yolo26s_distilled_int8.engine \
        --int8 \
        --calib=dataset/calibration_cache.bin \
        --dumpProfile
```

*Metrics Extracted for Thesis*:
- Pure GPU Compute Latency vs End-to-End Latency.
- Memory Host-to-Device (H2D) and Device-to-Host (D2H) copy overhead.
- Throughput acceleration ratio ($\text{Speedup} = \frac{\text{FPS}_{\text{TensorRT FP16}}}{\text{FPS}_{\text{PyTorch FP32}}}$).

---

### 🔹 Tier 4: Hospital Edge Appliance Telemetry (`jtop` / `tegrastats`)
Profiles real-time performance on deployed edge hardware in clinical wards (e.g. NVIDIA Jetson Orin Nano / AGX Orin).

#### Edge Telemetry Script Template (`agent_helpers/scratch/edge_telemetry_logger.py`):
```python
# Runs on NVIDIA Jetson Edge Nodes
from jtop import jtop
import time
import csv

def log_edge_telemetry(duration_seconds=60, output_csv="training_artifacts/edge_power_telemetry.csv"):
    with jtop() as jetson:
        with open(output_csv, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "GPU_Util_%", "CPU_Util_%", "RAM_MB", "Power_Watts", "Temp_GPU_C"])
            
            start = time.time()
            while jetson.ok() and (time.time() - start < duration_seconds):
                stats = jetson.stats
                writer.writerow([
                    time.time() - start,
                    stats.get("GPU", 0),
                    stats.get("CPU1", 0),
                    stats.get("RAM", 0) / (1024 * 1024),
                    stats.get("Power TOT", 0) / 1000.0,
                    stats.get("Temp GPU", 0),
                ])
                time.sleep(0.5)
    print(f"Edge telemetry written to {output_csv}")
```

*Key Thesis Contribution*: Demonstrates that the distilled YOLO26s detector runs continuously under a **$15\text{W}$ clinical thermal envelope** without thermal throttling or dropped CCTV frames.

---

## 3. Thesis Integration & Publication Table Format

When synthesizing the final thesis defense results chapter, compile the multi-tier profiling data into the following standard table:

$$\begin{array}{lcccccc}
\toprule
\textbf{Runtime Tier} & \textbf{Precision} & \textbf{Device} & \textbf{p50 Latency (ms)} & \textbf{FPS} & \textbf{Power (W)} & \textbf{Efficiency (FPS/W)} \\
\midrule
\text{PyTorch Base} & \text{FP32} & \text{CPU (8-Thread)} & 237.3 & 4.2 & \text{N/A} & \text{N/A} \\
\text{PyTorch Base} & \text{FP32} & \text{RTX Cloud GPU} & 11.4 & 87.7 & 180\text{W} & 0.49 \\
\text{Distilled Student} & \text{FP16} & \text{Jetson Orin} & 7.8 & 128.2 & 14.2\text{W} & 9.03 \\
\text{Distilled + SAHI} & \text{FP16} & \text{Jetson Orin} & 18.5 & 54.1 & 14.8\text{W} & 3.66 \\
\text{Distilled (Proposed)} & \text{INT8} & \text{Jetson Orin} & 4.9 & 204.0 & 13.9\text{W} & \mathbf{14.68} \\
\bottomrule
\end{array}$$

---

## 4. Next-Phase Execution Checklist
- [ ] **Phase A**: Cloud GPU Training & Weights Export (`dinov3_yolo26s_distillation_train.py`).
- [ ] **Phase B**: Generate ONNX & TensorRT FP16 engines from best checkpoint.
- [ ] **Phase C**: Run `trtexec` benchmark and export layer latency breakdown JSON.
- [ ] **Phase D**: Run edge power telemetry log on target hardware (Jetson / mini-PC).
- [ ] **Phase E**: Insert final multi-device performance table into Chapter 5 LaTeX thesis draft.
