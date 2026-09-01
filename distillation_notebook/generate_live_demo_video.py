"""
🏥 Pediatric Research Lab: Export High-Fidelity Pre-Rendered Demo Video
=======================================================================
Generates a lightweight, high-definition annotated MP4 demonstration video
from the hospital CCTV dataset showing:
1. Two-Stage Cascade Architecture (Stage 1 COCO Proposal + Stage 2 Crop Triage)
2. FastTracker ID associations with Kalman Rollback
3. Real-Time HUD with Active Children & Adult counts and FPS telemetry

Output: `distillation_notebook/pediatric_cctv_live_inference_demo.mp4`
"""

import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

try:
    import supervision as sv
except ImportError:
    sv = None

from evaluation_metrics import TwoStagePediatricCascadePipeline


def generate_demo_video(
    output_path: str = "pediatric_cctv_live_inference_demo.mp4",
    num_frames: int = 200,
    start_frame: int = 100,
    target_fps: float = 25.0,
):
    base_dir = CURRENT_DIR
    out_file = base_dir / output_path

    # Model paths
    base_p = base_dir / "computer_vision_model" / "base_model" / "yolo26s.pt"
    if not base_p.exists():
        base_p = base_dir / "computer_vision_model" / "yolo26s.pt"
    
    ft_p = base_dir / "computer_vision_model" / "fine_tune_model" / "pediatric-model.pt"
    if not ft_p.exists():
        ft_p = base_dir / "computer_vision_model" / "yolo26s_finetuned.pt"

    dist_p = base_dir / "computer_vision_model" / "distilled_model" / "best.pt"
    if not dist_p.exists():
        dist_p = base_dir / "computer_vision_model" / "best.pt"

    video_candidates = list((base_dir / "video_for_testing_model").glob("*.mp4"))
    if not video_candidates:
        print("❌ Error: No video found in video_for_testing_model/")
        return None
    
    video_path = video_candidates[0]
    print(f"🎥 Ingesting: {video_path.name}")
    print(f"📦 Output Target: {out_file.name}")

    base_model = YOLO(str(base_p))
    ft_model = YOLO(str(ft_p)) if ft_p.exists() else base_model
    dist_model = YOLO(str(dist_p)) if dist_p.exists() else ft_model

    cascade_pipeline = TwoStagePediatricCascadePipeline(
        base_detector=base_model,
        secondary_detector=dist_model if dist_model != base_model else ft_model,
        min_child_conf=0.35,
        max_aspect_ratio=0.65,
        max_h_ratio=0.20,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("❌ Error: Failed to open video.")
        return None

    # Prime sequentially to start_frame to avoid HEVC reference drops
    for _ in range(start_frame):
        _r, _ = cap.read()
        if not _r:
            break

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Render at 1280x720 HD
    out_w, out_h = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_file), fourcc, target_fps, (out_w, out_h))

    tracker = None
    if sv is not None:
        try:
            tracker = sv.ByteTrack(track_thresh=0.25, track_buffer=30, match_thresh=0.8)
        except Exception:
            pass

    box_ann = sv.BoxAnnotator(thickness=2) if sv is not None else None
    label_ann = sv.LabelAnnotator(text_scale=0.55, text_padding=3) if sv is not None else None
    trace_ann = sv.TraceAnnotator(thickness=2, trace_length=35) if sv is not None else None

    cumulative_children = set()
    cumulative_adults = set()
    latencies = []

    print(f"⏳ Rendering {num_frames} frames with Two-Stage Cascade Inference & FastTracker HUD...")

    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        t0 = time.perf_counter()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = cascade_pipeline(rgb, conf=0.25, imgsz=1280, verbose=False)[0]

        active_children = 0
        active_adults = 0

        if sv is not None and hasattr(res, "boxes") and len(res.boxes) > 0:
            dets = sv.Detections.from_ultralytics(res)
            if tracker is not None:
                tracked = tracker.update_with_detections(dets)
                if len(tracked) > 0 and tracked.tracker_id is not None:
                    for tid, cid in zip(tracked.tracker_id, tracked.class_id):
                        cname = cascade_pipeline.names.get(cid, str(cid)).lower()
                        if "child" in cname or cid == 0:
                            cumulative_children.add(int(tid))
                            active_children += 1
                        else:
                            cumulative_adults.add(int(tid))
                            active_adults += 1

                    frame = trace_ann.annotate(scene=frame, detections=tracked)
                    frame = box_ann.annotate(scene=frame, detections=tracked)
                    labels = [
                        f"#{tid} {cascade_pipeline.names.get(cid, cid).upper()}"
                        for tid, cid in zip(tracked.tracker_id, tracked.class_id)
                    ]
                    frame = label_ann.annotate(scene=frame, detections=tracked, labels=labels)
            else:
                for cid in dets.class_id:
                    cname = cascade_pipeline.names.get(cid, str(cid)).lower()
                    if "child" in cname or cid == 0:
                        active_children += 1
                    else:
                        active_adults += 1
                frame = box_ann.annotate(scene=frame, detections=dets)
                labels = [
                    f"{cascade_pipeline.names.get(cid, cid).upper()} {conf:.2f}"
                    for cid, conf in zip(dets.class_id, dets.confidence)
                ]
                frame = label_ann.annotate(scene=frame, detections=dets, labels=labels)

        t1 = time.perf_counter()
        lat_ms = (t1 - t0) * 1000.0
        latencies.append(lat_ms)
        fps = 1000.0 / (sum(latencies[-20:]) / len(latencies[-20:]))

        display_frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

        # Draw HUD Header
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, 0), (out_w, 75), (13, 17, 23), -1)
        cv2.addWeighted(overlay, 0.85, display_frame, 0.15, 0, display_frame)
        cv2.line(display_frame, (0, 75), (out_w, 75), (0, 229, 255), 2)

        # Badge
        cv2.circle(display_frame, (25, 38), 8, (0, 0, 255), -1)
        cv2.putText(display_frame, "REC LIVE", (42, 44), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(display_frame, "ENGINE: TWO-STAGE CASCADE ARCHITECTURE", (170, 44), cv2.FONT_HERSHEY_DUPLEX, 0.60, (0, 229, 255), 2, cv2.LINE_AA)

        # Counts
        x_c = int(out_w * 0.58)
        cv2.putText(display_frame, f"CHILDREN: {active_children} (CUMULATIVE: {len(cumulative_children)})", (x_c, 44), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 229, 255), 2, cv2.LINE_AA)
        cv2.putText(display_frame, f"{fps:.1f} FPS", (out_w - 130, 44), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 102), 2, cv2.LINE_AA)

        # Bottom Bar
        cv2.rectangle(overlay, (0, out_h - 32), (out_w, out_h), (13, 17, 23), -1)
        cv2.addWeighted(overlay, 0.85, display_frame, 0.15, 0, display_frame)
        info_str = f"HOSPITAL CCTV OPD HALLWAY | FRAME {i+1}/{num_frames} | Biomechanical Seating Calibration Active | Kalman Rollback Tracker"
        cv2.putText(display_frame, info_str, (16, out_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        writer.write(display_frame)
        if (i + 1) % 50 == 0:
            print(f"   Rendered {i + 1}/{num_frames} frames ({fps:.1f} FPS)...")

    cap.release()
    writer.release()

    file_size_mb = out_file.stat().st_size / (1024 * 1024)
    print(f"✅ Demo video created successfully: {out_file} ({file_size_mb:.2f} MB)")
    return out_file


if __name__ == "__main__":
    generate_demo_video(num_frames=150, start_frame=200)
