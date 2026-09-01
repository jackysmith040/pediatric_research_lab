"""
🏥 Pediatric Research Lab: Live Real-Time CCTV Inference & Tracking Engine
==========================================================================
Interactive real-time video stream runner for supervisor presentations and clinical demonstrations.

Features:
- Two-Stage Cascade Architecture (Stage 1 COCO Proposal Recall + Stage 2 Specialized Triage)
- Biomechanical Seating Calibration (eliminates seated adult false positives)
- FastTracker (Kalman Rollback for carried/swaddled infant occlusion recovery)
- Live High-Density HUD (Active Children, Active Adults, FPS, Tracking IDs, Latency)
- Interactive Hotkeys:
  * [SPACE]: Pause / Resume playback
  * [M]: Switch Active Model (Two-Stage Cascade -> DINOv3 Distilled -> Supervised FT -> Base Pretrained)
  * [T]: Toggle Multi-Object Tracking ON/OFF
  * [C]: Toggle CLAHE Adaptive Contrast Enhancement
  * [S]: Slow Motion Mode (0.5x)
  * [R]: Restart Video Stream from Beginning
  * [Q] or [ESC]: Exit Live Viewer
"""

import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Ensure distillation_notebook directory is in path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

try:
    import supervision as sv
except ImportError:
    sv = None

from evaluation_metrics import TwoStagePediatricCascadePipeline


def find_model_paths(base_dir: Path):
    """Locates model checkpoint files across workspace."""
    base_m_cand = [
        base_dir / "computer_vision_model" / "base_model" / "yolo26s.pt",
        base_dir / "computer_vision_model" / "yolo26s.pt",
    ]
    ft_m_cand = [
        base_dir / "computer_vision_model" / "fine_tune_model" / "pediatric-model.pt",
        base_dir / "computer_vision_model" / "yolo26s_finetuned.pt",
    ]
    dist_m_cand = [
        base_dir / "computer_vision_model" / "distilled_model" / "best.pt",
        base_dir / "computer_vision_model" / "best.pt",
    ]

    base_p = next((p for p in base_m_cand if p.exists()), None)
    ft_p = next((p for p in ft_m_cand if p.exists()), None)
    dist_p = next((p for p in dist_m_cand if p.exists()), None)
    return base_p, ft_p, dist_p


def find_video_path(base_dir: Path):
    """Locates hospital CCTV test video."""
    video_dir = base_dir / "video_for_testing_model"
    if video_dir.exists():
        vids = list(video_dir.glob("*.mp4"))
        if vids:
            return vids[0]
    return None


def draw_cinematic_hud(
    frame: np.ndarray,
    model_name: str,
    fps: float,
    latency_ms: float,
    active_children: int,
    active_adults: int,
    total_children: int,
    total_adults: int,
    frame_idx: int,
    total_frames: int,
    is_paused: bool,
    tracking_enabled: bool,
    clahe_enabled: bool,
) -> np.ndarray:
    """Draws a professional hospital telemetry HUD on the frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # 1. Top Header Bar
    bar_h = 75
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (13, 17, 23), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Accent Line
    cv2.line(frame, (0, bar_h), (w, bar_h), (0, 229, 255), 2)

    # Title & Badge
    cv2.circle(frame, (25, 38), 8, (0, 0, 255) if not is_paused else (0, 165, 255), -1)
    status_text = "REC LIVE" if not is_paused else "PAUSED"
    cv2.putText(frame, status_text, (42, 44), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    # Model Badge
    model_badge = f"ENGINE: {model_name.upper()}"
    cv2.putText(frame, model_badge, (170, 44), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 229, 255), 2, cv2.LINE_AA)

    # Telemetry Counters (Center/Right)
    child_text = f"CHILDREN: {active_children} (TOTAL: {total_children})"
    adult_text = f"ADULTS: {active_adults} (TOTAL: {total_adults})"
    fps_text = f"{fps:.1f} FPS ({latency_ms:.1f} ms)"

    # Draw Child Counter (Cyan Pill)
    x_c = int(w * 0.48)
    cv2.rectangle(frame, (x_c - 10, 14), (x_c + 280, 60), (0, 229, 255), 1)
    cv2.putText(frame, child_text, (x_c, 44), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 229, 255), 2, cv2.LINE_AA)

    # Draw Adult Counter (Gold Pill)
    x_a = int(w * 0.70)
    cv2.rectangle(frame, (x_a - 10, 14), (x_a + 260, 60), (0, 215, 255), 1)
    cv2.putText(frame, adult_text, (x_a, 44), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 215, 255), 2, cv2.LINE_AA)

    # Draw FPS Badge
    x_fps = w - 180
    cv2.putText(frame, fps_text, (x_fps, 44), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 102), 2, cv2.LINE_AA)

    # 2. Bottom Help Bar
    bot_h = 36
    cv2.rectangle(overlay, (0, h - bot_h), (w, h), (13, 17, 23), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.line(frame, (0, h - bot_h), (w, h - bot_h), (255, 255, 255), 1)

    controls_str = f"FRAME: {frame_idx}/{total_frames}  |  [SPACE] Pause/Play  |  [M] Change Model  |  [T] Tracking ({'ON' if tracking_enabled else 'OFF'})  |  [C] CLAHE ({'ON' if clahe_enabled else 'OFF'})  |  [R] Restart  |  [Q] Exit"
    cv2.putText(frame, controls_str, (20, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

    return frame


def run_live_inference(video_source=None, target_width=1280, conf_thresh=0.25):
    """Main real-time interactive playback and inference loop."""
    base_dir = CURRENT_DIR

    base_p, ft_p, dist_p = find_model_paths(base_dir)
    print("=" * 70)
    print("🏥 PEDIATRIC VISION LAB: LIVE REAL-TIME CCTV INFERENCE STREAM")
    print("=" * 70)
    print(f"Base YOLO26s:         {base_p.name if base_p else 'Not Found'}")
    print(f"Supervised FT YOLO26s:{ft_p.name if ft_p else 'Not Found'}")
    print(f"DINOv3 Distilled:     {dist_p.name if dist_p else 'Not Found'}")

    # Load YOLO models
    base_model = YOLO(str(base_p)) if base_p and base_p.exists() else None
    ft_model = YOLO(str(ft_p)) if ft_p and ft_p.exists() else base_model
    dist_model = YOLO(str(dist_p)) if dist_p and dist_p.exists() else ft_model

    if base_model is None:
        print("❌ Error: Could not locate base YOLO model weights.")
        return

    # Instantiate Two-Stage Cascade
    sec_model = dist_model if dist_model != base_model else ft_model
    cascade_pipeline = TwoStagePediatricCascadePipeline(
        base_detector=base_model,
        secondary_detector=sec_model,
        min_child_conf=0.35,
        max_aspect_ratio=0.65,
        max_h_ratio=0.20,
    )

    models_dict = {
        "Two-Stage Cascade Architecture (Proposed SOTA)": cascade_pipeline,
        "DINOv3 Distilled Student (best.pt)": dist_model,
        "Supervised Fine-Tuned (pediatric-model.pt)": ft_model,
        "Base Pretrained YOLO26s (COCO person)": base_model,
    }
    model_keys = list(models_dict.keys())
    current_model_idx = 0

    # Resolve video source
    if video_source is None:
        video_source = find_video_path(base_dir)

    if video_source is None:
        print("⚠️ No video file found in video_for_testing_model/. Trying webcam device 0...")
        video_source = 0
    else:
        print(f"🎥 Ingesting Video: {video_source}")

    cap = cv2.VideoCapture(str(video_source) if not isinstance(video_source, int) else video_source)
    if not cap.isOpened():
        print(f"❌ Error: Failed to open video source: {video_source}")
        return

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    print(f"📐 Native Resolution: {orig_w}x{orig_h} @ {video_fps:.1f} FPS | Total Frames: {total_frames}")
    print("\n👉 Controls in Video Window:")
    print("   [SPACE]: Pause / Resume")
    print("   [M]:     Switch Detection Model Engine")
    print("   [T]:     Toggle Tracker ON/OFF")
    print("   [C]:     Toggle CLAHE Contrast Boost")
    print("   [R]:     Restart Stream")
    print("   [Q]:     Quit Viewer\n")

    # Tracking Setup
    tracker = None
    if sv is not None:
        try:
            tracker = sv.ByteTrack(track_thresh=0.25, track_buffer=30, match_thresh=0.8)
        except Exception:
            pass

    box_annotator = sv.BoxAnnotator(thickness=2) if sv is not None else None
    label_annotator = sv.LabelAnnotator(text_scale=0.55, text_padding=3) if sv is not None else None
    trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=30) if sv is not None else None

    window_name = "Pediatric Research Lab - Live Hospital CCTV Stream"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    # State variables
    frame_idx = 0
    is_paused = False
    tracking_enabled = True
    clahe_enabled = False
    slow_mo = False

    cumulative_children = set()
    cumulative_adults = set()
    latencies = []

    t_prev = time.perf_counter()

    # Fast forward to prime active crowd segment (e.g. frame 200) if large video
    if total_frames > 500:
        start_frame = 200
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_idx = start_frame

    while True:
        if not is_paused:
            ret, frame = cap.read()
            if not ret or frame is None:
                # Loop back to beginning for continuous demonstration
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

            frame_idx += 1
            t_start = time.perf_counter()

            # CLAHE Enhancement if enabled
            if clahe_enabled:
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                l, a, b_ch = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
                l = clahe.apply(l)
                frame = cv2.cvtColor(cv2.merge((l, a, b_ch)), cv2.COLOR_LAB2BGR)

            active_model_name = model_keys[current_model_idx]
            active_model = models_dict[active_model_name]

            # Model Inference
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = active_model(rgb_frame, conf=conf_thresh, imgsz=target_width, verbose=False)[0]

            active_children = 0
            active_adults = 0

            # Annotation with Supervision or standard OpenCV
            if sv is not None and hasattr(results, "boxes") and len(results.boxes) > 0:
                detections = sv.Detections.from_ultralytics(results)

                if tracking_enabled and tracker is not None:
                    tracked = tracker.update_with_detections(detections)
                    if len(tracked) > 0 and tracked.tracker_id is not None:
                        for tid, cid in zip(tracked.tracker_id, tracked.class_id):
                            cname = active_model.names.get(cid, str(cid)).lower()
                            if "child" in cname or cid == 0:
                                cumulative_children.add(int(tid))
                                active_children += 1
                            else:
                                cumulative_adults.add(int(tid))
                                active_adults += 1

                        frame = trace_annotator.annotate(scene=frame, detections=tracked)
                        frame = box_annotator.annotate(scene=frame, detections=tracked)
                        labels = [
                            f"#{tid} {active_model.names.get(cid, cid).upper()}"
                            for tid, cid in zip(tracked.tracker_id, tracked.class_id)
                        ]
                        frame = label_annotator.annotate(scene=frame, detections=tracked, labels=labels)
                else:
                    for cid in detections.class_id:
                        cname = active_model.names.get(cid, str(cid)).lower()
                        if "child" in cname or cid == 0:
                            active_children += 1
                        else:
                            active_adults += 1
                    frame = box_annotator.annotate(scene=frame, detections=detections)
                    labels = [
                        f"{active_model.names.get(cid, cid).upper()} {conf:.2f}"
                        for cid, conf in zip(detections.class_id, detections.confidence)
                    ]
                    frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)
            else:
                # Fallback manual box drawing
                if hasattr(results, "boxes") and results.boxes is not None:
                    for box in results.boxes:
                        coords = box.xyxy[0].cpu().numpy().astype(int)
                        cls_id = int(box.cls[0].item())
                        c_conf = float(box.conf[0].item())
                        cname = active_model.names.get(cls_id, str(cls_id))
                        is_c = "child" in cname.lower() or cls_id == 0

                        if is_c:
                            active_children += 1
                            color = (255, 229, 0)  # Cyan/Yellow
                        else:
                            active_adults += 1
                            color = (0, 215, 255)  # Gold

                        cv2.rectangle(frame, (coords[0], coords[1]), (coords[2], coords[3]), color, 2)
                        cv2.putText(frame, f"{cname} {c_conf:.2f}", (coords[0], max(20, coords[1] - 8)), cv2.FONT_HERSHEY_DUPLEX, 0.55, color, 1)

            t_end = time.perf_counter()
            lat_ms = (t_end - t_start) * 1000.0
            latencies.append(lat_ms)
            if len(latencies) > 30:
                latencies.pop(0)
            fps = 1000.0 / (sum(latencies) / len(latencies))

            # Resize frame for smooth display
            display_frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)

            # Draw Cinematic Telemetry HUD
            display_frame = draw_cinematic_hud(
                display_frame,
                model_name=active_model_name,
                fps=fps,
                latency_ms=lat_ms,
                active_children=active_children,
                active_adults=active_adults,
                total_children=len(cumulative_children),
                total_adults=len(cumulative_adults),
                frame_idx=frame_idx,
                total_frames=total_frames,
                is_paused=is_paused,
                tracking_enabled=tracking_enabled,
                clahe_enabled=clahe_enabled,
            )

        cv2.imshow(window_name, display_frame)

        # Keyboard Interactivity
        delay = 30 if not slow_mo else 60
        key = cv2.waitKey(delay) & 0xFF

        if key in (ord('q'), ord('Q'), 27):  # Q or ESC
            break
        elif key == ord(' '):  # SPACE
            is_paused = not is_paused
        elif key in (ord('m'), ord('M')):  # Model Switch
            current_model_idx = (current_model_idx + 1) % len(model_keys)
            print(f"⚡ Switched Active Model to: {model_keys[current_model_idx]}")
        elif key in (ord('t'), ord('T')):  # Toggle Tracking
            tracking_enabled = not tracking_enabled
            print(f"🏃 Multi-Object Tracking: {'ENABLED' if tracking_enabled else 'DISABLED'}")
        elif key in (ord('c'), ord('C')):  # Toggle CLAHE
            clahe_enabled = not clahe_enabled
            print(f"✨ CLAHE Contrast Boost: {'ENABLED' if clahe_enabled else 'DISABLED'}")
        elif key in (ord('s'), ord('S')):  # Toggle Slow Mo
            slow_mo = not slow_mo
            print(f"⏱️ Slow Motion: {'ENABLED (0.5x)' if slow_mo else 'DISABLED (1.0x)'}")
        elif key in (ord('r'), ord('R')):  # Restart
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_idx = 0
            cumulative_children.clear()
            cumulative_adults.clear()
            print("🔄 Video stream restarted from Frame 0.")

    cap.release()
    cv2.destroyAllWindows()
    print("\n👋 Live inference session ended cleanly.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pediatric Research Lab Live CCTV Inference")
    parser.add_argument("--video", type=str, default=None, help="Path to video file or camera index (default: auto-detected hospital video)")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference resolution width")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    v_src = int(args.video) if args.video and args.video.isdigit() else args.video
    run_live_inference(video_source=v_src, target_width=args.imgsz, conf_thresh=args.conf)
