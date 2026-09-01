---
neuron_id: pediatric_occlusion_and_slicing
title: Pediatric Occlusion Handling & SAHI Slicing in CCTV
synaptic_weight: 90
corpus_callosum: visual_mental_models_pediatric_cv
blindspot: false
summary: Multi-scale inference slicing (SAHI) combined with temporal ByteTrack association to resolve severe occlusion when children are carried or obscured in hospital CCTV.
---

# 👁️ Pediatric Occlusion Handling & Slicing Architecture

## 1. The Occlusion Challenge in Hospital CCTV
In pediatric monitoring, the child is frequently occluded:
- Carried in arms or piggyback on mother/parent.
- Swaddled in blankets, strollers, or cribs.
- Low-resolution CCTV angle (overhead / wide hallway).

Standard detectors suffer from **Bounding Box Merging** (predicting one box for the adult and missing the child) or **False Suppression** via standard Non-Maximum Suppression (NMS).

## 2. Technical Countermeasures
1. **SAHI (Slicing Aided Hyper Inference) via Supervision `InferenceSlicer`**:
   - Slices $1920 \times 1080$ CCTV frames into overlapping $640 \times 640$ patches (e.g. `overlap_ratio_wh=(0.2, 0.2)`).
   - Runs student model on high-resolution crop slices to preserve tiny infant pixel details (heads, limbs).
   - Merges bounding boxes across slices using NMS / NMM (Non-Maximum Merging).
2. **Tracker Architecture: FastTracker vs. Deep OC-SORT / TrackTrack**:
   - **Stationary Hospital CCTV Context**: Fixed camera angle eliminates ego-motion. The primary failure mode is co-located bounding box overlap (infant on parent's back/arms) causing Kalman drift.
   - **Primary Choice — FastTracker (Occlusion-Aware Kalman Rollback)**: Freezes uncertainty drift when parent-child bounding boxes overlap; avoids ReID compute overhead while preventing ID fragmentation.
   - **Secondary / Heavy Choice — TrackTrack / Deep OC-SORT**: Utilizes adaptive appearance feature fusion and multi-cue association when ID swaps across long multi-person occlusions occur.

3. **Supervision Zone Counting (`PolygonZone` / `LineZone`)**:
   - Directional vector thresholds anchored to parent-child centroids to guarantee single, non-duplicated counts on hallway entry/exit.
