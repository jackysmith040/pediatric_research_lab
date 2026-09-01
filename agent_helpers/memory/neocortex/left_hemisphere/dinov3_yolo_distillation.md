---
neuron_id: dinov3_yolo_distillation
title: DINOv3 Vision Foundation Distillation to YOLO
synaptic_weight: 85
corpus_callosum: visual_mental_models_pediatric_cv
blindspot: false
summary: Pretraining lightweight YOLO backbones using dense feature distillation from DINOv3/DINOv2 teacher models on unlabeled domain images.
---

# 🔬 DINOv3 to YOLO Knowledge Distillation Architecture

## 1. Core Problem & Concept
Pediatric patients in hospital CCTV present extreme visual variability (body scale, infant proportions, poses, carrying methods) that standard pre-trained COCO object detectors fail on. 
Instead of training a small YOLO model from scratch or standard COCO weights on a tiny labeled dataset, **Dense Vision Foundation Distillation** transfers the semantic grounding and spatial patch representations of a frozen Foundation Teacher (**DINOv3 / DINOv2**) into a fast student detector (**YOLO26s / YOLO11s / RT-DETR**).

## 2. Mathematical Foundation
Let $T(x)$ be the frozen DINOv3 teacher feature representation at patch layer $l$, and $S_\theta(x)$ be the student YOLO backbone representation with projection head $P_\phi$.
The distillation loss $\mathcal{L}_{\text{distill}}$ aligns normalized feature embeddings:

$$\mathcal{L}_{\text{distill}} = 1 - \frac{\langle P_\phi(S_\theta(x)), T(x) \rangle}{\|P_\phi(S_\theta(x))\|_2 \cdot \|T(x)\|_2} + \alpha \mathcal{L}_{\text{smooth-L1}}(P_\phi(S_\theta(x)), T(x))$$

## 3. LightlyTrain Implementation Flow
1. **Unlabeled CCTV Ingestion**: Feed raw unannotated hospital CCTV video frames ($>10k$ frames).
2. **Pretrain Stage (`lightly_train.pretrain`)**:
   - `model="ultralytics/yolov8s.yaml"` (or YOLO11/YOLO26)
   - `method="distillation"` with `teacher="dinov3/vitb16"`
   - Output: `exported_models/exported_last.pt`
3. **Downstream Fine-Tuning (`ultralytics.YOLO`)**:
   - Load distilled backbone weights into YOLO head.
   - Train on labeled pediatric classes (`child`, `adult`, `carrying_child`, `stroller`).
