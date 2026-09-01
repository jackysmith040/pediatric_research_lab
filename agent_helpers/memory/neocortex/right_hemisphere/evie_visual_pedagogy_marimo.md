---
neuron_id: evie_visual_pedagogy_marimo
title: Evie Visual Pedagogy & Marimo Interactive Scaffolding
synaptic_weight: 90
corpus_callosum: pediatric_occlusion_and_slicing
blindspot: false
summary: Design principles for teaching computer vision concepts to visual learners using Marimo reactive notebooks and step-by-step visual scaffolding.
---

# 🌸 Evie's Visual Pedagogy Principles for Marimo Notebooks

## 1. Core Educational Philosophy
1. **Never Show Naked Equations Without A Visual Intuition First**: Every mathematical concept (loss function, NMS threshold, feature map) must have an interactive slider or diagram before the formula is presented.
2. **Reactive Visual Sandboxes**: Marimo notebooks allow direct reactive feedback. When the Director moves a confidence slider (`mo.ui.slider`), the bounding box overlays and slice grids update immediately.
3. **Pacing & Cognitive Bandwidth**: Break the pipeline into 4 clear visual phases:
   - **Phase 1: Ingestion & Seeing** (CCTV frame exploration, Roboflow Supervision annotation).
   - **Phase 2: The Occlusion Challenge** (Slicing & Zooming interactive playground).
   - **Phase 3: The Distillation Chamber** (DINOv3 Teacher -> YOLO Student representation transfer).
   - **Phase 4: Tracking & Counting** (Zone counters, ByteTrack, entry/exit gates).
