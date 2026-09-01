---
neuron_id: neocortex/INDEX.md
title: Neocortex — Memory Palace Lobby
synaptic_weight: 100
---

# Neocortex — Memory Palace Lobby

*The spatial hub of all structured knowledge. Navigate by Room.*

---

## 🧠 Left Hemisphere — Analytical Chamber
> Formal logic, mathematics, structured reasoning, deterministic systems.

- [`dinov3_yolo_distillation`](left_hemisphere/dinov3_yolo_distillation.md): DINOv3 Vision Foundation Distillation to YOLO
- [`pediatric_occlusion_and_slicing`](left_hemisphere/pediatric_occlusion_and_slicing.md): Pediatric Occlusion Handling & SAHI Slicing in CCTV
- [`academic_evaluation_and_thesis_metrics`](left_hemisphere/academic_evaluation_and_thesis_metrics.md): Academic Evaluation, COCO mAP & Thesis Benchmarking
- [`hardware_profiling_and_edge_deployment_roadmap`](left_hemisphere/hardware_profiling_and_edge_deployment_roadmap.md): Deep Hardware Profiling, TensorRT & Edge Telemetry Roadmap
- [`jax_evaluation_and_comparative_framework`](left_hemisphere/jax_evaluation_and_comparative_framework.md): JAX Architecture, Functional Transformations & PyTorch Comparison
- [`meta_cognitive_self_evolution_and_super_command`](left_hemisphere/meta_cognitive_self_evolution_and_super_command.md): Meta-Cognitive Self-Evolution & Super Command Protocol


---

## 🎨 Right Hemisphere — Intuition Chamber  
> Spatial reasoning, metaphor, pattern recognition, creative synthesis.

- [`visual_mental_models_pediatric_cv`](right_hemisphere/visual_mental_models_pediatric_cv.md): Visual Mental Models & Metaphors for Pediatric Vision
- [`evie_visual_pedagogy_marimo`](right_hemisphere/evie_visual_pedagogy_marimo.md): Evie Visual Pedagogy & Marimo Interactive Scaffolding

---

## How to populate a Room
1. Create a `.md` file in `neocortex/left_hemisphere/` or `neocortex/right_hemisphere/`
2. Add the required frontmatter (see template below)
3. Axon will run the scanner automatically — the graph updates itself

### Neuron Template
```yaml
---
neuron_id: unique-concept-id
title: Human-Readable Title
synaptic_weight: 40          # 1–100, increases on reinforcement
corpus_callosum: other-id    # link to opposite hemisphere counterpart
blindspot: false             # set true if Director has made errors here before
summary: One-sentence description for the graph tooltip
---
```
