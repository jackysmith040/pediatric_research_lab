"""
Unit Tests for Pediatric Vision Distillation & Evaluation Engine
===============================================================
Tests:
- Vectorized IoU and single IoU calculations.
- COCO 101-point AP calculation.
- Precision, Recall, F1, mAP@50, and mAP@[50:95].
- Occlusion segmentation calculations.
- Distillation fidelity (Cosine similarity & MSE).
- Hardware profiler (CPU parallel fallback & thread pool).
- 4-Way ablation matrix execution.
- LaTeX booktabs export and CSV export formatting.
- Synthetic benchmark generation.
"""

import math
import numpy as np
import pytest
import torch

from evaluation_metrics import (
    AblationExperimentRow,
    BoundingBox,
    DistillationFidelityResult,
    EvaluationMetricsResult,
    HardwareProfileResult,
    box_iou_matrix,
    box_iou_single,
    compute_ap_coco,
    compute_distillation_fidelity,
    compute_full_academic_metrics,
    compute_occlusion_segmented_map,
    evaluate_detection_predictions,
    export_ablation_to_csv,
    export_ablation_to_latex_table,
    generate_synthetic_pediatric_benchmark,
    profile_model_hardware,
    run_4way_ablation_matrix,
)


# ==============================================================================
# 1. IOU & GEOMETRY TESTS
# ==============================================================================

def test_box_iou_single_identical():
    box_a = BoundingBox(0.0, 0.0, 1.0, 1.0)
    box_b = BoundingBox(0.0, 0.0, 1.0, 1.0)
    assert math.isclose(box_iou_single(box_a, box_b), 1.0, rel_tol=1e-5)


def test_box_iou_single_disjoint():
    box_a = BoundingBox(0.0, 0.0, 1.0, 1.0)
    box_b = BoundingBox(2.0, 2.0, 3.0, 3.0)
    assert box_iou_single(box_a, box_b) == 0.0


def test_box_iou_single_half_overlap():
    box_a = BoundingBox(0.0, 0.0, 2.0, 1.0)  # area = 2
    box_b = BoundingBox(1.0, 0.0, 3.0, 1.0)  # area = 2, inter = [1,0,2,1]=1, union = 3
    assert math.isclose(box_iou_single(box_a, box_b), 1.0 / 3.0, rel_tol=1e-5)


def test_box_iou_matrix():
    boxes_a = [BoundingBox(0.0, 0.0, 1.0, 1.0), BoundingBox(0.0, 0.0, 2.0, 1.0)]
    boxes_b = [BoundingBox(0.0, 0.0, 1.0, 1.0), BoundingBox(5.0, 5.0, 6.0, 6.0)]
    mat = box_iou_matrix(boxes_a, boxes_b)
    assert mat.shape == (2, 2)
    assert math.isclose(mat[0, 0], 1.0, rel_tol=1e-5)
    assert mat[0, 1] == 0.0


# ==============================================================================
# 2. AP & COCO EVALUATION TESTS
# ==============================================================================

def test_compute_ap_coco_perfect():
    recalls = np.array([0.2, 0.5, 1.0])
    precisions = np.array([1.0, 1.0, 1.0])
    ap = compute_ap_coco(recalls, precisions)
    assert math.isclose(ap, 1.0, rel_tol=1e-5)


def test_compute_ap_coco_empty():
    ap = compute_ap_coco(np.array([]), np.array([]))
    assert ap == 0.0


def test_evaluate_detection_perfect_match():
    gts = [[BoundingBox(0.1, 0.1, 0.5, 0.5, class_id=0, occlusion_tier="clear")]]
    preds = [[BoundingBox(0.1, 0.1, 0.5, 0.5, confidence=0.95, class_id=0)]]
    prec, rec, f1, rec_curv, prec_curv, tp, fp, fn = evaluate_detection_predictions(
        gts, preds, iou_threshold=0.5
    )
    assert tp == 1
    assert fp == 0
    assert fn == 0
    assert math.isclose(prec, 1.0)
    assert math.isclose(rec, 1.0)
    assert math.isclose(f1, 1.0)


def test_compute_full_academic_metrics():
    gts, pred_gen = generate_synthetic_pediatric_benchmark(num_images=10)
    preds = pred_gen(model_type="distilled", slicing=True)
    metrics = compute_full_academic_metrics(gts, preds, confidence_threshold=0.25)

    assert isinstance(metrics, EvaluationMetricsResult)
    assert 0.0 <= metrics.map_50 <= 1.0
    assert 0.0 <= metrics.map_50_95 <= 1.0
    assert "clear" in metrics.segmented_map50
    assert "partial" in metrics.segmented_map50
    assert "heavy" in metrics.segmented_map50
    assert metrics.total_ground_truths == 30  # 3 per image * 10 images


# ==============================================================================
# 3. DISTILLATION FIDELITY TESTS
# ==============================================================================

def test_compute_distillation_fidelity_identical():
    features = torch.randn(2, 64, 16, 16)
    fidelity = compute_distillation_fidelity(features, features)
    assert isinstance(fidelity, DistillationFidelityResult)
    assert math.isclose(fidelity.mean_cosine_similarity, 1.0, rel_tol=1e-4)
    assert math.isclose(fidelity.normalized_mse_loss, 0.0, abs_tol=1e-5)


def test_compute_distillation_fidelity_different_dims():
    t_feat = torch.randn(2, 768, 14, 14)  # DINOv3 ViT-B shape
    s_feat = torch.randn(2, 512, 14, 14)  # YOLO backbone shape
    fidelity = compute_distillation_fidelity(t_feat, s_feat)
    assert -1.0 <= fidelity.mean_cosine_similarity <= 1.0
    assert fidelity.latent_dimension == 512


# ==============================================================================
# 4. HARDWARE PROFILER TESTS (CPU MULTI-THREADING)
# ==============================================================================

def test_profile_model_hardware_cpu():
    dummy_model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 16, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d((1, 1)),
    )
    result = profile_model_hardware(
        dummy_model,
        sample_input_shape=(1, 3, 64, 64),
        device="cpu",
        warmup_iters=3,
        timed_iters=5,
        num_threads=2,
    )
    assert isinstance(result, HardwareProfileResult)
    assert result.device == "CPU"
    assert result.num_threads_used == 2
    assert result.mean_latency_ms > 0.0
    assert result.fps > 0.0
    assert result.p50_latency_ms <= result.p95_latency_ms


# ==============================================================================
# 5. 6-WAY ABLATION & EXPORT TESTS
# ==============================================================================

def test_run_4way_ablation_matrix_and_exports():
    gts, pred_gen = generate_synthetic_pediatric_benchmark(num_images=5)

    def eval_fn(model_type: str, slicing: bool):
        preds = pred_gen(model_type=model_type, slicing=slicing)
        metrics = compute_full_academic_metrics(gts, preds)
        hw = HardwareProfileResult(
            device="CPU",
            num_threads_used=2,
            mean_latency_ms=15.0 if not slicing else 35.0,
            p50_latency_ms=14.5 if not slicing else 34.0,
            p95_latency_ms=16.0 if not slicing else 38.0,
            min_latency_ms=14.0,
            max_latency_ms=17.0,
            fps=66.7 if not slicing else 28.5,
            param_count_millions=11.2,
            estimated_gflops=28.5,
        )
        return metrics, hw

    ablation_rows = run_4way_ablation_matrix(eval_fn)
    assert len(ablation_rows) == 6

    # Verify LaTeX export
    latex = export_ablation_to_latex_table(ablation_rows)
    assert r"\begin{table}" in latex
    assert r"\toprule" in latex
    assert r"\bottomrule" in latex
    assert r"\end{table}" in latex
    assert r"\checkmark" in latex

    # Verify CSV export
    csv_str = export_ablation_to_csv(ablation_rows)
    assert "Config Name,Model,Distillation,SAHI Slicing" in csv_str
    assert "M1: Base Pretrained YOLO26s" in csv_str
    assert "M2: Traditional Fine-Tuned YOLO26s" in csv_str
    assert "M3: DINOv3 Distilled YOLO26s" in csv_str
    assert "M6: Sliced DINOv3 Distilled YOLO26s (Proposed)" in csv_str


def test_7model_registry_and_light_switches():
    """Test ALL_7_MODELS_REGISTRY and dynamic enabled_model_ids filtering."""
    from evaluation_metrics import (
        ALL_7_MODELS_REGISTRY,
        run_7model_ablation_matrix,
        generate_thesis_markdown_report,
    )
    assert len(ALL_7_MODELS_REGISTRY) == 7
    assert "base_pt" in ALL_7_MODELS_REGISTRY
    assert "distilled_student_onnx" in ALL_7_MODELS_REGISTRY

    def dummy_eval_fn(model_type: str, slicing: bool):
        metrics = EvaluationMetricsResult(
            precision=0.85, recall=0.80, f1_score=0.82, map_50=0.88, map_50_95=0.62,
            segmented_map50={"heavy": 0.75}
        )
        hw = HardwareProfileResult(
            device="CPU", num_threads_used=4, mean_latency_ms=15.0, p50_latency_ms=15.0,
            p95_latency_ms=18.0, min_latency_ms=12.0, max_latency_ms=20.0, fps=66.7,
            param_count_millions=11.2, estimated_gflops=28.5
        )
        return metrics, hw

    # Test all 7 enabled
    all_rows = run_7model_ablation_matrix(dummy_eval_fn, enabled_model_ids=None)
    assert len(all_rows) == 7

    # Test subset enabled via light switches (e.g., 3 active)
    subset_rows = run_7model_ablation_matrix(dummy_eval_fn, enabled_model_ids=["base_pt", "distilled_student_pt", "distilled_student_onnx"])
    assert len(subset_rows) == 3

    # Test Markdown report generation
    report = generate_thesis_markdown_report(subset_rows)
    assert "Academic Thesis Benchmark Report" in report
    assert "3 Enabled Models" in report


if __name__ == "__main__":
    pytest.main(["-v", __file__])

