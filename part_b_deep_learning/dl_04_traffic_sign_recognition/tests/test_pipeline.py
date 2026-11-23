"""
Tier 3 & Tier 4 Automated Tests: End-to-End Pipeline, Metric Persistence, and PDF Report Compilation.
"""

import json
import os
import sys
from pathlib import Path
import pytest
import torch

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.evaluate import analyze_speed_vs_warning_confusion, train_pipeline
from src.model import TrafficSignMultiScaleCNN
from src.utils import compile_project_report, load_checkpoint, set_seed


def test_speed_vs_warning_confusion_analysis():
    """Validates specialized speed sign vs warning sign confusion computation."""
    # Class 1 (30 km/h, speed) predicted as Class 2 (50 km/h, speed) -> intra-speed confusion
    # Class 3 (60 km/h, speed) predicted as Class 18 (General caution, warning) -> speed_to_warning
    # Class 19 (Curve left, warning) predicted as Class 4 (70 km/h, speed) -> warning_to_speed
    # Class 25 (Road work, warning) predicted as Class 25 (Road work) -> correct
    y_true = [1, 3, 19, 25]
    y_pred = [2, 18, 4, 25]

    res = analyze_speed_vs_warning_confusion(y_true, y_pred)
    assert res["total_speed_samples"] == 2
    assert res["total_warning_samples"] == 2
    assert res["speed_to_warning_confusions"] == 1
    assert res["warning_to_speed_confusions"] == 1
    assert res["intra_speed_confusions"] == 1
    assert res["speed_to_warning_rate"] == 0.5
    assert res["warning_to_speed_rate"] == 0.5


def test_end_to_end_train_pipeline(tmp_path):
    """Executes a 1-epoch mini pipeline verifying metric logging, model checkpointing and plots."""
    set_seed(42)
    metrics = train_pipeline(
        epochs=1,
        lr=0.005,
        batch_size=16,
        device="cpu",
        data_dir=str(tmp_path / "data"),
        results_dir=str(tmp_path / "results"),
        model_save_path=str(tmp_path / "models" / "model.pth")
    )

    # Verify returned metrics
    assert "test_accuracy" in metrics
    assert "macro_f1" in metrics
    assert "speed_vs_warning_analysis" in metrics
    assert metrics["epochs_trained"] == 1

    # Verify files created on disk
    assert (tmp_path / "results" / "metrics.json").exists()
    assert (tmp_path / "results" / "evaluation_plots" / "training_curves.png").exists()
    assert (tmp_path / "results" / "evaluation_plots" / "confusion_matrix.png").exists()
    assert (tmp_path / "results" / "evaluation_plots" / "speed_sign_analysis.png").exists()
    assert (tmp_path / "models" / "model.pth").exists()

    # Verify model checkpoint can be loaded and perform inference
    loaded_model = TrafficSignMultiScaleCNN(num_classes=43)
    load_checkpoint(loaded_model, tmp_path / "models" / "model.pth", device="cpu")
    loaded_model.eval()
    test_tensor = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        out = loaded_model(test_tensor)
    assert out.shape == (1, 43)


def test_pdf_report_compilation(tmp_path):
    """Verifies that compile_project_report generates an enterprise 12-section PDF."""
    mock_metrics = tmp_path / "metrics.json"
    mock_metrics.write_text(json.dumps({
        "test_accuracy": 0.965,
        "macro_f1": 0.952,
        "macro_precision": 0.958,
        "macro_recall": 0.949,
        "total_parameters": 856235
    }), encoding="utf-8")

    pdf_target = tmp_path / "project_report.pdf"
    res_path = compile_project_report(output_path=pdf_target, metrics_path=mock_metrics)

    assert Path(res_path).exists()
    assert os.path.getsize(res_path) > 5000  # Non-trivial PDF output
