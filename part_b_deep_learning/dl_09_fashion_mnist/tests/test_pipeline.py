"""
Integration and end-to-end pipeline tests for DL-09: Fashion Item Classification.
Adheres to Tier 3 and Tier 4 testing standards.
"""

import importlib
import json
import os
import sys
import numpy as np
import pytest
import torch
import torch.nn as nn

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from src.data_loader import CLASS_NAMES, load_fashion_mnist
from src.evaluate import (
    analyze_apparel_confusion,
    compute_metrics,
    evaluate_model,
    train_epoch,
)
from src.model import FashionCNN
from src.preprocessor import create_dataloaders
from src.utils import compile_project_report, save_metrics, set_seed


def test_mini_training_and_evaluation_pipeline(tmp_path):
    """Test 1 mini-epoch of training and evaluation, verifying loss and weights change."""
    set_seed(42)
    device = torch.device("cpu")

    train_ds, test_ds = load_fashion_mnist(use_synthetic=True, n_synthetic_samples=60, seed=42)
    train_loader, test_loader = create_dataloaders(train_ds, test_ds, batch_size=16)

    model = FashionCNN(num_classes=10).to(device)
    initial_weights = model.classifier[1].weight.clone().detach()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    assert train_loss > 0.0
    assert 0.0 <= train_acc <= 1.0

    # Assert weights updated
    updated_weights = model.classifier[1].weight.clone().detach()
    assert not torch.equal(initial_weights, updated_weights)

    # Evaluate
    val_loss, y_true, y_pred, y_probs = evaluate_model(model, test_loader, criterion, device)
    assert val_loss > 0.0
    assert len(y_true) == len(test_ds)
    assert len(y_pred) == len(test_ds)

    metrics = compute_metrics(y_true, y_pred, y_probs)
    assert "test_accuracy" in metrics
    assert "macro_f1" in metrics
    assert "top_3_accuracy" in metrics
    assert "per_class_f1" in metrics


def test_analyze_apparel_confusion():
    """Verify analyze_apparel_confusion properly isolates Shirt, T-shirt, Coat, and Pullover."""
    # Construct a synthetic 10x10 confusion matrix with known values
    cm = np.zeros((10, 10), dtype=int)
    # Diagonal correct predictions
    for i in range(10):
        cm[i, i] = 20

    # Inject errors in tops:
    # Class 0: T-shirt/top, Class 2: Pullover, Class 4: Coat, Class 6: Shirt
    cm[6, 0] = 5  # Shirt misclassified as T-shirt
    cm[6, 2] = 4  # Shirt misclassified as Pullover
    cm[4, 2] = 3  # Coat misclassified as Pullover
    cm[1, 7] = 2  # Trouser misclassified as Sneaker (non-top error)

    result = analyze_apparel_confusion(cm, CLASS_NAMES)

    assert "tops_classes" in result
    assert len(result["tops_classes"]) == 4
    sub_cm = np.array(result["sub_confusion_matrix"])
    assert sub_cm.shape == (4, 4)
    assert result["tops_internal_errors"] == 12  # 5 + 4 + 3
    assert result["tops_error_share"] > 0.0
    assert 0.0 <= result["tops_intra_accuracy"] <= 1.0
    assert len(result["confused_pairs"]) > 0
    top_pair = result["confused_pairs"][0]
    assert top_pair["true_class"] == "Shirt"
    assert top_pair["predicted_class"] == "T-shirt/top"
    assert top_pair["error_count"] == 5


def test_metrics_persistence(tmp_path):
    """Verify metrics can be cleanly persisted to and loaded from JSON."""
    metrics_file = tmp_path / "metrics.json"
    dummy_metrics = {
        "project_id": "DL-09",
        "metrics": {"test_accuracy": 0.915, "macro_f1": 0.910},
    }
    save_metrics(dummy_metrics, str(metrics_file))

    assert os.path.exists(metrics_file)
    with open(metrics_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["project_id"] == "DL-09"
    assert loaded["metrics"]["test_accuracy"] == 0.915


def test_pdf_report_compilation(tmp_path):
    """Verify ReportLab PDF compilation succeeds and produces valid document > 5000 bytes."""
    output_pdf = str(tmp_path / "test_report.pdf")
    metrics = {
        "test_accuracy": 0.9142,
        "macro_f1": 0.9105,
        "top_3_accuracy": 0.9850,
        "tops_confusion_rate": 0.1240,
    }
    pdf_path = compile_project_report(
        project_id="DL-09",
        project_title="Fashion Item Classification",
        metrics=metrics,
        output_pdf_path=output_pdf,
    )
    assert os.path.exists(pdf_path)
    file_size = os.path.getsize(pdf_path)
    assert file_size > 5000, f"Generated PDF size {file_size} is unexpectedly small"


def test_streamlit_app_import():
    """Verify app.py imports cleanly without syntax or top-level execution errors."""
    orig_dir = os.getcwd()
    try:
        os.chdir(project_dir)
        import app
        assert hasattr(app, "__file__")
    finally:
        os.chdir(orig_dir)
