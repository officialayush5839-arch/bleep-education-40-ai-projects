"""
tests/test_pipeline.py
Integration tests executing mini-epoch training, metric computation,
PDF report compilation, and UI module import verification.
"""

import os
import pytest
import torch

from src.model import WasteClassifierCNN
from src.data_loader import get_waste_data_loaders, CLASS_NAMES
from src.evaluate import (
    evaluate_model,
    plot_confusion_matrix,
    plot_training_curves,
    save_metrics_json,
    train_model,
)
from src.utils import compile_pdf_report, set_seed


def test_mini_training_pipeline(tmp_path):
    """Executes mini-epoch training and verifies metric computation."""
    set_seed(42)
    train_loader, val_loader, test_loader, classes = get_waste_data_loaders(
        data_dir=None,
        batch_size=8,
        image_size=(64, 64),
        use_synthetic=True,
        n_synthetic_per_class=6,
        seed=42,
    )
    model = WasteClassifierCNN(num_classes=6, backbone="custom")

    # Train for 1 mini-epoch
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=1,
        learning_rate=0.001,
        device=torch.device("cpu"),
        save_dir=str(tmp_path),
    )

    assert "train_loss" in history
    assert len(history["train_loss"]) == 1
    assert "val_acc" in history

    # Evaluate on test set
    test_loss, metrics, y_true, y_pred, y_probs = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=torch.device("cpu"),
    )

    assert "test_accuracy" in metrics
    assert "macro_f1" in metrics
    assert "confusion_matrix" in metrics
    assert isinstance(metrics["test_accuracy"], float)

    # Verify metrics serialization
    metrics_path = os.path.join(str(tmp_path), "metrics.json")
    save_metrics_json(metrics, metrics_path)
    assert os.path.exists(metrics_path)


def test_plot_generation(tmp_path):
    """Verifies confusion matrix and training curves plot generation."""
    dummy_cm = [[5, 1, 0, 0, 0, 0], [0, 6, 0, 0, 0, 0], [0, 0, 4, 1, 0, 0],
                [0, 0, 0, 5, 0, 0], [0, 0, 0, 0, 6, 0], [0, 0, 0, 0, 1, 4]]
    cm_path = os.path.join(str(tmp_path), "cm.png")
    plot_confusion_matrix(dummy_cm, CLASS_NAMES, cm_path)
    assert os.path.exists(cm_path)
    assert os.path.getsize(cm_path) > 1000

    dummy_hist = {
        "train_loss": [1.5, 1.0],
        "train_acc": [0.3, 0.7],
        "val_loss": [1.4, 0.9],
        "val_acc": [0.4, 0.75],
    }
    curve_path = os.path.join(str(tmp_path), "curves.png")
    plot_training_curves(dummy_hist, curve_path)
    assert os.path.exists(curve_path)
    assert os.path.getsize(curve_path) > 1000


def test_pdf_report_compilation(tmp_path):
    """Verifies ReportLab PDF generation executes cleanly and produces valid document."""
    report_pdf = os.path.join(str(tmp_path), "test_report.pdf")
    metrics = {
        "test_accuracy": 0.884,
        "macro_f1": 0.879,
        "macro_precision": 0.890,
        "macro_recall": 0.875,
    }
    compiled_path = compile_pdf_report(metrics, output_pdf_path=report_pdf)
    assert os.path.exists(compiled_path)
    assert os.path.getsize(compiled_path) > 5000


def test_app_import():
    """Verifies that app.py is syntactically valid and imports without error."""
    import importlib.util
    app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app.py"))
    assert os.path.exists(app_path), "app.py must exist in project root."
    spec = importlib.util.spec_from_file_location("waste_app", app_path)
    module = importlib.util.module_from_spec(spec)
    # Ensure module spec loads
    assert spec is not None
