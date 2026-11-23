"""
tests/test_pipeline.py
Tier 3 & 4 Tests: End-to-end integration, metric computation, and PDF report compilation.
"""

from pathlib import Path
from src.data_loader import load_data
from src.preprocessor import preprocess_data
from src.model import MLModel
from src.evaluate import evaluate_predictions
from src.utils import compile_project_report


def test_end_to_end_pipeline_evaluation():
    df = load_data()
    prep = preprocess_data(df)
    model = MLModel()
    model.train(prep["X_train"], prep["y_train"])

    preds = model.predict(prep["X_test"])
    if "binary_classification" == "clustering":
        metrics = evaluate_predictions(prep["X_test"], preds)
        assert "silhouette_score" in metrics
    elif "binary_classification" == "regression":
        metrics = evaluate_predictions(prep["y_test"], preds)
        assert "r2_score" in metrics
    else:
        probs = model.predict_proba(prep["X_test"]) if hasattr(model, "predict_proba") else None
        metrics = evaluate_predictions(prep["y_test"], preds, probs)
        assert "accuracy" in metrics
        if "binary_classification" == "multiclass_classification":
            assert metrics["accuracy"] >= 0.10
        else:
            assert metrics["accuracy"] >= 0.50


def test_report_compilation_workflow(tmp_path):
    pdf_path = tmp_path / "project_report.pdf"
    md_path = tmp_path / "project_report.md"
    
    success = compile_project_report(
        project_id="ML-03",
        project_title="Credit Card Fraud Detection",
        domain="FinTech",
        metrics_path="results/metrics.json",
        output_pdf=str(pdf_path),
        output_md=str(md_path)
    )
    assert success is True
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 2000
