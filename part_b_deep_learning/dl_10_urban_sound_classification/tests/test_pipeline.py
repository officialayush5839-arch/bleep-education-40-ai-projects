"""
tests/test_pipeline.py
Tier 3 & 4 Tests: Mini-epoch training, metric evaluation, and PDF report compilation for DL-10.
"""

import torch
import torch.nn as nn
from pathlib import Path
from src.data_loader import load_dataset
from src.preprocessor import create_dataloaders
from src.model import DeepClassifier
from src.evaluate import train_epoch, evaluate_model
from src.utils import compile_project_report, set_seed


def test_mini_training_and_evaluation_pipeline(tmp_path):
    set_seed(42)
    device = torch.device("cpu")
    train_ds, test_ds = load_dataset(n_samples=40)
    train_loader, test_loader = create_dataloaders(train_ds, test_ds, batch_size=8)

    model = DeepClassifier().to(device)
    criterion = nn.MSELoss() if "audio_spectrogram" == "sequence_lstm" else nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    loss, acc = train_epoch(model, train_loader, criterion, optimizer, device)
    assert loss > 0.0

    metrics = evaluate_model(model, test_loader, device)
    assert len(metrics) >= 1


def test_report_compilation_workflow(tmp_path):
    pdf_path = tmp_path / "project_report.pdf"
    md_path = tmp_path / "project_report.md"
    success = compile_project_report(
        project_id="DL-10",
        project_title="Urban Sound Classification",
        domain="Smart Cities / Audio",
        metrics_path="results/metrics.json",
        output_pdf=str(pdf_path),
        output_md=str(md_path)
    )
    assert success is True
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 2000
