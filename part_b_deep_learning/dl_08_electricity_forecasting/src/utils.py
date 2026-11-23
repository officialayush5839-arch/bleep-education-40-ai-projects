"""
src/utils.py
Utilities and PDF report bridge for DL-08.
"""

import os
import json
import random
import torch
import numpy as np
from pathlib import Path


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compile_project_report(
    project_id: str = "DL-08",
    project_title: str = "Household Electricity Consumption Forecasting",
    domain: str = "Energy Management",
    metrics_path: str = "results/metrics.json",
    output_pdf: str = "reports/project_report.pdf",
    output_md: str = "reports/project_report.md"
) -> bool:
    try:
        from common_utils.report_generator import generate_pdf_report
    except ImportError:
        import sys
        sys.path.append(str(Path(__file__).resolve().parents[2]))
        from common_utils.report_generator import generate_pdf_report

    metrics_data = {}
    if Path(metrics_path).exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)

    generate_pdf_report(
        project_id=project_id,
        project_title=project_title,
        domain=domain,
        output_path=output_pdf,
        summary={
            "author": "Bleep AI Engineering Team",
            "organization": "Bleep Education",
            "model": "Deep Neural Network Architecture (sequence_lstm)",
            "target": "Household Electricity Consumption Forecasting"
        },
        metrics=metrics_data
    )
    return True
