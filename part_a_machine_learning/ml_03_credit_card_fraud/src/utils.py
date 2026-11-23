"""
src/utils.py
Common utility and PDF compilation bridge for ML-03.
"""

import json
from pathlib import Path
from typing import Dict, Any


def compile_project_report(
    project_id: str = "ML-03",
    project_title: str = "Credit Card Fraud Detection",
    domain: str = "FinTech",
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
            "model": "Supervised Learning Architecture with Clean Zero-Leakage Pipeline",
            "target": "Is_Fraud"
        },
        metrics=metrics_data
    )
    return True
