"""
tests/test_pipeline.py
Tier 3 & 4 Tests: 10-scenario evaluation benchmark and PDF report compilation for AG-02.
"""

from pathlib import Path
from src.evaluate import evaluate_agent_scenarios
from src.utils import compile_project_report


def test_10_scenario_evaluation_benchmark():
    metrics = evaluate_agent_scenarios()
    assert metrics["total_scenarios"] == 10
    assert metrics["passed_scenarios"] == 10
    assert metrics["success_rate"] == 1.0


def test_report_compilation_workflow(tmp_path):
    pdf_path = tmp_path / "project_report.pdf"
    md_path = tmp_path / "project_report.md"
    success = compile_project_report(
        project_id="AG-02",
        project_title="Customer Support Ticket Triage & Resolution",
        domain="Customer Service",
        metrics_path="results/metrics.json",
        output_pdf=str(pdf_path),
        output_md=str(md_path)
    )
    assert success is True
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 2000
