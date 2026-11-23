"""
common_utils/tests/test_report_generator.py
Unit tests for common_utils.report_generator verifying 12-section technical
report generation, NumberedCanvas pagination, robustness against edge cases,
and Bleep 100-mark rubric content compliance.
"""

import os
import tempfile
import pytest
from PIL import Image

try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz
    except ImportError:
        fitz = None

from common_utils.report_generator import (
    generate_pdf_report,
    STANDARD_SECTIONS,
    NumberedCanvas,
)


@pytest.fixture
def sample_metadata():
    return {
        "project_id": "ML-01",
        "project_title": "Predictive Maintenance & Equipment Failure",
        "domain": "Machine Learning",
        "summary": {
            "author": "Bleep AI Engineering Team",
            "organization": "Bleep Education",
            "target": "Machine failure (binary 0/1)",
            "model": "Random Forest / LightGBM Ensemble",
        },
        "metrics": {
            "accuracy": 0.9840,
            "recall": 0.8850,
            "f1_score": 0.9020,
            "roc_auc": 0.9760,
        },
        "sections": {
            1: "Custom Executive Summary for ML-01 testing.",
            2: "Detailed dataset processing with zero leakage.",
        }
    }


def test_generate_pdf_report_basic(tmp_path, sample_metadata):
    """Verifies basic PDF generation produces a valid non-empty PDF file."""
    output_pdf = tmp_path / "test_report.pdf"
    
    result_path = generate_pdf_report(
        project_id=sample_metadata["project_id"],
        project_title=sample_metadata["project_title"],
        domain=sample_metadata["domain"],
        output_path=str(output_pdf),
        summary=sample_metadata["summary"],
        metrics=sample_metadata["metrics"],
        sections=sample_metadata["sections"]
    )
    
    assert os.path.exists(result_path)
    assert os.path.isabs(result_path)
    assert os.path.getsize(result_path) > 2048
    
    # Verify PDF magic signature
    with open(result_path, "rb") as f:
        header = f.read(5)
    assert header == b"%PDF-"


def test_generate_pdf_report_all_domains(tmp_path):
    """Verifies report generation across all 4 domains (ML, DL, GenAI, Agentic)."""
    domains = [
        ("ML-01", "Predictive Maintenance", "Machine Learning"),
        ("DL-01", "Automated Waste Classification", "Deep Learning"),
        ("GEN-01", "Enterprise HR Policy Assistant", "Generative AI / RAG"),
        ("AG-01", "Agricultural Advisory Multi-Agent", "Agentic AI"),
    ]
    
    for pid, title, dom in domains:
        pdf_path = tmp_path / f"{pid.lower()}_report.pdf"
        res = generate_pdf_report(
            project_id=pid,
            project_title=title,
            domain=dom,
            output_path=str(pdf_path),
            metrics={"primary_metric": 0.95}
        )
        assert os.path.exists(res)
        assert os.path.getsize(res) > 2048


def test_generate_pdf_report_empty_metrics(tmp_path):
    """Verifies that empty or None metrics do not crash the generator."""
    pdf_path = tmp_path / "empty_metrics.pdf"
    res = generate_pdf_report(
        project_id="ML-02",
        project_title="Patient Readmission Risk",
        domain="Machine Learning",
        output_path=str(pdf_path),
        metrics={}
    )
    assert os.path.exists(res)
    assert os.path.getsize(res) > 2048


def test_generate_pdf_report_empty_sections(tmp_path):
    """Verifies that empty sections dict falls back to all 12 default sections."""
    pdf_path = tmp_path / "empty_sections.pdf"
    res = generate_pdf_report(
        project_id="DL-03",
        project_title="Chest X-Ray Pneumonia",
        domain="Deep Learning",
        output_path=str(pdf_path),
        sections={}
    )
    assert os.path.exists(res)
    assert os.path.getsize(res) > 2048


def test_generate_pdf_report_special_characters(tmp_path):
    """Verifies XML special characters in text strings are safely escaped."""
    pdf_path = tmp_path / "special_chars.pdf"
    special_text = "Testing mathematical conditions: Loss < 0.05 & Accuracy > 95% with 'quotes' & \"double\"."
    res = generate_pdf_report(
        project_id="GEN-04",
        project_title="Legal Document Assistant",
        domain="Generative AI / RAG",
        output_path=str(pdf_path),
        sections={1: special_text}
    )
    assert os.path.exists(res)
    assert os.path.getsize(res) > 2048


def test_generate_pdf_report_unicode_strings(tmp_path):
    """Verifies mathematical symbols and unicode are handled without encoding crash."""
    pdf_path = tmp_path / "unicode_report.pdf"
    unicode_text = "Statistical parameters: Mean μ = 42.0, Std σ = 1.5, Tolerance ±0.01, Threshold ≥ 0.90."
    res = generate_pdf_report(
        project_id="ML-08",
        project_title="Solar Power Forecasting",
        domain="Machine Learning",
        output_path=str(pdf_path),
        sections={1: unicode_text}
    )
    assert os.path.exists(res)
    assert os.path.getsize(res) > 2048


def test_generate_pdf_report_nested_output_directory(tmp_path):
    """Verifies automatic creation of deeply nested parent directories."""
    deep_path = tmp_path / "deeply" / "nested" / "reports" / "report.pdf"
    res = generate_pdf_report(
        project_id="AG-05",
        project_title="Travel Planner Agent",
        domain="Agentic AI",
        output_path=str(deep_path)
    )
    assert os.path.exists(res)
    assert os.path.getsize(res) > 2048


def test_generate_pdf_report_chart_embedding(tmp_path):
    """Verifies embedding a valid PNG image and handling a non-existent chart path."""
    # 1. Valid image
    chart_file = tmp_path / "test_chart.png"
    img = Image.new("RGB", (300, 200), color=(73, 109, 137))
    img.save(chart_file)
    
    pdf_with_chart = tmp_path / "report_with_chart.pdf"
    res1 = generate_pdf_report(
        project_id="ML-03",
        project_title="Credit Card Fraud",
        domain="Machine Learning",
        output_path=str(pdf_with_chart),
        summary={"chart_path": str(chart_file)}
    )
    assert os.path.exists(res1)
    
    # 2. Missing image path
    pdf_missing_chart = tmp_path / "report_missing_chart.pdf"
    res2 = generate_pdf_report(
        project_id="ML-03",
        project_title="Credit Card Fraud",
        domain="Machine Learning",
        output_path=str(pdf_missing_chart),
        summary={"chart_path": str(tmp_path / "non_existent_chart.png")}
    )
    assert os.path.exists(res2)


@pytest.mark.skipif(fitz is None, reason="PyMuPDF not installed")
def test_generate_pdf_report_content_verification(tmp_path, sample_metadata):
    """Extracts text using PyMuPDF and verifies presence of all 12 section titles and metadata."""
    output_pdf = tmp_path / "content_verify.pdf"
    generate_pdf_report(
        project_id=sample_metadata["project_id"],
        project_title=sample_metadata["project_title"],
        domain=sample_metadata["domain"],
        output_path=str(output_pdf),
        summary=sample_metadata["summary"],
        metrics=sample_metadata["metrics"]
    )
    
    doc = fitz.open(str(output_pdf))
    full_text = ""
    for page in doc:
        full_text += page.get_text() + "\n"
    doc.close()
    
    # Verify metadata in text
    assert sample_metadata["project_id"] in full_text
    assert sample_metadata["domain"] in full_text
    assert "Accuracy" in full_text or "accuracy" in full_text
    
    # Verify presence of all 12 section titles
    for sec_num, sec_title in STANDARD_SECTIONS:
        # Title or section number must be present in the document
        assert f"Section {sec_num}:" in full_text


@pytest.mark.skipif(fitz is None, reason="PyMuPDF not installed")
def test_generate_pdf_report_rubric_checklist(tmp_path):
    """Verifies Section 12 contains Bleep 100-mark rubric criteria and Part E checklist."""
    output_pdf = tmp_path / "rubric_verify.pdf"
    generate_pdf_report(
        project_id="AG-10",
        project_title="Educational Content Multi-Agent",
        domain="Agentic AI",
        output_path=str(output_pdf)
    )
    
    doc = fitz.open(str(output_pdf))
    full_text = ""
    for page in doc:
        full_text += page.get_text() + "\n"
    doc.close()
    
    # Rubric keywords
    assert "100 / 100 Marks" in full_text or "100-Mark Rubric" in full_text
    assert "Problem Understanding" in full_text
    assert "Technical Implementation" in full_text
    assert "System Evaluation" in full_text
    
    # Deliverables checklist keywords
    assert "README.md" in full_text
    assert "requirements.txt" in full_text
    assert "app.py" in full_text
    assert "VERIFIED" in full_text


def test_generate_pdf_report_invalid_inputs():
    """Verifies that invalid or empty required arguments raise ValueError."""
    with pytest.raises(ValueError):
        generate_pdf_report(project_id="", project_title="Title", domain="Domain", output_path="out.pdf")
    with pytest.raises(ValueError):
        generate_pdf_report(project_id="ID", project_title="", domain="Domain", output_path="out.pdf")
    with pytest.raises(ValueError):
        generate_pdf_report(project_id="ID", project_title="Title", domain="", output_path="out.pdf")


@pytest.mark.skipif(fitz is None, reason="PyMuPDF not installed")
def test_multipage_running_header_text_verification(tmp_path):
    """Verifies pages >= 2 contain dynamic project_id and domain in running headers (not shadowed default)."""
    output_pdf = tmp_path / "multipage_header.pdf"
    project_id = "DL-05"
    domain = "Deep Learning"
    
    generate_pdf_report(
        project_id=project_id,
        project_title="Traffic Sign Classification",
        domain=domain,
        output_path=str(output_pdf),
    )
    
    doc = fitz.open(str(output_pdf))
    assert len(doc) >= 2, f"Document must be multi-page (got {len(doc)} pages)"
    
    # Page 1 must not have running header
    p1_text = doc[0].get_text()
    assert f"Bleep Education Technical Report | {project_id}" not in p1_text
    
    # All subsequent pages (pages >= 2) must include project-specific running header
    for page_idx in range(1, len(doc)):
        p_text = doc[page_idx].get_text()
        assert f"Bleep Education Technical Report | {project_id}" in p_text, f"Missing header on page {page_idx+1}"
        assert f"{domain} Domain" in p_text, f"Missing domain header on page {page_idx+1}"
        assert "AI Project" not in p_text, "Shadowed default 'AI Project' detected in running header"
        assert "Universal AI" not in p_text, "Shadowed default 'Universal AI' detected in running header"
    doc.close()


@pytest.mark.skipif(fitz is None, reason="PyMuPDF not installed")
def test_corrupted_image_and_directory_chart_path(tmp_path):
    """Verifies corrupted image or directory as chart_path falls back to callout placeholder."""
    # 1. Corrupted image file
    corrupt_file = tmp_path / "corrupt_chart.png"
    with open(corrupt_file, "wb") as f:
        f.write(b"NOT_A_VALID_PNG_OR_IMAGE_DATA_1234567890")
        
    pdf_corrupt = tmp_path / "corrupt_chart_report.pdf"
    res1 = generate_pdf_report(
        project_id="ML-04",
        project_title="Customer Churn Prediction",
        domain="Machine Learning",
        output_path=str(pdf_corrupt),
        summary={"chart_path": str(corrupt_file)}
    )
    assert os.path.exists(res1)
    doc1 = fitz.open(str(res1))
    full_text1 = "\n".join(p.get_text() for p in doc1)
    doc1.close()
    assert "Evaluation Chart Notice" in full_text1
    assert "Chart image unavailable or invalid" in full_text1

    # 2. Directory passed as chart_path
    dir_chart = tmp_path / "chart_directory"
    dir_chart.mkdir()
    pdf_dir = tmp_path / "dir_chart_report.pdf"
    res2 = generate_pdf_report(
        project_id="ML-04",
        project_title="Customer Churn Prediction",
        domain="Machine Learning",
        output_path=str(pdf_dir),
        summary={"chart_path": str(dir_chart)}
    )
    assert os.path.exists(res2)
    doc2 = fitz.open(str(res2))
    full_text2 = "\n".join(p.get_text() for p in doc2)
    doc2.close()
    assert "Evaluation Chart Notice" in full_text2


@pytest.mark.skipif(fitz is None, reason="PyMuPDF not installed")
def test_non_string_keys_in_metrics_dict(tmp_path):
    """Verifies integer or non-string keys in metrics dict do not trigger AttributeError."""
    pdf_path = tmp_path / "non_string_metrics.pdf"
    non_str_metrics = {
        101: 0.9542,
        202: True,
        303: "Custom Score",
        ("custom", "key"): 0.8876
    }
    res = generate_pdf_report(
        project_id="GEN-02",
        project_title="Clinical Trial Matcher",
        domain="Generative AI / RAG",
        output_path=str(pdf_path),
        metrics=non_str_metrics
    )
    assert os.path.exists(res)
    assert os.path.getsize(res) > 2048
    
    doc = fitz.open(str(res))
    full_text = "\n".join(p.get_text() for p in doc)
    doc.close()
    assert "101" in full_text
    assert "202" in full_text


@pytest.mark.skipif(fitz is None, reason="PyMuPDF not installed")
def test_oversized_section_9_disclaimer(tmp_path):
    """Verifies oversized Section 9 text (> 10,000 characters) renders safely without LayoutError."""
    # Create disclaimer longer than 10,000 characters
    oversized_disclaimer = (
        "Disclaimer and Operational Notice: All automated predictive inferences generated by this model "
        "must be reviewed by accredited engineering personnel prior to actuation. "
    ) * 80
    assert len(oversized_disclaimer) > 10000
    
    pdf_path = tmp_path / "oversized_disclaimer.pdf"
    res = generate_pdf_report(
        project_id="AG-03",
        project_title="Industrial Equipment Dispatch Agent",
        domain="Agentic AI",
        output_path=str(pdf_path),
        sections={9: oversized_disclaimer}
    )
    assert os.path.exists(res)
    assert os.path.getsize(res) > 2048
    
    doc = fitz.open(str(res))
    full_text = "\n".join(p.get_text() for p in doc)
    doc.close()
    assert "Disclaimer and Operational Notice" in full_text
    assert "accredited engineering personnel" in full_text
