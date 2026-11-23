"""
tests/test_pipeline.py
Tier 3 & 4 Tests: End-to-end integration, citation traceability, 15-query evaluation, and PDF report generation.
"""

import os
import json
import pytest
from pathlib import Path
from src.data_loader import DataLoader
from src.chunker import RecursiveChunker, Chunk
from src.vectorstore import VectorStore
from src.rag_pipeline import RAGPipeline
from src.evaluate import RAGEvaluator
from src.utils import compile_project_report


def test_end_to_end_ingest_index_query(tmp_path):
    doc_path = tmp_path / "policy.md"
    doc_path.write_text("# Enterprise Benefits\n\n## 401k Matching\n**Page**: 1\nThe company matches 100% up to 6%.", encoding="utf-8")

    loader = DataLoader()
    docs = loader.load_file(str(doc_path))

    chunker = RecursiveChunker()
    chunks = chunker.chunk_documents(docs)

    vstore = VectorStore(score_threshold=0.10)
    vstore.add_chunks(chunks)

    pipeline = RAGPipeline(vectorstore=vstore, score_threshold=0.10)
    resp = pipeline.query("What is the 401k matching percentage?")

    assert not resp.is_abstained
    assert "100%" in resp.answer
    assert len(resp.citations) >= 1
    assert "[Doc: Enterprise Benefits, Page: 1, Section: 401k Matching]" in resp.citations[0]


def test_citation_format_traceability():
    vstore = VectorStore(score_threshold=0.10)
    c = Chunk(text="Tuition reimbursement is up to $5250 annually.", metadata={"title": "Benefits Guide", "page": 3, "section": "Tuition"})
    vstore.add_chunks([c])

    pipeline = RAGPipeline(vectorstore=vstore, score_threshold=0.10)
    resp = pipeline.query("How much tuition reimbursement?")
    assert len(resp.citations) >= 1
    assert resp.citations[0] == "[Doc: Benefits Guide, Page: 3, Section: Tuition]"


def test_vectorstore_save_and_load(tmp_path):
    vstore = VectorStore()
    c = Chunk(text="Sample knowledge text.", metadata={"title": "T", "page": 1, "section": "S"})
    vstore.add_chunks([c])

    save_dir = str(tmp_path / "vstore_cache")
    vstore.save(save_dir)

    new_vstore = VectorStore()
    success = new_vstore.load(save_dir)
    assert success is True
    assert len(new_vstore.chunks) == 1
    assert new_vstore.chunks[0].text == "Sample knowledge text."


def test_15_query_evaluation_benchmark():
    base_dir = Path(__file__).resolve().parents[1]
    raw_dir = base_dir / "data" / "raw_documents"
    if not raw_dir.exists():
        raw_dir = base_dir / "data" / "raw"
    loader = DataLoader()
    docs = loader.load_directory(str(raw_dir))
    assert len(docs) >= 4, "Must have at least 4 raw documents"

    chunker = RecursiveChunker(chunk_size=500, chunk_overlap=80)
    chunks = chunker.chunk_documents(docs)
    assert len(chunks) >= 8, "Must produce multiple indexed chunks"

    vstore = VectorStore(score_threshold=0.20)
    vstore.add_chunks(chunks)

    pipeline = RAGPipeline(vectorstore=vstore, score_threshold=0.20)
    evaluator = RAGEvaluator(pipeline)

    metrics_file = base_dir / "results" / "metrics.json"
    assert metrics_file.exists(), "results/metrics.json must exist"

    with open(metrics_file, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)

    query_set = [
        {
            "query": qr["query"],
            "ground_truth_answer": qr["expected"],
            "is_abstention": qr["is_abstention_expected"]
        }
        for qr in metrics_data["query_results"]
    ]

    metrics = evaluator.evaluate_query_set(query_set)
    assert metrics["total_queries"] == 15
    assert metrics["abstention_accuracy"] == 1.0, "Must achieve 100% abstention on unanswerable queries"
    assert metrics["citation_precision"] == 1.0, "Must cite sources for all answerable queries"
    assert metrics["overall_accuracy"] >= 0.85, "Must achieve high semantic accuracy"


def test_report_compilation_workflow(tmp_path):
    metrics_path = tmp_path / "metrics.json"
    pdf_path = tmp_path / "project_report.pdf"
    md_path = tmp_path / "project_report.md"

    metrics_path.write_text(json.dumps({
        "overall_accuracy": 0.96,
        "citation_precision": 1.0,
        "abstention_accuracy": 1.0,
        "mean_latency_ms": 12.5
    }), encoding="utf-8")

    success = compile_project_report(
        project_id="GEN-06",
        project_title="Agricultural Knowledge Assistant",
        domain="AgriTech",
        metrics_path=str(metrics_path),
        output_pdf=str(pdf_path),
        output_md=str(md_path)
    )

    assert success is True
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 2000
    assert md_path.exists()
