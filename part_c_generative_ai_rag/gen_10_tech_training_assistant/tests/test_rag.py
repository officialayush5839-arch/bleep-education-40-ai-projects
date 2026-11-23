"""
tests/test_rag.py
Tier 2 Tests: Boundary conditions, zero-context queries, thresholding, and abstention.
"""

import pytest
from src.chunker import RecursiveChunker, Chunk
from src.vectorstore import VectorStore
from src.rag_pipeline import RAGPipeline, ABSTENTION_MESSAGE


def test_empty_document_handling():
    chunker = RecursiveChunker()
    vstore = VectorStore()
    assert chunker.chunk_documents([]) == []
    assert vstore.add_chunks([]) == 0
    assert vstore.search("query") == []


def test_chunker_short_text():
    chunker = RecursiveChunker(chunk_size=500, chunk_overlap=50)
    doc = type("Doc", (), {"text": "Brief text.", "metadata": {"doc_id": "D1", "title": "T1", "page": 1, "section": "S"}})()
    res = chunker.chunk_document(doc)
    assert len(res) == 1
    assert res[0].text == "Brief text."


def test_out_of_scope_abstention():
    vstore = VectorStore(score_threshold=0.20)
    c1 = Chunk(text="Employees are entitled to 25 days annual leave.", metadata={"title": "Handbook", "page": 1, "section": "Leave"})
    vstore.add_chunks([c1])

    pipeline = RAGPipeline(vectorstore=vstore, score_threshold=0.20)
    resp = pipeline.query("What is the nuclear warp drive maintenance frequency?")
    assert resp.is_abstained is True
    assert ABSTENTION_MESSAGE in resp.answer
    assert len(resp.citations) == 0


def test_threshold_filtering():
    vstore = VectorStore(score_threshold=0.50)
    c1 = Chunk(text="Company health insurance covers dental procedures.", metadata={"title": "Benefits", "page": 2, "section": "Dental"})
    vstore.add_chunks([c1])

    results = vstore.search("sports equipment warranty", threshold=0.50)
    assert len(results) == 0


def test_mandatory_disclaimer_presence():
    vstore = VectorStore(score_threshold=0.10)
    c1 = Chunk(text="Administer 50mg orally twice daily with meals.", metadata={"title": "Clinical Guide", "page": 1, "section": "Dosage"})
    vstore.add_chunks([c1])

    disclaimer = "NOTICE: Research tool only. Not medical advice."
    pipeline = RAGPipeline(vectorstore=vstore, disclaimer=disclaimer, score_threshold=0.10)

    resp = pipeline.query("What is the oral dose?")
    assert not resp.is_abstained
    assert disclaimer in resp.answer
