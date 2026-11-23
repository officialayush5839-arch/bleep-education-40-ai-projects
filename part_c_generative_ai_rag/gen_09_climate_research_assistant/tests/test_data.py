"""
tests/test_data.py
Tier 1 Tests: Multi-format data loading, chunking boundaries, and metadata inheritance.
"""

import pytest
from pathlib import Path
from src.data_loader import DataLoader, Document
from src.chunker import RecursiveChunker, Chunk


def test_load_single_markdown_file(tmp_path):
    md_file = tmp_path / "test_doc.md"
    md_file.write_text("# Test Policy\n\n## Leave Rules\n**Page**: 2\nEmployees get 25 days leave.", encoding="utf-8")

    loader = DataLoader(domain="test")
    docs = loader.load_file(str(md_file))
    assert len(docs) >= 1
    assert docs[0].title == "Test Policy"
    assert docs[0].page == 2
    assert "25 days" in docs[0].text


def test_load_single_text_file(tmp_path):
    txt_file = tmp_path / "test_doc.txt"
    txt_file.write_text("Paragraph one content.\n\nParagraph two content.", encoding="utf-8")

    loader = DataLoader(domain="test")
    docs = loader.load_file(str(txt_file))
    assert len(docs) == 2
    assert docs[0].metadata["doc_id"] == "TEST_DOC"


def test_load_directory(tmp_path):
    (tmp_path / "doc1.md").write_text("# Doc 1\nContent 1", encoding="utf-8")
    (tmp_path / "doc2.txt").write_text("Content 2", encoding="utf-8")

    loader = DataLoader()
    docs = loader.load_directory(str(tmp_path))
    assert len(docs) >= 2


def test_chunker_size_and_overlap():
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
    long_text = "This is a sentence. " * 20
    doc = Document(text=long_text, metadata={"doc_id": "D1", "title": "T1", "page": 1, "section": "S1"})

    chunks = chunker.chunk_document(doc)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 140


def test_chunker_metadata_inheritance():
    chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
    doc = Document(text="Sentence A. Sentence B. Sentence C.", metadata={"doc_id": "DOC_99", "title": "Spec 99", "page": 3, "section": "Sec 4"})

    chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 1
    for c in chunks:
        assert c.doc_id == "DOC_99"
        assert c.title == "Spec 99"
        assert c.page == 3
        assert c.section == "Sec 4"
        assert c.format_citation() == "[Doc: Spec 99, Page: 3, Section: Sec 4]"
