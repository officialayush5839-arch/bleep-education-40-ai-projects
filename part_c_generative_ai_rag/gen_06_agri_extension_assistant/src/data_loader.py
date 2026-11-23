"""
src/data_loader.py
Universal Multi-Format Document Ingestion Engine for Part C RAG.
Supports Markdown (.md), Plain Text (.txt), HTML (.html), and PDF (.pdf).
Extracts structured metadata: doc_id, title, page, section, text, source_path.
"""

import os
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Document:
    """Standardized Document Data Model."""
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def doc_id(self) -> str:
        return self.metadata.get("doc_id", "DOC_UNKNOWN")

    @property
    def title(self) -> str:
        return self.metadata.get("title", "Untitled Document")

    @property
    def page(self) -> int:
        return self.metadata.get("page", 1)

    @property
    def section(self) -> str:
        return self.metadata.get("section", "General")


class SimpleHTMLTextExtractor(HTMLParser):
    """Pure-Python HTML tag stripper and heading extractor."""
    def __init__(self):
        super().__init__()
        self.text_parts: List[str] = []
        self.title: str = "HTML Document"
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "title":
            self._in_title = True

    def handle_endtag(self, tag):
        if tag.lower() == "title":
            self._in_title = False

    def handle_data(self, data):
        clean = data.strip()
        if self._in_title and clean:
            self.title = clean
        elif clean:
            self.text_parts.append(clean)

    def get_text(self) -> str:
        return "\n".join(self.text_parts)


class DataLoader:
    """Multi-format document loader with provenance extraction."""

    SUPPORTED_EXTENSIONS = {".md", ".txt", ".html", ".htm", ".pdf"}

    def __init__(self, domain: str = "general"):
        self.domain = domain

    def load_file(self, file_path: str) -> List[Document]:
        """Load and parse a single file into one or more Document instances."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            return []

        if ext == ".md":
            return self._parse_markdown(path)
        elif ext == ".txt":
            return self._parse_text(path)
        elif ext in {".html", ".htm"}:
            return self._parse_html(path)
        elif ext == ".pdf":
            return self._parse_pdf(path)
        return []

    def _parse_markdown(self, path: Path) -> List[Document]:
        content = path.read_text(encoding="utf-8", errors="replace")
        if not content.strip():
            return []

        title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        doc_title = title_match.group(1).strip() if title_match else path.stem.replace("_", " ").title()
        doc_id = path.stem.upper()

        sections = re.split(r"(?m)^(?=##\s+)", content)
        documents = []

        for idx, sec in enumerate(sections):
            sec_clean = sec.strip()
            if not sec_clean:
                continue

            sec_title_match = re.search(r"^##\s+(.+)$", sec_clean, re.MULTILINE)
            sec_title = sec_title_match.group(1).strip() if sec_title_match else "Overview"

            page_match = re.search(r"\*\*Page\*\*:\s*(\d+)", sec_clean)
            page_num = int(page_match.group(1)) if page_match else (idx + 1)

            body_text = re.sub(r"^#+.*$", "", sec_clean, flags=re.MULTILINE).strip()
            body_text = re.sub(r"\*\*Page\*\*:\s*\d+\s*\|?\s*\*\*Section\*\*:[^\n]+", "", body_text).strip()

            if body_text:
                documents.append(Document(
                    text=body_text,
                    metadata={
                        "doc_id": doc_id,
                        "title": doc_title,
                        "page": page_num,
                        "section": sec_title,
                        "source_path": str(path.resolve()),
                        "domain": self.domain,
                    }
                ))

        if not documents and content.strip():
            documents.append(Document(
                text=content.strip(),
                metadata={
                    "doc_id": doc_id,
                    "title": doc_title,
                    "page": 1,
                    "section": "General",
                    "source_path": str(path.resolve()),
                    "domain": self.domain,
                }
            ))

        return documents

    def _parse_text(self, path: Path) -> List[Document]:
        content = path.read_text(encoding="utf-8", errors="replace").strip()
        if not content:
            return []

        doc_title = path.stem.replace("_", " ").title()
        doc_id = path.stem.upper()

        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        documents = []
        for i, para in enumerate(paragraphs, 1):
            documents.append(Document(
                text=para,
                metadata={
                    "doc_id": doc_id,
                    "title": doc_title,
                    "page": 1,
                    "section": f"Paragraph {i}",
                    "source_path": str(path.resolve()),
                    "domain": self.domain,
                }
            ))
        return documents

    def _parse_html(self, path: Path) -> List[Document]:
        raw_html = path.read_text(encoding="utf-8", errors="replace")
        parser = SimpleHTMLTextExtractor()
        parser.feed(raw_html)
        text = parser.get_text()
        if not text:
            return []

        return [Document(
            text=text,
            metadata={
                "doc_id": path.stem.upper(),
                "title": parser.title or path.stem.title(),
                "page": 1,
                "section": "Main Content",
                "source_path": str(path.resolve()),
                "domain": self.domain,
            }
        )]

    def _parse_pdf(self, path: Path) -> List[Document]:
        documents = []
        try:
            import pymupdf
            doc = pymupdf.open(str(path))
            doc_id = path.stem.upper()
            title = doc.metadata.get("title") or path.stem.replace("_", " ").title()

            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text().strip()
                if page_text:
                    documents.append(Document(
                        text=page_text,
                        metadata={
                            "doc_id": doc_id,
                            "title": title,
                            "page": page_num + 1,
                            "section": f"Page {page_num + 1}",
                            "source_path": str(path.resolve()),
                            "domain": self.domain,
                        }
                    ))
            doc.close()
        except ImportError:
            content = f"PDF content fallback for {path.name}"
            documents.append(Document(
                text=content,
                metadata={
                    "doc_id": path.stem.upper(),
                    "title": path.stem.title(),
                    "page": 1,
                    "section": "PDF Fallback",
                    "source_path": str(path.resolve()),
                    "domain": self.domain,
                }
            ))
        return documents

    def load_directory(self, dir_path: str) -> List[Document]:
        """Scan directory and parse all supported document types."""
        path = Path(dir_path)
        if not path.exists() or not path.is_dir():
            return []

        all_docs = []
        for file_p in sorted(path.rglob("*")):
            if file_p.is_file() and file_p.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                all_docs.extend(self.load_file(str(file_p)))
        return all_docs
