"""
src/chunker.py
Universal Semantic Chunker with Metadata Inheritance.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from src.data_loader import Document


@dataclass
class Chunk:
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        return self.metadata.get("chunk_id", "CHK_UNKNOWN")

    @property
    def doc_id(self) -> str:
        return self.metadata.get("doc_id", "DOC_UNKNOWN")

    @property
    def title(self) -> str:
        return self.metadata.get("title", "Untitled Document")

    @property
    def page(self) -> int:
        return int(self.metadata.get("page", 1))

    @property
    def section(self) -> str:
        return self.metadata.get("section", "General")

    def format_citation(self) -> str:
        return f"[Doc: {self.title}, Page: {self.page}, Section: {self.section}]"


class RecursiveChunker:
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 80,
        separators: Optional[List[str]] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

    def chunk_document(self, doc: Document) -> List[Chunk]:
        text = doc.text.strip()
        if not text:
            return []

        raw_splits = self._split_text_recursive(text, self.separators)
        chunks = []

        doc_id = getattr(doc, "doc_id", doc.metadata.get("doc_id", "DOC"))
        page = getattr(doc, "page", doc.metadata.get("page", 1))

        char_offset = 0
        for i, split_text in enumerate(raw_splits):
            meta = dict(doc.metadata)
            meta["chunk_id"] = f"{doc_id}_p{page}_c{i+1}"
            meta["chunk_index"] = i + 1
            meta["char_start"] = char_offset
            meta["char_end"] = char_offset + len(split_text)
            char_offset += max(1, len(split_text) - self.chunk_overlap)

            chunks.append(Chunk(text=split_text, metadata=meta))

        return chunks

    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks

    def _split_text_recursive(self, text: str, separators: List[str]) -> List[str]:
        final_chunks = []
        separator = separators[-1]
        new_separators = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        splits = text.split(separator) if separator != "" else list(text)

        good_splits = []
        for s in splits:
            if separator and s != splits[0]:
                s = separator + s
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged = self._merge_splits(good_splits, separator)
                    final_chunks.extend(merged)
                    good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = self._split_text_recursive(s, new_separators)
                    final_chunks.extend(other_info)

        if good_splits:
            merged = self._merge_splits(good_splits, separator)
            final_chunks.extend(merged)

        return final_chunks

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        docs = []
        current_doc: List[str] = []
        total = 0

        for d in splits:
            _len = len(d)
            if total + _len >= self.chunk_size:
                if total > self.chunk_size:
                    pass
                if current_doc:
                    doc = "".join(current_doc)
                    if doc.strip():
                        docs.append(doc.strip())
                    while total > self.chunk_overlap or (total + _len > self.chunk_size and total > 0):
                        total -= len(current_doc[0])
                        current_doc = current_doc[1:]
            current_doc.append(d)
            total += _len

        doc = "".join(current_doc)
        if doc.strip():
            docs.append(doc.strip())
        return docs
