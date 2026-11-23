"""
src/vectorstore.py
Robust Stemmed TF-IDF Vector Store with Deterministic Cosine Similarity Matching.
Guarantees clean word and morphology overlap while strictly rejecting out-of-scope queries.
"""

import os
import re
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.chunker import Chunk

STOP_WORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'of', 'to', 'in', 'for', 'with', 'on', 'at', 'by',
    'from', 'what', 'how', 'when', 'where', 'who', 'which', 'stated', 'regarding', 'mandatory',
    'guidelines', 'protocol', 'section', 'about', 'and', 'or', 'be', 'this', 'that', 'these', 'those'
}

def stem_word(word: str) -> str:
    w = word.lower().strip()
    for suff in ['ation', 'tions', 'tion', 'ties', 'ally', 'ely', 'ing', 'ies', 'es', 'ed', 'ly', 's']:
        if len(w) > len(suff) + 2 and w.endswith(suff):
            return w[:-len(suff)]
    return w

def preprocess_text(text: str) -> str:
    tokens = re.findall(r'[a-zA-Z0-9%]+', text.lower())
    filtered = [stem_word(t) for t in tokens if t not in STOP_WORDS]
    return " ".join(filtered) if filtered else text.lower()


class VectorStore:
    def __init__(self, score_threshold: float = 0.20):
        self.score_threshold = score_threshold
        self.chunks: List[Chunk] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.matrix: Optional[Any] = None

    def add_chunks(self, chunks: List[Chunk]) -> int:
        if not chunks:
            return 0
        self.chunks.extend(chunks)
        preprocessed_corpus = [preprocess_text(c.text) for c in self.chunks]
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True
        )
        self.matrix = self.vectorizer.fit_transform(preprocessed_corpus)
        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = 3,
        threshold: Optional[float] = None
    ) -> List[Tuple[Chunk, float]]:
        th = threshold if threshold is not None else self.score_threshold
        if not self.chunks or self.vectorizer is None or self.matrix is None:
            return []

        clean_query = preprocess_text(query.strip())
        if not clean_query.strip():
            return []

        query_vec = self.vectorizer.transform([clean_query])
        similarities = cosine_similarity(query_vec, self.matrix)[0]

        ranked_indices = np.argsort(similarities)[::-1]
        results = []

        for idx in ranked_indices:
            score = float(similarities[idx])
            if score < th:
                break
            results.append((self.chunks[idx], score))
            if len(results) >= top_k:
                break

        return results

    def save(self, directory_path: str) -> bool:
        target = Path(directory_path)
        target.mkdir(parents=True, exist_ok=True)
        with open(target / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        with open(target / "vectorizer.pkl", "wb") as f:
            pickle.dump(self.vectorizer, f)
        with open(target / "matrix.pkl", "wb") as f:
            pickle.dump(self.matrix, f)
        with open(target / "config.json", "w", encoding="utf-8") as f:
            json.dump({"score_threshold": self.score_threshold, "count": len(self.chunks)}, f)
        return True

    def load(self, directory_path: str) -> bool:
        target = Path(directory_path)
        if not (target / "chunks.pkl").exists():
            return False
        with open(target / "chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)
        with open(target / "vectorizer.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)
        with open(target / "matrix.pkl", "rb") as f:
            self.matrix = pickle.load(f)
        return True
