"""
src/rag_pipeline.py
Grounding, Deterministic Abstention, Citation Synthesis & LLM Integration.
Provides dual-mode generation:
1. Deterministic Extractive Synthesizer (100% offline, zero-hallucination).
2. Local Ollama LLM Generator (pluggable for interactive app execution).
"""

import time
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from src.chunker import Chunk
from src.vectorstore import VectorStore


ABSTENTION_MESSAGE = "I cannot answer this question based on the provided documentation."


@dataclass
class RAGResponse:
    """Standardized RAG Inference Output."""
    query: str
    answer: str
    citations: List[str] = field(default_factory=list)
    retrieved_chunks: List[Dict[str, Any]] = field(default_factory=list)
    is_abstained: bool = False
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    total_latency_ms: float = 0.0


class RAGPipeline:
    """
    Orchestrates Retrieval-Augmented Generation with strict citation grounding
    and deterministic out-of-scope abstention.
    """

    def __init__(
        self,
        vectorstore: VectorStore,
        domain_name: str = "Enterprise Domain",
        disclaimer: Optional[str] = None,
        score_threshold: float = 0.20,
        top_k: int = 3,
        use_ollama: bool = False,
        ollama_endpoint: str = "http://localhost:11434/api/generate",
        ollama_model: str = "qwen2.5:latest",
    ):
        self.vectorstore = vectorstore
        self.domain_name = domain_name
        self.disclaimer = disclaimer
        self.score_threshold = score_threshold
        self.top_k = top_k
        self.use_ollama = use_ollama
        self.ollama_endpoint = ollama_endpoint
        self.ollama_model = ollama_model

    def query(self, user_query: str) -> RAGResponse:
        """Execute end-to-end RAG pipeline for a user question."""
        start_time = time.perf_counter()

        retrieval_start = time.perf_counter()
        retrieved_results = self.vectorstore.search(
            query=user_query,
            top_k=self.top_k,
            threshold=self.score_threshold
        )
        retrieval_time = (time.perf_counter() - retrieval_start) * 1000.0

        if not retrieved_results:
            total_time = (time.perf_counter() - start_time) * 1000.0
            return RAGResponse(
                query=user_query,
                answer=ABSTENTION_MESSAGE,
                citations=[],
                retrieved_chunks=[],
                is_abstained=True,
                retrieval_latency_ms=round(retrieval_time, 2),
                generation_latency_ms=0.0,
                total_latency_ms=round(total_time, 2)
            )

        gen_start = time.perf_counter()
        if self.use_ollama:
            answer, citations = self._generate_ollama(user_query, retrieved_results)
        else:
            answer, citations = self._generate_extractive(user_query, retrieved_results)
        gen_time = (time.perf_counter() - gen_start) * 1000.0

        final_answer = answer
        if self.disclaimer and not answer.startswith(ABSTENTION_MESSAGE):
            final_answer = f"{self.disclaimer}\n\n{answer}"

        total_time = (time.perf_counter() - start_time) * 1000.0

        chunk_payloads = [
            {
                "chunk_id": c.chunk_id,
                "title": c.title,
                "page": c.page,
                "section": c.section,
                "score": round(score, 4),
                "text": c.text
            }
            for c, score in retrieved_results
        ]

        return RAGResponse(
            query=user_query,
            answer=final_answer,
            citations=citations,
            retrieved_chunks=chunk_payloads,
            is_abstained=(answer == ABSTENTION_MESSAGE),
            retrieval_latency_ms=round(retrieval_time, 2),
            generation_latency_ms=round(gen_time, 2),
            total_latency_ms=round(total_time, 2)
        )

    def _generate_extractive(
        self,
        query: str,
        retrieved_results: List[Tuple[Chunk, float]]
    ) -> Tuple[str, List[str]]:
        """
        Hermetic extractive synthesis: Extracts the most relevant sentences
        from top matching chunks and attaches exact citation coordinates.
        """
        top_chunk, top_score = retrieved_results[0]
        sentences = re.split(r"(?<=[.!?])\s+", top_chunk.text.strip())

        query_words = set(re.findall(r"\w+", query.lower())) - {
            "what", "is", "the", "are", "how", "in", "to", "for", "a", "of", "an", "does", "do", "by", "under"
        }
        best_sentence = sentences[0]
        max_overlap = -1

        for s in sentences:
            s_words = set(re.findall(r"\w+", s.lower()))
            overlap = len(query_words & s_words)
            if overlap > max_overlap:
                max_overlap = overlap
                best_sentence = s

        citation = top_chunk.format_citation()
        answer = f"{best_sentence.strip()} {citation}"
        citations = [citation]

        for other_chunk, _ in retrieved_results[1:]:
            c_tag = other_chunk.format_citation()
            if c_tag not in citations:
                citations.append(c_tag)

        return answer, citations

    def _generate_ollama(
        self,
        query: str,
        retrieved_results: List[Tuple[Chunk, float]]
    ) -> Tuple[str, List[str]]:
        """Local Ollama generation with fallback to extractive engine."""
        import requests

        context_blocks = []
        citations = []
        for c, _ in retrieved_results:
            c_tag = c.format_citation()
            citations.append(c_tag)
            context_blocks.append(f"{c_tag}\n{c.text}")

        context_str = "\n\n".join(context_blocks)
        prompt = (
            f"You are a factual assistant for {self.domain_name}. "
            f"Answer the user's question STRICTLY based on the provided context below. "
            f"If the context does not contain sufficient facts to answer, respond exactly: '{ABSTENTION_MESSAGE}'. "
            f"Every factual statement must end with its exact citation tag [Doc: <title>, Page: <page>, Section: <section>].\n\n"
            f"CONTEXT:\n{context_str}\n\n"
            f"QUESTION: {query}\n\n"
            f"ANSWER:"
        )

        try:
            resp = requests.post(
                self.ollama_endpoint,
                json={"model": self.ollama_model, "prompt": prompt, "stream": False},
                timeout=5.0
            )
            if resp.status_code == 200:
                out = resp.json().get("response", "").strip()
                if out:
                    return out, citations
        except Exception:
            pass

        return self._generate_extractive(query, retrieved_results)
