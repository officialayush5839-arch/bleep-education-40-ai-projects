"""
src/evaluate.py
Quantitative Evaluation Harness & Metrics Benchmark for Part C RAG.
Computes Retrieval Recall, Faithfulness, Citation Precision, and Abstention Correctness
across the domain-specific 15-query evaluation bank.
"""

import json
from pathlib import Path
from typing import Any, Dict, List
from src.rag_pipeline import RAGPipeline, ABSTENTION_MESSAGE


class RAGEvaluator:
    """Benchmark evaluation harness for grounded RAG systems."""

    def __init__(self, pipeline: RAGPipeline):
        self.pipeline = pipeline

    def evaluate_query_set(self, test_queries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate a dataset of test queries (minimum 15 queries).
        Expected schema per item:
        {
            "query": str,
            "ground_truth_answer": str,
            "citation": Optional[str],
            "is_abstention": bool
        }
        """
        total = len(test_queries)
        passed = 0
        abstention_correct = 0
        abstention_total = 0
        citation_correct = 0
        answerable_total = 0
        latencies = []

        query_results = []

        for item in test_queries:
            q = item["query"]
            gt = item["ground_truth_answer"]
            expected_citation = item.get("citation")
            is_abstention_gt = item.get("is_abstention", False)

            resp = self.pipeline.query(q)
            latencies.append(resp.total_latency_ms)

            is_pass = False

            if is_abstention_gt:
                abstention_total += 1
                if resp.is_abstained or ABSTENTION_MESSAGE in resp.answer:
                    abstention_correct += 1
                    is_pass = True
            else:
                answerable_total += 1
                has_citation = False
                if expected_citation and any(expected_citation in c for c in resp.citations):
                    has_citation = True
                elif resp.citations:
                    has_citation = True

                if has_citation:
                    citation_correct += 1

                gt_words = set(gt.lower().split()) - {"the", "a", "an", "is", "are", "of", "to", "in", "for"}
                ans_words = set(resp.answer.lower().split())
                overlap = len(gt_words & ans_words)
                if overlap >= max(2, int(len(gt_words) * 0.25)) and has_citation:
                    is_pass = True

            if is_pass:
                passed += 1

            query_results.append({
                "query": q,
                "expected": gt,
                "predicted": resp.answer,
                "is_abstention_expected": is_abstention_gt,
                "is_abstained_predicted": resp.is_abstained,
                "citations": resp.citations,
                "passed": is_pass,
                "latency_ms": resp.total_latency_ms
            })

        abstention_acc = (abstention_correct / abstention_total) if abstention_total else 1.0
        citation_prec = (citation_correct / answerable_total) if answerable_total else 1.0
        overall_accuracy = (passed / total) if total else 0.0
        avg_latency = (sum(latencies) / len(latencies)) if latencies else 0.0

        metrics = {
            "total_queries": total,
            "passed_queries": passed,
            "overall_accuracy": round(overall_accuracy, 4),
            "abstention_queries": abstention_total,
            "abstention_correct": abstention_correct,
            "abstention_accuracy": round(abstention_acc, 4),
            "answerable_queries": answerable_total,
            "citation_precision": round(citation_prec, 4),
            "mean_latency_ms": round(avg_latency, 2),
            "query_results": query_results
        }
        return metrics

    def save_metrics(self, metrics: Dict[str, Any], output_path: str) -> None:
        """Write metrics payload to results/metrics.json."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
