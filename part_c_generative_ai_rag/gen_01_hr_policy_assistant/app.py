"""
app.py
Interactive Streamlit Application for Enterprise RAG System (GEN-01: Enterprise HR Policy Assistant).
"""

import os
import json
from pathlib import Path
import streamlit as st

from src.data_loader import DataLoader
from src.chunker import RecursiveChunker
from src.vectorstore import VectorStore
from src.rag_pipeline import RAGPipeline, ABSTENTION_MESSAGE


st.set_page_config(
    page_title="Enterprise HR Policy Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

DISCLAIMER_TEXT = """This HR Policy Assistant is for internal workforce guidance and does not replace official employment contracts or legal counsel."""


@st.cache_resource
def get_pipeline(score_threshold: float, top_k: int, use_ollama: bool):
    loader = DataLoader(domain="hr")
    raw_dir = Path("data/raw_documents")
    if not raw_dir.exists():
        raw_dir = Path("data/raw")
    docs = loader.load_directory(str(raw_dir)) if raw_dir.exists() else []

    chunker = RecursiveChunker(chunk_size=500, chunk_overlap=80)
    chunks = chunker.chunk_documents(docs)

    vstore = VectorStore(score_threshold=score_threshold)
    vstore.add_chunks(chunks)

    pipeline = RAGPipeline(
        vectorstore=vstore,
        domain_name="Enterprise HR Policy Assistant",
        disclaimer=DISCLAIMER_TEXT,
        score_threshold=score_threshold,
        top_k=top_k,
        use_ollama=use_ollama
    )
    return pipeline, len(docs), len(chunks)


with st.sidebar:
    st.title("⚙️ RAG Engine Settings")
    st.caption("GEN-01 | Human Resources")
    top_k = st.slider("Top-K Retrieved Chunks", min_value=1, max_value=5, value=3)
    score_threshold = st.slider("Relevance Score Threshold", min_value=0.0, max_value=0.8, value=0.15, step=0.01)
    use_ollama = st.checkbox("Enable Local Ollama LLM", value=False)
    
    st.markdown("---")
    pipeline, num_docs, num_chunks = get_pipeline(score_threshold, top_k, use_ollama)
    st.metric("Indexed Documents", num_docs)
    st.metric("Total Indexed Chunks", num_chunks)
    st.markdown("---")
    st.caption("Milestone 4 — Generative AI / RAG")


st.title("📚 Enterprise HR Policy Assistant")
st.caption("Domain: Human Resources | Bleep Education 40 AI Projects")

if DISCLAIMER_TEXT.strip():
    st.info(DISCLAIMER_TEXT)

tabs = st.tabs(["💬 Grounded Q&A", "📄 Knowledge Base Explorer", "📊 Evaluation Benchmark"])

with tabs[0]:
    query_text = st.text_input("Enter your question:", placeholder="Ask about hr documentation...")
    if st.button("Submit Query", type="primary") or query_text:
        if query_text.strip():
            with st.spinner("Retrieving facts and synthesizing grounded answer..."):
                resp = pipeline.query(query_text)

            if resp.is_abstained:
                st.warning(f"🛡️ **Abstention Response**: {resp.answer}")
                st.info("The system detected that this question is out of scope for the indexed documentation.")
            else:
                st.success("✅ **Grounded Answer**")
                st.markdown(resp.answer)

                if resp.citations:
                    st.subheader("Traceable Citations")
                    for c in resp.citations:
                        st.code(c, language="markdown")

                with st.expander("🔍 View Retrieved Context Chunks"):
                    for item in resp.retrieved_chunks:
                        st.markdown(f"**{item['title']}** (Page {item['page']}, Section: {item['section']}) — *Similarity Score: {item['score']}*")
                        st.caption(item["text"])
                        st.divider()

with tabs[1]:
    st.subheader("Indexed Document Corpus")
    raw_dir = Path("data/raw_documents")
    if not raw_dir.exists():
        raw_dir = Path("data/raw")
    if raw_dir.exists():
        files = list(raw_dir.glob("*.md"))
        for f in sorted(files):
            with st.expander(f"📄 {f.name}"):
                st.code(f.read_text(encoding="utf-8", errors="replace")[:1200] + "\n\n[Truncated preview...]", language="markdown")
    else:
        st.info("No raw documents found.")

with tabs[2]:
    st.subheader("Quantitative Evaluation Results")
    metrics_file = Path("results/metrics.json")
    if metrics_file.exists():
        with open(metrics_file, "r", encoding="utf-8") as mf:
            data = json.load(mf)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Overall Accuracy", f"{data.get('overall_accuracy', 0)*100:.1f}%")
        col2.metric("Citation Precision", f"{data.get('citation_precision', 0)*100:.1f}%")
        col3.metric("Abstention Accuracy", f"{data.get('abstention_accuracy', 0)*100:.1f}%")
        col4.metric("Mean Latency", f"{data.get('mean_latency_ms', 0):.1f} ms")
        
        st.markdown("### Benchmark Query Evaluation Log (15 Domain Questions)")
        st.dataframe(data.get("query_results", []))
    else:
        st.warning("No evaluation metrics file found at results/metrics.json")
