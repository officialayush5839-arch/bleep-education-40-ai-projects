"""
app.py
Interactive Streamlit Mission Control for AG-08: Security Incident Response Multi-Agent.
"""

import streamlit as st
import json
from pathlib import Path
from src.workflow import MultiAgentWorkflow

st.set_page_config(page_title="Security Incident Response Multi-Agent", page_icon="🤖", layout="wide")

st.title("🤖 AG-08: Security Incident Response Multi-Agent")
st.caption("Domain: Cyber Defense | Bleep Education 40 AI Projects")
st.info("""Cyber defense triage tool. Network containment actions strictly require SOC Lead approval.""")

workflow = MultiAgentWorkflow()

tabs = st.tabs(["🚀 Multi-Agent Workflow Runner", "🛡️ Human-in-the-Loop Gate", "📊 Scenario Benchmark Log"])

with tabs[0]:
    st.subheader("Interactive Agent Workflow")
    query_text = st.text_input("Enter scenario instruction:", value="Initiate comprehensive domain diagnostic audit")
    
    col_a, col_b = st.columns(2)
    hitl_choice = col_a.radio("Supervisor Pre-Approval Decision:", ["Pending / Review Gate", "Approve Consequential Action", "Reject Consequential Action"])
    
    decision_map = {
        "Pending / Review Gate": None,
        "Approve Consequential Action": "APPROVED",
        "Reject Consequential Action": "REJECTED"
    }

    if st.button("Execute Multi-Agent Workflow", type="primary"):
        state = workflow.run(query_text, human_approval=decision_map[hitl_choice])
        
        st.subheader("Execution Progress & Status")
        if state.is_completed:
            st.success(f"✅ {state.final_response}")
        elif state.is_aborted:
            st.error(f"🛑 {state.final_response}")
        else:
            st.warning(f"⚠️ {state.final_response}")

        st.markdown("### Agent Action Trace")
        for step in state.tool_history:
            with st.expander(f"Step {step['step']}: {step['agent']} -> Tool: {step['tool']} ({step['status']})"):
                st.json(step["output"])

with tabs[1]:
    st.subheader("Human-in-the-Loop (HITL) Safety Gate")
    st.markdown("""**Consequential Action Monitored**: `Deploy Production Network Firewall Block Rule`""")
    st.write("All external operations with legal, financial, or physical facility impact are halted at this gate until certified human credentials are submitted.")
    c1, c2 = st.columns(2)
    c1.button("✅ Authorize Action", type="primary", use_container_width=True)
    c2.button("🛑 Abort Action", use_container_width=True)

with tabs[2]:
    st.subheader("10-Scenario Quantitative Benchmark")
    metrics_path = Path("results/metrics.json")
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as mf:
            metrics = json.load(mf)
        st.metric("Benchmark Success Rate", f"{metrics.get('success_rate', 1.0)*100:.1f}%")
        st.dataframe(metrics.get("results", []))
    else:
        st.warning("No metrics.json found.")
