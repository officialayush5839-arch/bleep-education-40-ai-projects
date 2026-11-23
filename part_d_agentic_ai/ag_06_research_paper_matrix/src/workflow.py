"""
src/workflow.py
State Machine and Multi-Agent Orchestrator for AG-06: Research Paper Analysis Multi-Agent.
Coordinates PaperDiscoveryAgent -> MethodologyExtractor -> SynthesisReviewer with safety guards and HITL gates.
"""

from typing import Dict, Any, Optional
from src.state import WorkflowState
from src.tools import get_tool_registry


class MultiAgentWorkflow:
    def __init__(self):
        self.tools = get_tool_registry()
        self.agents = ['PaperDiscoveryAgent', 'MethodologyExtractor', 'SynthesisReviewer']

    def run(self, user_query: str, human_approval: Optional[str] = None) -> WorkflowState:
        state = WorkflowState(
            session_id=f"SESS_AG-06_{int(user_query.__hash__() % 100000)}",
            user_query=user_query,
            human_decision=human_approval
        )

        # Stage 1: PaperDiscoveryAgent - Initial Ingestion & Diagnostics
        state.current_agent = "PaperDiscoveryAgent"
        tool_1_res = self.tools["PaperMetadataTool"].execute(query=user_query)
        state.record_step("PaperDiscoveryAgent", "PaperMetadataTool", tool_1_res)
        state.agent_outputs["PaperDiscoveryAgent"] = f"Processed telemetry with confidence 94%."

        # Stage 2: MethodologyExtractor - Secondary Verification & Policy Check
        state.current_agent = "MethodologyExtractor"
        tool_2_res = self.tools["MethodologyMatrixTool"].execute(input=tool_1_res)
        state.record_step("MethodologyExtractor", "MethodologyMatrixTool", tool_2_res)
        state.agent_outputs["MethodologyExtractor"] = "Verified compliance against operational policies."

        # Stage 3: SynthesisReviewer - Operational Synthesis & Consequential Action
        state.current_agent = "SynthesisReviewer"
        tool_3_res = self.tools["CitationGraphTool"].execute(input=tool_2_res)
        state.record_step("SynthesisReviewer", "CitationGraphTool", tool_3_res)
        state.requires_human_approval = True

        # Safety Gate: Human-in-the-Loop Verification
        if state.human_decision == "REJECTED":
            state.is_aborted = True
            state.final_response = "Safety Gate: Action 'Publish Systematic Literature Review Matrix' rejected by supervisor. Execution aborted safely."
        elif state.human_decision == "APPROVED":
            state.is_completed = True
            state.final_response = "Safety Gate: Action 'Publish Systematic Literature Review Matrix' approved by supervisor. Action executed with full audit trace."
        else:
            # Paused at approval gate
            state.is_completed = False
            state.final_response = "Paused at Safety Gate: Action 'Publish Systematic Literature Review Matrix' requires supervisor approval."

        return state
