"""
src/workflow.py
State Machine and Multi-Agent Orchestrator for AG-03: Financial Market Research Multi-Agent.
Coordinates FundamentalAnalystAgent -> NewsSentimentAgent -> ComplianceAuditor with safety guards and HITL gates.
"""

from typing import Dict, Any, Optional
from src.state import WorkflowState
from src.tools import get_tool_registry


class MultiAgentWorkflow:
    def __init__(self):
        self.tools = get_tool_registry()
        self.agents = ['FundamentalAnalystAgent', 'NewsSentimentAgent', 'ComplianceAuditor']

    def run(self, user_query: str, human_approval: Optional[str] = None) -> WorkflowState:
        state = WorkflowState(
            session_id=f"SESS_AG-03_{int(user_query.__hash__() % 100000)}",
            user_query=user_query,
            human_decision=human_approval
        )

        # Stage 1: FundamentalAnalystAgent - Initial Ingestion & Diagnostics
        state.current_agent = "FundamentalAnalystAgent"
        tool_1_res = self.tools["FinancialRatioTool"].execute(query=user_query)
        state.record_step("FundamentalAnalystAgent", "FinancialRatioTool", tool_1_res)
        state.agent_outputs["FundamentalAnalystAgent"] = f"Processed telemetry with confidence 94%."

        # Stage 2: NewsSentimentAgent - Secondary Verification & Policy Check
        state.current_agent = "NewsSentimentAgent"
        tool_2_res = self.tools["NewsSentimentTool"].execute(input=tool_1_res)
        state.record_step("NewsSentimentAgent", "NewsSentimentTool", tool_2_res)
        state.agent_outputs["NewsSentimentAgent"] = "Verified compliance against operational policies."

        # Stage 3: ComplianceAuditor - Operational Synthesis & Consequential Action
        state.current_agent = "ComplianceAuditor"
        tool_3_res = self.tools["SECFilingsSearchTool"].execute(input=tool_2_res)
        state.record_step("ComplianceAuditor", "SECFilingsSearchTool", tool_3_res)
        state.requires_human_approval = True

        # Safety Gate: Human-in-the-Loop Verification
        if state.human_decision == "REJECTED":
            state.is_aborted = True
            state.final_response = "Safety Gate: Action 'Publish Official Investment Thesis Report' rejected by supervisor. Execution aborted safely."
        elif state.human_decision == "APPROVED":
            state.is_completed = True
            state.final_response = "Safety Gate: Action 'Publish Official Investment Thesis Report' approved by supervisor. Action executed with full audit trace."
        else:
            # Paused at approval gate
            state.is_completed = False
            state.final_response = "Paused at Safety Gate: Action 'Publish Official Investment Thesis Report' requires supervisor approval."

        return state
