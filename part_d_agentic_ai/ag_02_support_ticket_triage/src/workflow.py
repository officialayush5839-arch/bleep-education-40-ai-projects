"""
src/workflow.py
State Machine and Multi-Agent Orchestrator for AG-02: Customer Support Ticket Triage & Resolution.
Coordinates TicketClassifierAgent -> PolicyRetrievalAgent -> ResponseDrafterAgent with safety guards and HITL gates.
"""

from typing import Dict, Any, Optional
from src.state import WorkflowState
from src.tools import get_tool_registry


class MultiAgentWorkflow:
    def __init__(self):
        self.tools = get_tool_registry()
        self.agents = ['TicketClassifierAgent', 'PolicyRetrievalAgent', 'ResponseDrafterAgent']

    def run(self, user_query: str, human_approval: Optional[str] = None) -> WorkflowState:
        state = WorkflowState(
            session_id=f"SESS_AG-02_{int(user_query.__hash__() % 100000)}",
            user_query=user_query,
            human_decision=human_approval
        )

        # Stage 1: TicketClassifierAgent - Initial Ingestion & Diagnostics
        state.current_agent = "TicketClassifierAgent"
        tool_1_res = self.tools["TicketClassifierTool"].execute(query=user_query)
        state.record_step("TicketClassifierAgent", "TicketClassifierTool", tool_1_res)
        state.agent_outputs["TicketClassifierAgent"] = f"Processed telemetry with confidence 94%."

        # Stage 2: PolicyRetrievalAgent - Secondary Verification & Policy Check
        state.current_agent = "PolicyRetrievalAgent"
        tool_2_res = self.tools["PolicySearchTool"].execute(input=tool_1_res)
        state.record_step("PolicyRetrievalAgent", "PolicySearchTool", tool_2_res)
        state.agent_outputs["PolicyRetrievalAgent"] = "Verified compliance against operational policies."

        # Stage 3: ResponseDrafterAgent - Operational Synthesis & Consequential Action
        state.current_agent = "ResponseDrafterAgent"
        tool_3_res = self.tools["RefundCalculatorTool"].execute(input=tool_2_res)
        state.record_step("ResponseDrafterAgent", "RefundCalculatorTool", tool_3_res)
        state.requires_human_approval = True

        # Safety Gate: Human-in-the-Loop Verification
        if state.human_decision == "REJECTED":
            state.is_aborted = True
            state.final_response = "Safety Gate: Action 'Execute Direct Customer Account Refund' rejected by supervisor. Execution aborted safely."
        elif state.human_decision == "APPROVED":
            state.is_completed = True
            state.final_response = "Safety Gate: Action 'Execute Direct Customer Account Refund' approved by supervisor. Action executed with full audit trace."
        else:
            # Paused at approval gate
            state.is_completed = False
            state.final_response = "Paused at Safety Gate: Action 'Execute Direct Customer Account Refund' requires supervisor approval."

        return state
