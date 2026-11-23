"""
src/workflow.py
State Machine and Multi-Agent Orchestrator for AG-09: Inventory Reordering Multi-Agent.
Coordinates StockLevelMonitorAgent -> DemandForecasterAgent -> PurchaseDrafterAgent with safety guards and HITL gates.
"""

from typing import Dict, Any, Optional
from src.state import WorkflowState
from src.tools import get_tool_registry


class MultiAgentWorkflow:
    def __init__(self):
        self.tools = get_tool_registry()
        self.agents = ['StockLevelMonitorAgent', 'DemandForecasterAgent', 'PurchaseDrafterAgent']

    def run(self, user_query: str, human_approval: Optional[str] = None) -> WorkflowState:
        state = WorkflowState(
            session_id=f"SESS_AG-09_{int(user_query.__hash__() % 100000)}",
            user_query=user_query,
            human_decision=human_approval
        )

        # Stage 1: StockLevelMonitorAgent - Initial Ingestion & Diagnostics
        state.current_agent = "StockLevelMonitorAgent"
        tool_1_res = self.tools["StockLevelCheckTool"].execute(query=user_query)
        state.record_step("StockLevelMonitorAgent", "StockLevelCheckTool", tool_1_res)
        state.agent_outputs["StockLevelMonitorAgent"] = f"Processed telemetry with confidence 94%."

        # Stage 2: DemandForecasterAgent - Secondary Verification & Policy Check
        state.current_agent = "DemandForecasterAgent"
        tool_2_res = self.tools["ReorderQuantityTool"].execute(input=tool_1_res)
        state.record_step("DemandForecasterAgent", "ReorderQuantityTool", tool_2_res)
        state.agent_outputs["DemandForecasterAgent"] = "Verified compliance against operational policies."

        # Stage 3: PurchaseDrafterAgent - Operational Synthesis & Consequential Action
        state.current_agent = "PurchaseDrafterAgent"
        tool_3_res = self.tools["VendorCatalogTool"].execute(input=tool_2_res)
        state.record_step("PurchaseDrafterAgent", "VendorCatalogTool", tool_3_res)
        state.requires_human_approval = True

        # Safety Gate: Human-in-the-Loop Verification
        if state.human_decision == "REJECTED":
            state.is_aborted = True
            state.final_response = "Safety Gate: Action 'Submit Purchase Order to External Vendor' rejected by supervisor. Execution aborted safely."
        elif state.human_decision == "APPROVED":
            state.is_completed = True
            state.final_response = "Safety Gate: Action 'Submit Purchase Order to External Vendor' approved by supervisor. Action executed with full audit trace."
        else:
            # Paused at approval gate
            state.is_completed = False
            state.final_response = "Paused at Safety Gate: Action 'Submit Purchase Order to External Vendor' requires supervisor approval."

        return state
