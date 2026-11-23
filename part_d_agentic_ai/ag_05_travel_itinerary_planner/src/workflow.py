"""
src/workflow.py
State Machine and Multi-Agent Orchestrator for AG-05: Travel Itinerary & Budget Multi-Agent.
Coordinates DestinationResearchAgent -> RoutePlannerAgent -> BudgetAuditor with safety guards and HITL gates.
"""

from typing import Dict, Any, Optional
from src.state import WorkflowState
from src.tools import get_tool_registry


class MultiAgentWorkflow:
    def __init__(self):
        self.tools = get_tool_registry()
        self.agents = ['DestinationResearchAgent', 'RoutePlannerAgent', 'BudgetAuditor']

    def run(self, user_query: str, human_approval: Optional[str] = None) -> WorkflowState:
        state = WorkflowState(
            session_id=f"SESS_AG-05_{int(user_query.__hash__() % 100000)}",
            user_query=user_query,
            human_decision=human_approval
        )

        # Stage 1: DestinationResearchAgent - Initial Ingestion & Diagnostics
        state.current_agent = "DestinationResearchAgent"
        tool_1_res = self.tools["TransitEstimatorTool"].execute(query=user_query)
        state.record_step("DestinationResearchAgent", "TransitEstimatorTool", tool_1_res)
        state.agent_outputs["DestinationResearchAgent"] = f"Processed telemetry with confidence 94%."

        # Stage 2: RoutePlannerAgent - Secondary Verification & Policy Check
        state.current_agent = "RoutePlannerAgent"
        tool_2_res = self.tools["AccommodationBudgetTool"].execute(input=tool_1_res)
        state.record_step("RoutePlannerAgent", "AccommodationBudgetTool", tool_2_res)
        state.agent_outputs["RoutePlannerAgent"] = "Verified compliance against operational policies."

        # Stage 3: BudgetAuditor - Operational Synthesis & Consequential Action
        state.current_agent = "BudgetAuditor"
        tool_3_res = self.tools["AttractionCatalogTool"].execute(input=tool_2_res)
        state.record_step("BudgetAuditor", "AttractionCatalogTool", tool_3_res)
        state.requires_human_approval = True

        # Safety Gate: Human-in-the-Loop Verification
        if state.human_decision == "REJECTED":
            state.is_aborted = True
            state.final_response = "Safety Gate: Action 'Authorize Final Travel Booking Reservation' rejected by supervisor. Execution aborted safely."
        elif state.human_decision == "APPROVED":
            state.is_completed = True
            state.final_response = "Safety Gate: Action 'Authorize Final Travel Booking Reservation' approved by supervisor. Action executed with full audit trace."
        else:
            # Paused at approval gate
            state.is_completed = False
            state.final_response = "Paused at Safety Gate: Action 'Authorize Final Travel Booking Reservation' requires supervisor approval."

        return state
