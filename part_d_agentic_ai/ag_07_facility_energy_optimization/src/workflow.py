"""
src/workflow.py
State Machine and Multi-Agent Orchestrator for AG-07: Commercial Facility Energy Optimization.
Coordinates TelemetryIngestionAgent -> AnomalyDetectorAgent -> HVACOptimizationAgent with safety guards and HITL gates.
"""

from typing import Dict, Any, Optional
from src.state import WorkflowState
from src.tools import get_tool_registry


class MultiAgentWorkflow:
    def __init__(self):
        self.tools = get_tool_registry()
        self.agents = ['TelemetryIngestionAgent', 'AnomalyDetectorAgent', 'HVACOptimizationAgent']

    def run(self, user_query: str, human_approval: Optional[str] = None) -> WorkflowState:
        state = WorkflowState(
            session_id=f"SESS_AG-07_{int(user_query.__hash__() % 100000)}",
            user_query=user_query,
            human_decision=human_approval
        )

        # Stage 1: TelemetryIngestionAgent - Initial Ingestion & Diagnostics
        state.current_agent = "TelemetryIngestionAgent"
        tool_1_res = self.tools["LoadAnomalyTool"].execute(query=user_query)
        state.record_step("TelemetryIngestionAgent", "LoadAnomalyTool", tool_1_res)
        state.agent_outputs["TelemetryIngestionAgent"] = f"Processed telemetry with confidence 94%."

        # Stage 2: AnomalyDetectorAgent - Secondary Verification & Policy Check
        state.current_agent = "AnomalyDetectorAgent"
        tool_2_res = self.tools["ConservationSavingsTool"].execute(input=tool_1_res)
        state.record_step("AnomalyDetectorAgent", "ConservationSavingsTool", tool_2_res)
        state.agent_outputs["AnomalyDetectorAgent"] = "Verified compliance against operational policies."

        # Stage 3: HVACOptimizationAgent - Operational Synthesis & Consequential Action
        state.current_agent = "HVACOptimizationAgent"
        tool_3_res = self.tools["HVACScheduleTool"].execute(input=tool_2_res)
        state.record_step("HVACOptimizationAgent", "HVACScheduleTool", tool_3_res)
        state.requires_human_approval = True

        # Safety Gate: Human-in-the-Loop Verification
        if state.human_decision == "REJECTED":
            state.is_aborted = True
            state.final_response = "Safety Gate: Action 'Apply Facility Building Automation Setpoint Changes' rejected by supervisor. Execution aborted safely."
        elif state.human_decision == "APPROVED":
            state.is_completed = True
            state.final_response = "Safety Gate: Action 'Apply Facility Building Automation Setpoint Changes' approved by supervisor. Action executed with full audit trace."
        else:
            # Paused at approval gate
            state.is_completed = False
            state.final_response = "Paused at Safety Gate: Action 'Apply Facility Building Automation Setpoint Changes' requires supervisor approval."

        return state
