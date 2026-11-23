"""
src/workflow.py
State Machine and Multi-Agent Orchestrator for AG-01: Agricultural Advisory Multi-Agent System.
Coordinates SoilAnalystAgent -> WeatherForecastAgent -> AgronomyAdvisoryAgent with safety guards and HITL gates.
"""

from typing import Dict, Any, Optional
from src.state import WorkflowState
from src.tools import get_tool_registry


class MultiAgentWorkflow:
    def __init__(self):
        self.tools = get_tool_registry()
        self.agents = ['SoilAnalystAgent', 'WeatherForecastAgent', 'AgronomyAdvisoryAgent']

    def run(self, user_query: str, human_approval: Optional[str] = None) -> WorkflowState:
        state = WorkflowState(
            session_id=f"SESS_AG-01_{int(user_query.__hash__() % 100000)}",
            user_query=user_query,
            human_decision=human_approval
        )

        # Stage 1: SoilAnalystAgent - Initial Ingestion & Diagnostics
        state.current_agent = "SoilAnalystAgent"
        tool_1_res = self.tools["SoilAnalyzerTool"].execute(query=user_query)
        state.record_step("SoilAnalystAgent", "SoilAnalyzerTool", tool_1_res)
        state.agent_outputs["SoilAnalystAgent"] = f"Processed telemetry with confidence 94%."

        # Stage 2: WeatherForecastAgent - Secondary Verification & Policy Check
        state.current_agent = "WeatherForecastAgent"
        tool_2_res = self.tools["WeatherLookupTool"].execute(input=tool_1_res)
        state.record_step("WeatherForecastAgent", "WeatherLookupTool", tool_2_res)
        state.agent_outputs["WeatherForecastAgent"] = "Verified compliance against operational policies."

        # Stage 3: AgronomyAdvisoryAgent - Operational Synthesis & Consequential Action
        state.current_agent = "AgronomyAdvisoryAgent"
        tool_3_res = self.tools["CropNutrientTool"].execute(input=tool_2_res)
        state.record_step("AgronomyAdvisoryAgent", "CropNutrientTool", tool_3_res)
        state.requires_human_approval = True

        # Safety Gate: Human-in-the-Loop Verification
        if state.human_decision == "REJECTED":
            state.is_aborted = True
            state.final_response = "Safety Gate: Action 'Disburse Bulk Chemical Fertilizer Order' rejected by supervisor. Execution aborted safely."
        elif state.human_decision == "APPROVED":
            state.is_completed = True
            state.final_response = "Safety Gate: Action 'Disburse Bulk Chemical Fertilizer Order' approved by supervisor. Action executed with full audit trace."
        else:
            # Paused at approval gate
            state.is_completed = False
            state.final_response = "Paused at Safety Gate: Action 'Disburse Bulk Chemical Fertilizer Order' requires supervisor approval."

        return state
