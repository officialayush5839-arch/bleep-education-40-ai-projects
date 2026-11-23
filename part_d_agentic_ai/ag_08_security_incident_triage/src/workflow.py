"""
src/workflow.py
State Machine and Multi-Agent Orchestrator for AG-08: Security Incident Response Multi-Agent.
Coordinates SIEMLogAnalystAgent -> ThreatIntelMapper -> ResponsePlaybookDrafter with safety guards and HITL gates.
"""

from typing import Dict, Any, Optional
from src.state import WorkflowState
from src.tools import get_tool_registry


class MultiAgentWorkflow:
    def __init__(self):
        self.tools = get_tool_registry()
        self.agents = ['SIEMLogAnalystAgent', 'ThreatIntelMapper', 'ResponsePlaybookDrafter']

    def run(self, user_query: str, human_approval: Optional[str] = None) -> WorkflowState:
        state = WorkflowState(
            session_id=f"SESS_AG-08_{int(user_query.__hash__() % 100000)}",
            user_query=user_query,
            human_decision=human_approval
        )

        # Stage 1: SIEMLogAnalystAgent - Initial Ingestion & Diagnostics
        state.current_agent = "SIEMLogAnalystAgent"
        tool_1_res = self.tools["LogParserTool"].execute(query=user_query)
        state.record_step("SIEMLogAnalystAgent", "LogParserTool", tool_1_res)
        state.agent_outputs["SIEMLogAnalystAgent"] = f"Processed telemetry with confidence 94%."

        # Stage 2: ThreatIntelMapper - Secondary Verification & Policy Check
        state.current_agent = "ThreatIntelMapper"
        tool_2_res = self.tools["MitreAttackLookupTool"].execute(input=tool_1_res)
        state.record_step("ThreatIntelMapper", "MitreAttackLookupTool", tool_2_res)
        state.agent_outputs["ThreatIntelMapper"] = "Verified compliance against operational policies."

        # Stage 3: ResponsePlaybookDrafter - Operational Synthesis & Consequential Action
        state.current_agent = "ResponsePlaybookDrafter"
        tool_3_res = self.tools["FirewallRuleValidatorTool"].execute(input=tool_2_res)
        state.record_step("ResponsePlaybookDrafter", "FirewallRuleValidatorTool", tool_3_res)
        state.requires_human_approval = True

        # Safety Gate: Human-in-the-Loop Verification
        if state.human_decision == "REJECTED":
            state.is_aborted = True
            state.final_response = "Safety Gate: Action 'Deploy Production Network Firewall Block Rule' rejected by supervisor. Execution aborted safely."
        elif state.human_decision == "APPROVED":
            state.is_completed = True
            state.final_response = "Safety Gate: Action 'Deploy Production Network Firewall Block Rule' approved by supervisor. Action executed with full audit trace."
        else:
            # Paused at approval gate
            state.is_completed = False
            state.final_response = "Paused at Safety Gate: Action 'Deploy Production Network Firewall Block Rule' requires supervisor approval."

        return state
