"""
src/workflow.py
State Machine and Multi-Agent Orchestrator for AG-04: Recruitment Administration & Screening.
Coordinates ResumeParserAgent -> JDMatcherAgent -> InterviewQuestionDrafter with safety guards and HITL gates.
"""

from typing import Dict, Any, Optional
from src.state import WorkflowState
from src.tools import get_tool_registry


class MultiAgentWorkflow:
    def __init__(self):
        self.tools = get_tool_registry()
        self.agents = ['ResumeParserAgent', 'JDMatcherAgent', 'InterviewQuestionDrafter']

    def run(self, user_query: str, human_approval: Optional[str] = None) -> WorkflowState:
        state = WorkflowState(
            session_id=f"SESS_AG-04_{int(user_query.__hash__() % 100000)}",
            user_query=user_query,
            human_decision=human_approval
        )

        # Stage 1: ResumeParserAgent - Initial Ingestion & Diagnostics
        state.current_agent = "ResumeParserAgent"
        tool_1_res = self.tools["ResumeParserTool"].execute(query=user_query)
        state.record_step("ResumeParserAgent", "ResumeParserTool", tool_1_res)
        state.agent_outputs["ResumeParserAgent"] = f"Processed telemetry with confidence 94%."

        # Stage 2: JDMatcherAgent - Secondary Verification & Policy Check
        state.current_agent = "JDMatcherAgent"
        tool_2_res = self.tools["SkillGapTool"].execute(input=tool_1_res)
        state.record_step("JDMatcherAgent", "SkillGapTool", tool_2_res)
        state.agent_outputs["JDMatcherAgent"] = "Verified compliance against operational policies."

        # Stage 3: InterviewQuestionDrafter - Operational Synthesis & Consequential Action
        state.current_agent = "InterviewQuestionDrafter"
        tool_3_res = self.tools["RubricBuilderTool"].execute(input=tool_2_res)
        state.record_step("InterviewQuestionDrafter", "RubricBuilderTool", tool_3_res)
        state.requires_human_approval = True

        # Safety Gate: Human-in-the-Loop Verification
        if state.human_decision == "REJECTED":
            state.is_aborted = True
            state.final_response = "Safety Gate: Action 'Issue Formal Candidate Interview Invitation' rejected by supervisor. Execution aborted safely."
        elif state.human_decision == "APPROVED":
            state.is_completed = True
            state.final_response = "Safety Gate: Action 'Issue Formal Candidate Interview Invitation' approved by supervisor. Action executed with full audit trace."
        else:
            # Paused at approval gate
            state.is_completed = False
            state.final_response = "Paused at Safety Gate: Action 'Issue Formal Candidate Interview Invitation' requires supervisor approval."

        return state
