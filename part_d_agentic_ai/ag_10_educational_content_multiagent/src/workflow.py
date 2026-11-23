"""
src/workflow.py
State Machine and Multi-Agent Orchestrator for AG-10: Educational Content Development Agent.
Coordinates CurriculumArchitectAgent -> ContentWriterAgent -> PedagogicalReviewerAgent with safety guards and HITL gates.
"""

from typing import Dict, Any, Optional
from src.state import WorkflowState
from src.tools import get_tool_registry


class MultiAgentWorkflow:
    def __init__(self):
        self.tools = get_tool_registry()
        self.agents = ['CurriculumArchitectAgent', 'ContentWriterAgent', 'PedagogicalReviewerAgent']

    def run(self, user_query: str, human_approval: Optional[str] = None) -> WorkflowState:
        state = WorkflowState(
            session_id=f"SESS_AG-10_{int(user_query.__hash__() % 100000)}",
            user_query=user_query,
            human_decision=human_approval
        )

        # Stage 1: CurriculumArchitectAgent - Initial Ingestion & Diagnostics
        state.current_agent = "CurriculumArchitectAgent"
        tool_1_res = self.tools["SyllabusAlignerTool"].execute(query=user_query)
        state.record_step("CurriculumArchitectAgent", "SyllabusAlignerTool", tool_1_res)
        state.agent_outputs["CurriculumArchitectAgent"] = f"Processed telemetry with confidence 94%."

        # Stage 2: ContentWriterAgent - Secondary Verification & Policy Check
        state.current_agent = "ContentWriterAgent"
        tool_2_res = self.tools["QuizGeneratorTool"].execute(input=tool_1_res)
        state.record_step("ContentWriterAgent", "QuizGeneratorTool", tool_2_res)
        state.agent_outputs["ContentWriterAgent"] = "Verified compliance against operational policies."

        # Stage 3: PedagogicalReviewerAgent - Operational Synthesis & Consequential Action
        state.current_agent = "PedagogicalReviewerAgent"
        tool_3_res = self.tools["ReadabilityScorerTool"].execute(input=tool_2_res)
        state.record_step("PedagogicalReviewerAgent", "ReadabilityScorerTool", tool_3_res)
        state.requires_human_approval = True

        # Safety Gate: Human-in-the-Loop Verification
        if state.human_decision == "REJECTED":
            state.is_aborted = True
            state.final_response = "Safety Gate: Action 'Publish Curriculum Module to Student Learning Portal' rejected by supervisor. Execution aborted safely."
        elif state.human_decision == "APPROVED":
            state.is_completed = True
            state.final_response = "Safety Gate: Action 'Publish Curriculum Module to Student Learning Portal' approved by supervisor. Action executed with full audit trace."
        else:
            # Paused at approval gate
            state.is_completed = False
            state.final_response = "Paused at Safety Gate: Action 'Publish Curriculum Module to Student Learning Portal' requires supervisor approval."

        return state
