"""
src/state.py
Typed Workflow State & Execution Trace Data Model for AG-10.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time


@dataclass
class WorkflowState:
    session_id: str
    user_query: str
    current_agent: str = "CurriculumArchitectAgent"
    step_count: int = 0
    tool_history: List[Dict[str, Any]] = field(default_factory=list)
    agent_outputs: Dict[str, Any] = field(default_factory=dict)
    requires_human_approval: bool = False
    human_decision: Optional[str] = None  # None, "APPROVED", "REJECTED"
    final_response: Optional[str] = None
    is_completed: bool = False
    is_aborted: bool = False
    errors: List[str] = field(default_factory=list)

    def record_step(self, agent: str, tool_name: str, tool_res: Dict[str, Any]) -> None:
        self.step_count += 1
        self.tool_history.append({
            "step": self.step_count,
            "agent": agent,
            "tool": tool_name,
            "status": tool_res.get("status", "SUCCESS"),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "output": tool_res
        })
