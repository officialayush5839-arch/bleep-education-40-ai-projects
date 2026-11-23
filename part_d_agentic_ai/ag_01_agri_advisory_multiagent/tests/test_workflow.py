"""
tests/test_workflow.py
Tier 2 Tests: Multi-agent coordination and state transitions for AG-01.
"""

from src.workflow import MultiAgentWorkflow


def test_multiagent_happy_path():
    wf = MultiAgentWorkflow()
    state = wf.run("Test query", human_approval="APPROVED")
    assert state.is_completed is True
    assert len(state.tool_history) >= 2
    assert "SoilAnalystAgent" in state.agent_outputs


def test_safety_gate_rejection():
    wf = MultiAgentWorkflow()
    state = wf.run("Test risky action", human_approval="REJECTED")
    assert state.is_aborted is True
    assert "rejected" in state.final_response.lower()
