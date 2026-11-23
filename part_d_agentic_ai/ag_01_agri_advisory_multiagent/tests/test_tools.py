"""
tests/test_tools.py
Tier 1 Tests: Functional tool execution and schema contracts for AG-01.
"""

from src.tools import get_tool_registry


def test_tool_registry_and_execution():
    tools = get_tool_registry()
    assert len(tools) >= 2, "Must provide at least 2 functional tools"
    
    for name, tool in tools.items():
        res = tool.execute(query="test_query")
        assert "status" in res
        assert "execution_ms" in res
