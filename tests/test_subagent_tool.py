import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock
from agentic_core.tools.subagent import SpawnSubAgentsTool, SubAgentCoordinator
from agentic_core.engines.engine import AgentRunner, RunnerConfig
from agentic_core.memory.manager import MemoryManager
from agentic_core.interfaces import AgentResponse

class MockLLMResponse:
    def __init__(self, text="Mocked response"):
        self.text = text
        self.reasoning = ""
        self.tool_calls = []
        self.usage = {}

def mock_ask_impl(conversation, tools, stream=False):
    if stream:
        async def gen():
            yield MockLLMResponse()
        return gen()
    return MockLLMResponse()

@pytest.fixture
def subagent_tool():
    llm = MagicMock()
    llm.ask = MagicMock(side_effect=mock_ask_impl)

    tm = MagicMock()
    tm.prepare_turn = AsyncMock(return_value=None)
    tm.get_toolset_prompt.return_value = "Mock Toolset Prompt"
    tm.get_tools_from_toolset.return_value = []
    tm.get_mcp_loaded_tools.return_value = []


    return SpawnSubAgentsTool(llm, tm)

@pytest.mark.asyncio
async def test_spawn_subagents_basic_flow(subagent_tool):
    context = {
        "memory_manager": MemoryManager()
    }
    
    args = {
        "plan": {
            "nodes": {
                "n1": {"prompt": "Task 1"},
                "n2": {"prompt": "Task 2"}
            },
            "edges": [["n1", "n2"]]
        }
    }
    
    result = await subagent_tool.execute(args, context)
    
    assert "Sub-agent execution results" in result
    assert "[SUCCESS] Task n1" in result
    assert "[SUCCESS] Task n2" in result

@pytest.mark.asyncio
async def test_spawn_subagents_parallel_flow(subagent_tool):
    context = {
        "memory_manager": MemoryManager()
    }
    
    args = {
        "plan": {
            "nodes": {
                "n1": {"prompt": "Task 1"},
                "n2": {"prompt": "Task 2"}
            },
            "edges": []
        }
    }
    
    result = await subagent_tool.execute(args, context)
    
    assert "Sub-agent execution results" in result
    assert "[SUCCESS] Task n1" in result
    assert "[SUCCESS] Task n2" in result

@pytest.mark.asyncio
async def test_spawn_subagents_invalid_plan(subagent_tool):
    context = {
        "memory_manager": MemoryManager()
    }
    
    args = {
        "plan": {
            "nodes": {},
            "edges": []
        }
    }
    
    result = await subagent_tool.execute(args, context)
    assert "Error: The plan must contain at least one node" in result

@pytest.mark.asyncio
async def test_coordinator_direct_usage(subagent_tool):
    context = {
        "memory_manager": MemoryManager()
    }
    
    plan_data = {
        "nodes": {
            "n1": {"prompt": "Task 1"},
        },
        "edges": []
    }

    args = {
        "plan": plan_data
    }
    
    result = await subagent_tool.execute(args, context)
    assert "[SUCCESS] Task n1" in result
