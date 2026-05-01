import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from agentic_core.tools.subagent import SpawnSubAgentsTool
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
def mock_llm():
    llm = MagicMock()
    llm.ask = MagicMock(side_effect=mock_ask_impl)
    return llm

@pytest.fixture
def mock_tm():
    tm = MagicMock()
    tm.prepare_turn = AsyncMock(return_value=None)
    tm.tool_schemas = [{"type":"function", "function":{"name":"search_tool"}}]
    tm.get_toolset_prompt.return_value = "Mock Toolset Prompt"
    tm.get_tools_from_toolset.return_value = []
    tm.get_mcp_loaded_tools.return_value = []
    return tm

@pytest.fixture
def subagent_tool():
    return SpawnSubAgentsTool()

@pytest.mark.asyncio
async def test_spawn_subagents_basic_flow(subagent_tool, mock_llm, mock_tm):
    context = {
        
        "llm_client": mock_llm,
        "tools_manager": mock_tm
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
async def test_spawn_subagents_granular_tools(subagent_tool, mock_llm, mock_tm):
    context = {
        
        "llm_client": mock_llm,
        "tools_manager": mock_tm
    }

    with patch('agentic_core.tools.subagent.DAGAgentRunner') as MockDAG:
        MockDAG.return_value.execute = AsyncMock(return_value=MagicMock(error=None))
        # To make it return something that doesn't crash the summary loop
        MockDAG.return_value.nodes = {
            "n1": MagicMock(result=MockLLMResponse(), state=MagicMock(name="SUCCESS"))
        }

        args = {
            "plan": {
                "nodes": {
                    "n1": {
                        "prompt": "Task 1",
                        "tools": ["search_tool"]
                    }
                },
                "edges": []
            }
        }
        await subagent_tool.execute(args, context)

        # Inspect the nodes_def passed to DAGAgentRunner
        nodes_def = MockDAG.call_args[0][0]
        runner, config, prompt, max_retries = nodes_def["n1"]
        assert config.tools is not None 
        assert {"type":"function", "function":{"name":"search_tool"}} in config.tools

@pytest.mark.asyncio
async def test_spawn_subagents_missing_context(subagent_tool):
    context = {} # Missing llm and tm
    args = {"plan": {"nodes": {"n1": {"prompt": "X"}}, "edges": []}}
    result = await subagent_tool.execute(args, context)
    assert "Error: Sub-agent spawning requires 'llm_client' and 'tools_manager' in the context" in result

@pytest.mark.asyncio
async def test_spawn_subagents_invalid_plan(subagent_tool):
    context = {
        
        "llm_client": MagicMock(),
        "tools_manager": MagicMock()
    }
    args = {
        "plan": {
            "nodes": {},
            "edges": []
        }
    }
    result = await subagent_tool.execute(args, context)
    assert "Validation Error" in result
