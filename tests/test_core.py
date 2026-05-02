import pytest
from agentic_core.agents.builder import create_openai_agent
from agentic_core.config import RunnerConfig
from agentic_core.interfaces import IterationLimitReachedError
from agentic_core.llm_providers.base import ILLMClient, LLMResponse
from agentic_core.observers.standard import SilentObserver
from agentic_core.tools.base import BaseTool
from agentic_core.decisions import DecisionEvent, ToolStartDecision, LastIterationDecision
from tests.conftest import ControlObserver

# --- TESTS ---

@pytest.mark.asyncio
async def test_tool_json_decode_error(mock_llm_class):
    """Test that invalid JSON arguments from LLM are handled gracefully."""
    resp1 = LLMResponse(
       text="", usage={},
        tool_calls=[{"id": "call_1", "function": {"name": "add_numbers", "arguments": '{"a": 5, "b":'}}], finish_reason="tool_calls"
    )
    resp2 = LLMResponse(text="I fixed the JSON.", tool_calls=[], usage={})

    mock_llm = mock_llm_class([resp1, resp2])
    agent = create_openai_agent(api_key="mock_key")
    agent.llm = mock_llm

    result = await agent.run_turn("Test JSON error", SilentObserver())

    assert result.error is None
    assert "I fixed the JSON" in result.text
    history = agent.memory.get_history()
    assert any("Invalid JSON arguments" in msg["content"] for msg in history if msg["role"] == "tool")

@pytest.mark.asyncio
async def test_max_iterations_reached(mock_llm_class):
    """Test that agent stops after max_iterations."""
    resp = LLMResponse(
       text="", usage={},
        tool_calls=[{"id": "loop", "function": {"name": "loop_tool", "arguments": "{}"}}],
        finish_reason="tool_calls"
    )
    mock_llm = mock_llm_class([resp] * 10)

    agent = create_openai_agent(api_key="mock_key")
    agent.llm = mock_llm

    class LoopTool(BaseTool):
        def __init__(self):
            super().__init__()
            self._name = "loop_tool"
            self._schema = {"type": "function", "function": {"name": "loop_tool", "parameters": {"type": "object", "properties": {}}}}
        async def execute(self, args, ctx): return "Looping..."

    agent.tools.register_tool(LoopTool())

    config = RunnerConfig(max_iterations=3)
    result = await agent.run_turn("Loop me", SilentObserver(), config=config)

    assert isinstance(result.error, IterationLimitReachedError)
    assert mock_llm.call_count == 3

@pytest.mark.asyncio
async def test_tool_exception_handling(mock_llm_class, error_tool_factory):
    """Test that tool exceptions are captured and passed back to LLM."""
    resp1 = LLMResponse(
        text="", usage={},
        tool_calls=[{"id": "fail_1", "function": {"name": "error_tool", "arguments": "{}"}}],
        finish_reason="tool_calls"
    )
    resp2 = LLMResponse(text="The tool failed, but I'm okay.", tool_calls=[], usage={}, )

    mock_llm = mock_llm_class([resp1, resp2])
    agent = create_openai_agent(api_key="mock_key")
    agent.llm = mock_llm
    agent.tools.register_tool(error_tool_factory(should_fail=True))

    result = await agent.run_turn("Fail me", SilentObserver())

    assert result.error is None
    assert "The tool failed" in result.text
    history = agent.memory.get_history()
    assert any("Tool execution failed!" in msg["content"] for msg in history if msg["role"] == "tool")

@pytest.mark.asyncio
async def test_system_prompt_combination(mock_llm_class):
    """Test that toolset prompt and explicit system prompt are merged."""
    agent = create_openai_agent(api_key="mock_key")
    agent.tools.add_toolset("my_set", [], "TOOLSET PROMPT")

    assert "my_set" in agent.tools.toolsets
    assert "my_set" in agent.tools.toolset_prompts

    config = RunnerConfig(toolset="my_set", system_prompt="USER SYSTEM PROMPT")
    agent.llm = mock_llm_class([LLMResponse(text="Hi", tool_calls=[], usage={}, )])

    await agent.run_turn("Hello", SilentObserver(), config=config)

    assert "TOOLSET PROMPT" in agent.memory.system_prompt['content']
    assert "USER SYSTEM PROMPT" in agent.memory.system_prompt['content']

@pytest.mark.asyncio
async def test_observer_skip_tool(mock_llm_class, error_tool_factory):
    """Test that the observer can skip a tool call."""
    resp1 = LLMResponse(
        text="", usage={},
        tool_calls=[{"id": "skip_1", "function": {"name": "error_tool", "arguments": "{}"}}],
        finish_reason="tool_calls"
    )
    resp2 = LLMResponse(text="Tool was skipped.", tool_calls=[], usage={}, )

    mock_llm = mock_llm_class([resp1, resp2])
    agent = create_openai_agent(api_key="mock_key")
    agent.llm = mock_llm
    agent.tools.register_tool(error_tool_factory())

    observer = ControlObserver(tool_decision=(ToolStartDecision.SKIP,))

    result = await agent.run_turn("Skip me", observer)

    assert result.error is None
    assert "Tool was skipped" in result.text
    history = agent.memory.get_history()
    assert not any(msg["role"] == "tool" for msg in history)

@pytest.mark.asyncio
async def test_observer_abandon_turn(mock_llm_class, error_tool_factory):
    """Test that the observer can abandon the turn entirely."""
    resp1 = LLMResponse(
        text="", usage={},
        tool_calls=[{"id": "abandon_1", "function": {"name": "error_tool", "arguments": "{}"}}],
        finish_reason="tool_calls"
    )

    mock_llm = mock_llm_class([resp1])
    agent = create_openai_agent(api_key="mock_key")
    agent.llm = mock_llm
    agent.tools.register_tool(error_tool_factory())

    observer = ControlObserver(tool_decision=(ToolStartDecision.ABANDON,))

    result = await agent.run_turn("Abandon me", observer)

    assert isinstance(result.error, IterationLimitReachedError)
    assert result.text == ""

