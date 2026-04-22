
import pytest
import json
import asyncio
from typing import Iterator
from agentic_core.memory.manager import MemoryManager
from agentic_core.agents.builder import create_openai_agent
from agentic_core.interfaces.config import RunnerConfig
from agentic_core.llm_providers.base import ILLMClient, LLMResponse
from agentic_core.observers.standard import DefaultObserver
from agentic_core.tools.base import BaseTool
from agentic_core.observers import DecisionEvent, ToolStartDecision, LastIterationDecision

# --- MOCKS ---

class MockLLM(ILLMClient):
    def __init__(self, sequence: list[LLMResponse]):
        self.sequence = sequence
        self.call_count = 0
        self.model = "mock-gpt"

    def ask(self, messages: list[dict], tools: list[dict] | None = None, **kwargs) -> Iterator[LLMResponse]:
        if self.call_count < len(self.sequence):
            response = self.sequence[self.call_count]
            self.call_count += 1
            yield response
        else:
            yield LLMResponse(success=False, error="Mock sequence exhausted.")

class ErrorTool(BaseTool):
    def __init__(self, should_fail=False):
        super().__init__()
        self._name = "error_tool"
        self._schema = {
            "type": "function",
            "function": {
                "name": "error_tool",
                "description": "A tool that can fail",
                "parameters": {"type": "object", "properties": {}}
            }
        }
        self.should_fail = should_fail

    async def execute(self, args: dict, context: dict) -> str:
        if self.should_fail:
            raise RuntimeError("Tool execution failed!")
        return "Success"

class ControlObserver(DefaultObserver):
    def __init__(self, tool_decision=None, last_decision=None):
        super().__init__()
        self.tool_decision = tool_decision
        self.last_decision = last_decision

    def on_tool_start(self, tool_name, tool_id, tool_args):
        if self.tool_decision:
            # tool_decision is a tuple (action, message)
            action = self.tool_decision[0]
            msg = self.tool_decision[1] if len(self.tool_decision) > 1 else None
            return DecisionEvent(action=action, message=msg)
        return super().on_tool_start(tool_name, tool_id, tool_args)

    def on_final_iteration(self):
        if self.last_decision:
            action = self.last_decision[0]
            msg = self.last_decision[1] if len(self.last_decision) > 1 else None
            return DecisionEvent(action=action, message=msg)
        return super().on_final_iteration()

# --- TESTS ---

@pytest.mark.asyncio
async def test_tool_json_decode_error():
    """Test that invalid JSON arguments from LLM are handled gracefully."""
    resp1 = LLMResponse(
        success=True, text="", error=None, usage={},
        tool_calls=[{"id": "call_1", "function": {"name": "add_numbers", "arguments": '{"a": 5, "b":'}}] 
    )
    resp2 = LLMResponse(success=True, text="I fixed the JSON.", tool_calls=[], usage={}, error=None)

    mock_llm = MockLLM([resp1, resp2])
    agent = create_openai_agent(api_key="mock_key")
    agent.llm = mock_llm

    result = await agent.run_turn("Test JSON error", DefaultObserver())

    assert result["success"] is True
    assert "I fixed the JSON" in result["text"]
    history = agent.memory.get_history()
    assert any("Invalid JSON arguments" in msg["content"] for msg in history if msg["role"] == "tool")

@pytest.mark.asyncio
async def test_max_iterations_reached():
    """Test that agent stops after max_iterations."""
    resp = LLMResponse(
        success=True, text="", error=None, usage={},
        tool_calls=[{"id": "loop", "function": {"name": "loop_tool", "arguments": "{}"}}]
    )
    mock_llm = MockLLM([resp] * 10)

    agent = create_openai_agent(api_key="mock_key")
    agent.llm = mock_llm

    class LoopTool(BaseTool):
        def __init__(self):
            super().__init__()
            self._name = "loop_tool"
            self._schema = {"type": "function", "function": {"name": "loop_tool", "parameters": {"type": "object", "properties": {}}}}
        async def execute(self, args, ctx): return "Looping..."

    agent.tool_manager.register_tool(LoopTool())

    config = RunnerConfig(max_iterations=3)
    result = await agent.run_turn("Loop me", DefaultObserver(), config=config)

    assert result["success"] is True or "error" not in result
    assert mock_llm.call_count == 3

@pytest.mark.asyncio
async def test_tool_exception_handling():
    """Test that tool exceptions are captured and passed back to LLM."""
    resp1 = LLMResponse(
        success=True, text="", error=None, usage={},
        tool_calls=[{"id": "fail_1", "function": {"name": "error_tool", "arguments": "{}"}}]
    )
    resp2 = LLMResponse(success=True, text="The tool failed, but I'm okay.", tool_calls=[], usage={}, error=None)

    mock_llm = MockLLM([resp1, resp2])
    agent = create_openai_agent(api_key="mock_key")
    agent.llm = mock_llm
    agent.tool_manager.register_tool(ErrorTool(should_fail=True))

    result = await agent.run_turn("Fail me", DefaultObserver())

    assert result["success"] is True
    assert "The tool failed" in result["text"]
    history = agent.memory.get_history()
    assert any("Tool execution failed!" in msg["content"] for msg in history if msg["role"] == "tool")

@pytest.mark.asyncio
async def test_system_prompt_combination():
    """Test that toolset prompt and explicit system prompt are merged."""
    agent = create_openai_agent(api_key="mock_key")
    agent.tool_manager.add_toolset("my_set", [], "TOOLSET PROMPT")

    assert "my_set" in agent.tool_manager.toolsets
    assert "my_set" in agent.tool_manager.toolset_prompts

    config = RunnerConfig(toolset="my_set", system_prompt="USER SYSTEM PROMPT")
    agent.llm = MockLLM([LLMResponse(success=True, text="Hi", tool_calls=[], usage={}, error=None)])

    await agent.run_turn("Hello", DefaultObserver(), config=config)

    assert "TOOLSET PROMPT" in agent.memory.system_prompt['content']
    assert "USER SYSTEM PROMPT" in agent.memory.system_prompt['content']

@pytest.mark.asyncio
async def test_observer_skip_tool():
    """Test that the observer can skip a tool call."""
    resp1 = LLMResponse(
        success=True, text="", error=None, usage={},
        tool_calls=[{"id": "skip_1", "function": {"name": "error_tool", "arguments": "{}"}}]
    )
    resp2 = LLMResponse(success=True, text="Tool was skipped.", tool_calls=[], usage={}, error=None)

    mock_llm = MockLLM([resp1, resp2])
    agent = create_openai_agent(api_key="mock_key")
    agent.llm = mock_llm
    agent.tool_manager.register_tool(ErrorTool())

    observer = ControlObserver(tool_decision=(ToolStartDecision.SKIP,))

    result = await agent.run_turn("Skip me", observer)

    assert result["success"] is True
    assert "Tool was skipped" in result["text"]
    history = agent.memory.get_history()
    assert not any(msg["role"] == "tool" for msg in history)

@pytest.mark.asyncio
async def test_observer_abandon_turn():
    """Test that the observer can abandon the turn entirely."""
    resp1 = LLMResponse(
        success=True, text="", error=None, usage={},
        tool_calls=[{"id": "abandon_1", "function": {"name": "error_tool", "arguments": "{}"}}]
    )

    mock_llm = MockLLM([resp1])
    agent = create_openai_agent(api_key="mock_key")
    agent.llm = mock_llm
    agent.tool_manager.register_tool(ErrorTool())

    observer = ControlObserver(tool_decision=(ToolStartDecision.ABANDON,))

    result = await agent.run_turn("Abandon me", observer)

    assert result["success"] is True
    assert result["text"] == ""

def test_memory_smart_truncation():
    """Validates that massive JSON arrays are safely truncated."""
    memory = MemoryManager(max_chars=150) # Set extremely low limit for testing
    
    large_payload = [{"id": i, "data": "x" * 50} for i in range(10)]
    memory.add_message({"role": "tool", "content": json.dumps(large_payload), "tool_call_id": "123"})
    
    memory.enforce_context_limits()
    truncated_text = memory.messages[-1]["content"]
    
    # It should keep the first few items but append the truncation warning
    assert "ARRAY TRUNCATED" in truncated_text
    assert truncated_text.startswith('[{"id": 0')