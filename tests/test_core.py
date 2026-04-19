import pytest
import json
import asyncio
from typing import Iterator

from agentic_core.memory.manager import MemoryManager
from agentic_core.agents.builder import create_openai_agent
from agentic_core.engine import AgentRunner, RunnerConfig
from agentic_core.interfaces.llm import ILLMClient
from agentic_core.llm_providers.base import LLMResponse
from agentic_core.observers.standard import DefaultObserver
from agentic_core.tools.base import BaseTool

# --- Mocks ---

class MockLLM(ILLMClient):
    """A mock LLM that returns a predefined sequence of responses."""
    def __init__(self, sequence: list[LLMResponse]):
        self.sequence = sequence
        self.call_count = 0
        self.model = "mock-model"

    def ask(self, messages: list[dict], tools: list[dict] | None = None, **kwargs) -> Iterator[LLMResponse]:
        if self.call_count < len(self.sequence):
            response = self.sequence[self.call_count]
            self.call_count += 1
            yield response
        else:
            yield LLMResponse(success=True, text="Mock sequence exhausted.", tool_calls=[], usage={}, error=None)

class MockWeatherTool(BaseTool):
    def __init__(self):
        super().__init__()
        self._name = "get_weather"
        self._schema = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}
            }
        }
    
    def execute(self, args: dict, context: dict) -> str:
        return f"Weather in {args.get('location', 'unknown')} is sunny and 72F."

# --- Tests ---

def test_memory_manager_initialization():
    memory = MemoryManager(max_messages=50, max_chars=80000)
    memory.set_system_prompt("You are a test agent.")
    
    assert memory.system_prompt["role"] == "system"
    assert memory.system_prompt["content"] == "You are a test agent."
    assert len(memory.messages) == 0
    
    history = memory.get_history()
    assert len(history) == 1
    assert history[0]["content"] == "You are a test agent."

def test_memory_manager_smart_truncation():
    """Ensure JSON truncation doesn't completely destroy the structure."""
    memory = MemoryManager(max_messages=10, max_chars=100) # strict limit to force truncation
    
    # Large JSON list simulation
    large_list = [{"id": i, "val": "data"} for i in range(100)]
    memory.add_message({"role": "tool", "content": json.dumps(large_list)})
    
    memory.enforce_context_limits()
    truncated_content = memory.messages[0]["content"]
    
    # Should contain the warning but preserve the beginning structure
    assert "[TRUNCATED" in truncated_content
    assert truncated_content.startswith('[{"id": 0')

@pytest.mark.asyncio
async def test_agent_runner_tool_dispatch():
    """Integration test: Agent requests a tool, tool executes, agent returns final text."""
    
    # Turn 1: LLM requests weather tool
    resp1 = LLMResponse(
        success=True, text="", error=None, usage={},
        tool_calls=[{"id": "call_123", "function": {"name": "get_weather", "arguments": '{"location": "Tokyo"}'}}]
    )
    # Turn 2: LLM provides final answer based on tool
    resp2 = LLMResponse(
        success=True, text="It is currently sunny and 72F in Tokyo.", tool_calls=[], usage={}, error=None
    )
    
    mock_llm = MockLLM([resp1, resp2])
    
    # Setup Engine with injected mock
    agent = create_openai_agent(api_key="dummy")
    agent.llm = mock_llm  
    agent.tools.register_tool(MockWeatherTool())
    
    # Execute
    config = RunnerConfig(toolset="all", clear_loaded_tool=False)
    result = await agent.run_turn("What is the weather in Tokyo?", DefaultObserver(), config=config)
    
    # Assertions
    assert "sunny and 72F in Tokyo" in result["text"]
    assert mock_llm.call_count == 2
    
    history = agent.memory.get_history()
    tool_result_msg = next((m for m in history if m.get("role") == "tool"), None)
    assert tool_result_msg is not None
    assert "sunny and 72F" in tool_result_msg["content"]