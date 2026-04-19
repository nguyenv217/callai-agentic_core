
import pytest
import json
from typing import Iterator

from agentic_core.memory.manager import MemoryManager
from agentic_core.agents.builder import create_openai_agent
from agentic_core.engine import AgentRunner, RunnerConfig
from agentic_core.interfaces.llm import ILLMClient
from agentic_core.llm_providers.base import LLMResponse
from agentic_core.observers.standard import DefaultObserver
from agentic_core.tools.base import BaseTool

# --- MOCKS ---

class MockLLM(ILLMClient):
    """Mocks an LLM to simulate a multi-turn tool interaction."""
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

class CalculatorTool(BaseTool):
    """A simple tool for testing the dispatch engine."""
    def __init__(self):
        super().__init__()
        self._name = "add_numbers"
        self._schema = {
            "type": "function",
            "function": {
                "name": "add_numbers",
                "description": "Adds two numbers together",
                "parameters": {
                    "type": "object", 
                    "properties": {
                        "a": {"type": "number"},
                        "b": {"type": "number"}
                    }
                }
            }
        }
    
    def execute(self, args: dict, context: dict) -> str:
        # Handles potential stringified JSON from the framework
        if isinstance(args, str):
            args = json.loads(args)
        result = args.get('a', 0) + args.get('b', 0)
        return json.dumps({"result": result})

# --- TESTS ---

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

@pytest.mark.asyncio
async def test_agent_execution_loop_with_tool():
    """
    Integration test: Validates that the engine can route a request to the LLM, 
    extract the tool call, execute the local Python tool, and return to the LLM.
    """
    # 1. Setup Mock Sequence
    # Turn 1: LLM decides to call the calculator
    resp1 = LLMResponse(
        success=True, text="", error=None, usage={},
        tool_calls=[{"id": "call_abc", "function": {"name": "add_numbers", "arguments": '{"a": 5, "b": 7}'}}]
    )
    # Turn 2: LLM observes the tool result and provides final text
    resp2 = LLMResponse(
        success=True, text="The sum of 5 and 7 is 12.", tool_calls=[], usage={}, error=None
    )
    
    mock_llm = MockLLM([resp1, resp2])
    
    # 2. Initialize Agent Architecture
    agent = create_openai_agent(api_key="mock_key")
    agent.llm = mock_llm  # Inject our mock
    agent.tools.register_tool(CalculatorTool())
    
    # 3. Execute
    config = RunnerConfig(toolset="all")
    result = await agent.run_turn("What is 5 plus 7?", DefaultObserver(), config=config)
    
    # 4. Assertions
    assert result["success"] is True
    assert "12" in result["text"]
    assert mock_llm.call_count == 2
    
    # Verify memory states
    history = agent.memory.get_history()
    assert history[-2]["role"] == "tool" # The tool execution was recorded
    assert "12" in history[-2]["content"] # The math result was injected to context