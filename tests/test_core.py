
import pytest
import json
from typing import Iterator

from agentic_core.memory.manager import MemoryManager
from agentic_core.agents.builder import create_openai_agent
from agentic_core.config import RunnerConfig
from agentic_core.llm_providers.base import ILLMClient
from agentic_core.llm_providers.base import LLMResponse
from agentic_core.observers.standard import SilentObserver
from agentic_core.tools.base import BaseTool

# --- MOCKS moved to conftest.py ---
# --- TESTS ---

@pytest.mark.asyncio
async def test_agent_execution_loop_with_tool(mock_llm_class, calculator_tool):
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
    
    mock_llm = mock_llm_class([resp1, resp2])
    
    # 2. Initialize Agent Architecture
    agent = create_openai_agent(api_key="mock_key")
    agent.llm = mock_llm  # Inject our mock
    agent.tools.register_tool(calculator_tool)
    
    # 3. Execute
    config = RunnerConfig(toolset="all")
    result = await agent.run_turn("What is 5 plus 7?", SilentObserver(), config=config)
    
    # 4. Assertions
    assert result["success"] is True
    assert "12" in result["text"]
    assert mock_llm.call_count == 2
    
def test_memory_default_tool_truncation():
    """Validates that massive JSON arrays are safely truncated."""
    memory = MemoryManager(max_chars=150) # Set extremely low limit for testing
    
    large_payload = [{"id": i, "data": "x" * 50} for i in range(10)]
    memory.add_message({"role": "tool", "content": json.dumps(large_payload), "tool_call_id": "123"})
    
    memory.enforce_context_limits()
    truncated_text = memory.messages[-1]["content"]
    
    # It should keep the first few items but append the truncation warning
    assert "ARRAY TRUNCATED" in truncated_text
    assert truncated_text.startswith('[{"id": 0')
    
# --- PARALLEL TOOL EXECUTION TEST ---
# --- MOCKS moved to conftest.py ---
@pytest.mark.asyncio
async def test_parallel_tool_execution_timing(mock_llm_class, slow_tool):
    """Ensures multiple tool calls are executed concurrently."""
    import time
    # LLM returns two tool calls in one turn, then final text
    resp1 = LLMResponse(
        success=True, text="", error=None, usage={},
        tool_calls=[
            {"id": "t1", "function": {"name": "slow_task", "arguments": "{}"}},
            {"id": "t2", "function": {"name": "slow_task", "arguments": "{}"}}
        ]
    )
    resp2 = LLMResponse(
        success=True, text="All tasks completed.", tool_calls=[], usage={}, error=None
    )
    mock_llm = mock_llm_class([resp1, resp2])
    agent = create_openai_agent(api_key="mock_key")
    agent.llm = mock_llm
    agent.tools.register_tool(slow_tool)
    config = RunnerConfig(toolset="all")
    start = time.monotonic()
    result = await agent.run_turn("Run two slow tasks.", SilentObserver(), config=config)
    duration = time.monotonic() - start
    assert result["success"] is True
    # Both tasks should finish in roughly the time of the longest one (~0.2s), not sum (~0.4s)
    assert duration < 0.25, f"Parallel execution took too long: {duration}s"

    history = agent.memory.get_history()
    
    assert history[-2]["role"] == "tool" # The tool execution was recorded
    
    assert all("completed" in msg['content'] for msg in history[-3:-1]) # The math result was injected to context
