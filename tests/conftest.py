import pytest
import json
from typing import Iterator
from agentic_core.llm_providers.base import ILLMClient, LLMResponse
from agentic_core.tools.base import BaseTool
from agentic_core.observers.standard import SilentObserver
from agentic_core.observers import DecisionEvent, ToolStartDecision

# --- SHARED MOCKS ---

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
        if isinstance(args, str):
            args = json.loads(args)
        result = args.get('a', 0) + args.get('b', 0)
        return json.dumps({"result": result})

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

class SlowTool(BaseTool):
    """A tool that simulates a slow async operation."""
    def __init__(self):
        super().__init__()
        self._name = "slow_task"
        self._schema = {
            "type": "function",
            "function": {
                "name": "slow_task",
                "description": "Simulates a slow task",
                "parameters": {"type": "object", "properties": {}}
            }
        }
    
    async def execute(self, args: dict, context: dict) -> str:
        import asyncio, time
        await asyncio.sleep(0.2)
        return json.dumps({"status": "completed", "time": time.time()})

class ControlObserver(SilentObserver):
    def __init__(self, tool_decision=None, last_decision=None):
        super().__init__()
        self.tool_decision = tool_decision
        self.last_decision = last_decision

    def on_tool_start(self, tool_name, tool_id, tool_args):
        if self.tool_decision:
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

# --- FIXTURES ---

@pytest.fixture
def mock_llm_class():
    return MockLLM

@pytest.fixture
def calculator_tool():
    return CalculatorTool()

@pytest.fixture
def error_tool_factory():
    def _make_tool(should_fail=False):
        return ErrorTool(should_fail=should_fail)
    return _make_tool

@pytest.fixture
def slow_tool():
    return SlowTool()
