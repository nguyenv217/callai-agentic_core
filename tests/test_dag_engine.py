
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
from agentic_core.engine import AgentRunner, RunnerConfig
from agentic_core.dag_engine import DAGAgentRunner, NodeState, DAGEventObserver
from agentic_core.llm_providers.base import ILLMClient, LLMResponse
from agentic_core.tools import ToolManager
from agentic_core.memory.manager import MemoryManager

class MockLLMClient(ILLMClient):
    def __init__(self):
        self.call_counts = {}

    async def ask(self, messages, tools=None):
        last_message = messages[-1]["content"]
        node_id = "Unknown"
        # Infer node_id from prompt (hack for tests)
        for word in last_message.split():
            if word.startswith("Node_"):
                node_id = word

        self.call_counts[node_id] = self.call_counts.get(node_id, 0) + 1

        if "fail_permanent" in last_message:
            yield LLMResponse(success=False, error="Fatal error")
        elif "fail_transient" in last_message:
            # Fail first 2 times, succeed on 3rd
            if self.call_counts[node_id] <= 2:
                yield LLMResponse(success=False, error="Rate limit reached")
            else:
                yield LLMResponse(success=True, text="Recovered", reasoning="Done")
        else:
            yield LLMResponse(success=True, text="Success", reasoning="Done")

class MinimalToolManager(ToolManager):
    def __init__(self):
        super().__init__()
        self.tools_list = []

    async def prepare_turn(self, config):
        pass

    def get_tools_from_toolset(self, toolset):
        return self.tools_list

    def get_mcp_loaded_tools(self):
        return []

    def get_discovery_tools(self):
        return []

    async def execute(self, tool_name, tool_args, **kwargs):
        return "tool result"

@pytest.mark.asyncio
async def test_retry_success():
    llm = MockLLMClient()
    tools = MinimalToolManager()
    memory = MemoryManager()
    runner = AgentRunner(llm, tools, memory)
    config = RunnerConfig()

    # Node A fails twice with "rate limit", succeeds on 3rd. Max retries = 3.
    nodes_def = {
        "A": (runner, config, "Node_A fail_transient", 3),
    }
    edges = []

    dag = DAGAgentRunner(nodes_def, edges)
    results = await dag.execute()

    assert results["A"] == "SUCCESS"
    assert llm.call_counts["Node_A"] == 3

@pytest.mark.asyncio
async def test_retry_exhaustion():
    llm = MockLLMClient()
    tools = MinimalToolManager()
    memory = MemoryManager()
    runner = AgentRunner(llm, tools, memory)
    config = RunnerConfig()

    # Node A fails with "rate limit" and exhausts retries (max=1).
    nodes_def = {
        "A": (runner, config, "Node_A fail_transient", 1),
    }
    edges = []

    dag = DAGAgentRunner(nodes_def, edges)
    results = await dag.execute()

    assert results["A"] == "FAILED"
    # 1 initial + 1 retry = 2 calls
    assert llm.call_counts["Node_A"] == 2

@pytest.mark.asyncio
async def test_permanent_failure_no_retry():
    llm = MockLLMClient()
    tools = MinimalToolManager()
    memory = MemoryManager()
    runner = AgentRunner(llm, tools, memory)
    config = RunnerConfig()

    # Node A fails with "Fatal error". Should not retry even if max_retries > 0.
    nodes_def = {
        "A": (runner, config, "Node_A fail_permanent", 5),
    }
    edges = []

    dag = DAGAgentRunner(nodes_def, edges)
    results = await dag.execute()

    assert results["A"] == "FAILED"
    assert llm.call_counts["Node_A"] == 1

@pytest.mark.asyncio
async def test_multi_parent_cascade_bug():
    llm = MockLLMClient()
    tools = MinimalToolManager()
    memory = MemoryManager()
    runner1 = AgentRunner(llm, tools, memory)
    runner2 = AgentRunner(llm, tools, memory)
    runner3 = AgentRunner(llm, tools, memory)
    config = RunnerConfig()

    nodes_def = {
        "A": (runner1, config, "Node_A fail_permanent", 0),
        "B": (runner2, config, "Node_B success", 0),
        "C": (runner3, config, "Node_C", 0),
    }
    edges = [("A", "C"), ("B", "C")]

    dag = DAGAgentRunner(nodes_def, edges)
    results = await dag.execute()

    assert results["A"] == "FAILED"
    assert results["B"] == "SUCCESS"
    assert results["C"] == "FAILED_UPSTREAM"

@pytest.mark.asyncio
async def test_dag_success():
    llm = MockLLMClient()
    tools = MinimalToolManager()
    memory = MemoryManager()
    runner1 = AgentRunner(llm, tools, memory)
    runner2 = AgentRunner(llm, tools, memory)
    runner3 = AgentRunner(llm, tools, memory)
    config = RunnerConfig()

    nodes_def = {
        "A": (runner1, config, "Node_A success", 0),
        "B": (runner2, config, "Node_B success", 0),
        "C": (runner3, config, "Node_C success", 0),
    }
    edges = [("A", "B"), ("A", "C"), ("B", "C")]

    dag = DAGAgentRunner(nodes_def, edges)
    results = await dag.execute()

    assert results["A"] == "SUCCESS"
    assert results["B"] == "SUCCESS"
    assert results["C"] == "SUCCESS"
