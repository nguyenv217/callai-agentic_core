import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
from agentic_core.engines.engine import AgentRunner, RunnerConfig
from agentic_core.engines.dag_engine import DAGAgentRunner, NodeState, DAGEventObserver
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
            # We want the runner to return something that triggers the error handling
            # Based on the dag_engine.py: if "error" in result: raise Exception(result["error"])
            # The AgentRunner typically returns the final result.
            # To simulate a failure that the dag_engine catches, we can make the runner return a dict with "error".
            # However, the agentic_core AgentRunner.run_turn implementation might not do that.
            # Let's assume for these tests that MockLLMClient is used by a runner that handles this.
            # In reality, if the runner raises an exception, the dag_engine catches it.
            raise Exception("Fatal error")
        elif "fail_transient" in last_message:
            if self.call_counts[node_id] <= 2:
                raise Exception("Rate limit reached")
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

    nodes_def = {
        "A": (runner, config, "Node_A fail_transient", 3),
    }
    edges = []

    dag = DAGAgentRunner(nodes_def, edges)
    results = await dag.execute()

    assert results["A"]["state"] == "SUCCESS"
    assert llm.call_counts["Node_A"] == 3

@pytest.mark.asyncio
async def test_retry_exhaustion():
    llm = MockLLMClient()
    tools = MinimalToolManager()
    memory = MemoryManager()
    runner = AgentRunner(llm, tools, memory)
    config = RunnerConfig()

    nodes_def = {
        "A": (runner, config, "Node_A fail_transient", 1),
    }
    edges = []

    dag = DAGAgentRunner(nodes_def, edges)
    results = await dag.execute()

    assert results["A"]["state"] == "FAILED"
    assert llm.call_counts["Node_A"] == 2

@pytest.mark.asyncio
async def test_permanent_failure_no_retry():
    llm = MockLLMClient()
    tools = MinimalToolManager()
    memory = MemoryManager()
    runner = AgentRunner(llm, tools, memory)
    config = RunnerConfig()

    nodes_def = {
        "A": (runner, config, "Node_A fail_permanent", 5),
    }
    edges = []

    dag = DAGAgentRunner(nodes_def, edges)
    results = await dag.execute()

    assert results["A"]["state"] == "FAILED"
    assert llm.call_counts["Node_A"] == 1
    assert "Fatal error" in results["A"]["result"]
    assert results["A"]["error_details"] is not None

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

    assert results["A"]["state"] == "FAILED"
    assert results["B"]["state"] == "SUCCESS"
    assert results["C"]["state"] == "FAILED_UPSTREAM"
    assert results["C"]["failed_by"] == "A"

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

    assert results["A"]["state"] == "SUCCESS"
    assert results["B"]["state"] == "SUCCESS"
    assert results["C"]["state"] == "SUCCESS"
