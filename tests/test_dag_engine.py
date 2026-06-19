import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
from agentic_core.handlers.dag import DAGSmartRetryHandler
from agentic_core.engines.engine import AgentRunner, RunnerConfig
from agentic_core.engines.dag_engine import DAGAgentRunner, NodeState, DAGEventHandler, DAGTask
from agentic_core.models import AgentResponse
from agentic_core.llm_providers.base import ILLMClient, LLMResponse
from agentic_core.tools import ToolManager
from agentic_core.memory.manager import MemoryManager

class MockLLMClient(ILLMClient):
    def __init__(self):
        self.call_counts = {}
        self.last_prompts = {}

    async def ask(self, messages, tools=None, **kwargs):
        last_message = messages[-1]["content"]
        node_id = "Unknown"
        for word in last_message.split():
            if word.startswith("Node_"):
                node_id = word

        self.call_counts[node_id] = self.call_counts.get(node_id, 0) + 1
        self.last_prompts[node_id] = last_message

        if "fail_permanent" in last_message:
            raise Exception("Fatal error")
        elif "fail_transient" in last_message:
            if self.call_counts[node_id] <= 2:
                raise Exception("Rate limit reached")
            else:
                yield LLMResponse(text="Recovered", reasoning="Done")
        else:
            yield LLMResponse(text="Success", reasoning="Done")

class MinimalToolManager(ToolManager):
    def __init__(self):
        super().__init__()
        self.tools_list = []
    async def prepare_turn(self, config): pass
    def get_tools_from_toolset(self, toolset): return self.tools_list
    def get_mcp_loaded_tools(self): return []
    def get_discovery_tools(self): return []
    async def execute(self, tool_name, tool_args, **kwargs): return "tool result"

@pytest.mark.asyncio
async def test_retry_success():
    llm = MockLLMClient()
    runner = AgentRunner(llm, MinimalToolManager(), MemoryManager())
    
    nodes_def = {"A": DAGTask(runner, "Node_A fail_transient", max_retries=3)}
    dag = DAGAgentRunner(nodes_def, [], handler=DAGSmartRetryHandler())
    results = await dag.execute()
    assert results.nodes["A"].state == "SUCCESS"
    assert llm.call_counts["Node_A"] == 3

@pytest.mark.asyncio
async def test_retry_exhaustion():
    llm = MockLLMClient()
    runner = AgentRunner(llm, MinimalToolManager(), MemoryManager())
    
    nodes_def = {"A": DAGTask(runner, "Node_A fail_transient", max_retries=1)}
    dag = DAGAgentRunner(nodes_def, [], handler=DAGSmartRetryHandler(fallback_on_permanent_failure=False))
    results = await dag.execute()
    assert results.nodes["A"].state == "FAILED"
    assert llm.call_counts["Node_A"] == 2

@pytest.mark.asyncio
async def test_conditional_edges():
    llm = MockLLMClient()
    runner = AgentRunner(llm, MinimalToolManager(), MemoryManager())
    
    def condition_false(res: AgentResponse, state: dict) -> bool: return False
    def condition_true(res: AgentResponse, state: dict) -> bool: return True

    nodes_def = {
        "A": DAGTask(runner, "Node_A success"),
        "B": DAGTask(runner, "Node_B success"),
        "C": DAGTask(runner, "Node_C success"),
        "D": DAGTask(runner, "Node_D success"),
    }
    edges = [("A", "B", condition_true), ("A", "C", condition_false), ("B", "D"), ("C", "D")]

    dag = DAGAgentRunner(nodes_def, edges)
    results = await dag.execute()

    assert results.nodes["A"].state == "SUCCESS"
    assert results.nodes["B"].state == "SUCCESS"
    assert results.nodes["C"].state == "SKIPPED"
    assert results.nodes["D"].state == "SUCCESS"
    
@pytest.mark.asyncio
async def test_context_assembler():
    llm = MockLLMClient()
    runner = AgentRunner(llm, MinimalToolManager(), MemoryManager())
    
    def reducer(parents, state):
        return "\nCustom Reduced: " + parents["A"].text
        
    nodes_def = {
        "A": DAGTask(runner, "Node_A success"),
        "B": DAGTask(runner, "Node_B success", context_assembler=reducer)
    }
    edges = [("A", "B")]
    
    dag = DAGAgentRunner(nodes_def, edges)
    await dag.execute()
    
    assert "Custom Reduced: Success" in llm.last_prompts["Node_B"]

@pytest.mark.asyncio
async def test_context_fallback():
    llm = MockLLMClient()
    runner = AgentRunner(llm, MinimalToolManager(), MemoryManager())
    
    nodes_def = {
        "A": DAGTask(runner, "Node_A success"),
        "B": DAGTask(runner, "Node_B success")
    }
    edges = [("A", "B")]
    
    dag = DAGAgentRunner(nodes_def, edges)
    await dag.execute()
    
    assert "Parent Context:" in llm.last_prompts["Node_B"]
    assert "Node A result: Success" in llm.last_prompts["Node_B"]
