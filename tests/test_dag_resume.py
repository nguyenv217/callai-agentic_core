import asyncio
import pytest
from unittest.mock import MagicMock

from agentic_core.engines.engine import AgentRunner, RunnerConfig
from agentic_core.engines.dag_engine import DAGAgentRunner
from agentic_core.interfaces import AgentResponse
from agentic_core.llm_providers.base import ILLMClient, LLMResponse
from agentic_core.memory.manager import MemoryManager
from agentic_core.tools import ToolManager

class MockResumeLLMClient(ILLMClient):
    """Mocks an LLM to verify which nodes get executed."""
    def __init__(self):
        self.call_counts = {}
        self.last_prompts = {}

    async def ask(self, messages, tools=None, **kwargs):
        last_message = messages[-1]["content"]
        node_id = "Unknown"
        if "Node_A" in last_message: node_id = "A"
        elif "Node_B" in last_message: node_id = "B"
        elif "Node_C" in last_message: node_id = "C"
        
        self.call_counts[node_id] = self.call_counts.get(node_id, 0) + 1
        self.last_prompts[node_id] = last_message
        
        yield LLMResponse(text=f"Success from {node_id}", reasoning="Done")

@pytest.mark.asyncio
async def test_dag_resumption_with_checkpoint():
    llm = MockResumeLLMClient()
    tools = ToolManager()
    memory = MemoryManager()
    runner = AgentRunner(llm, tools, memory)
    config = RunnerConfig()

    # Graph structure: A -> C, B -> C
    nodes_def = {
        "A": (runner, config, "Node_A instruction", 0),
        "B": (runner, config, "Node_B instruction", 0),
        "C": (runner, config, "Node_C instruction", 0),
    }
    edges = [("A", "C"), ("B", "C")]

    # Checkpoint state simulating Node A having already successfully completed.
    checkpoint_state = {
        "A": AgentResponse(text="Checkpointed Result for A", reasoning="Loaded from disk", tool_calls=[], usage={})
    }

    dag = DAGAgentRunner(nodes_def, edges, checkpoint_state=checkpoint_state)
    results = await dag.execute()

    # 1. Verify Node A was NOT executed by the LLM
    assert "A" not in llm.call_counts
    assert results.nodes["A"].state == "SUCCESS"
    assert results.nodes["A"].result.text == "Checkpointed Result for A"

    # 2. Verify Node B WAS executed
    assert llm.call_counts.get("B") == 1
    assert results.nodes["B"].state == "SUCCESS"
    assert results.nodes["B"].result.text == "Success from B"

    # 3. Verify Node C WAS executed
    assert llm.call_counts.get("C") == 1
    assert results.nodes["C"].state == "SUCCESS"
    
    # 4. Verify Node C received the correct context from both the checkpointed A and the executed B
    c_prompt = llm.last_prompts["C"]
    assert "Parent Context:" in c_prompt
    assert "Checkpointed Result for A" in c_prompt
    assert "Success from B" in c_prompt