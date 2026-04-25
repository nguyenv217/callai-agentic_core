# DAG Agent Runner Guide

`The DAGAgentRunner` is a high-performance, asynchronous execution engine for complex agentic workflows.

## Core Concepts

#### The Graph Structure

The engine represents a workflow as a set of **Nodes** (Agents) and **Edges** (Dependencies).

*   **Nodes**: Each node contains an `AgentRunner` instance, a `RunnerConfig` configuration, a prompt, and optional retry settings.
*   **Edges**: A directed link from Node A to Node B means Node B cannot start until Node A completes successfully.

## Key Features

### 1. Critical Path Heuristic

The engine automatically calculates the "longest path" to the leaf nodes. Nodes that unlock the most downstream work are given higher priority in the execution queue.

#### 2. Async Worker Pool

Executes independent nodes concurrently using a fixed number of workers.

#### 3. Context Merging

Child nodes automatically receive the results of all their parent nodes injected into their prompt, creating a seamless data flow.

#### 4. Cascade Failure

If a node fails permanently, all its downstream dependents are automatically masked off as `FAILED_UPSTREAM` to save compute.

#### 5. Adaptive Retries

Supports exponential backoff for transient errors (e.g., Rate Limits, Timeouts).

## Usage Guide

### Basic Setup

```python
import asyncio
from agentic_core.engines import AgentRunner, RunnerConfig
from agentic_core.dag_engine import DAGAgentRunner
from agentic_core.llm_providers.openai import OpenAILLM
from agentic_core.tools import ToolManager
from agentic_core.memory.manager import MemoryManager

# Initialize shared components
llm = OpenAILLM(api_key="your_key")
tools = ToolManager()
memory = MemoryManager()
config = RunnerConfig()

# Define AgentRunners for each node
runner_research = AgentRunner(llm, tools, memory)
runner_writer = AgentRunner(llm, tools, memory)
runner_editor = AgentRunner(llm, tools, memory)

# Define the Graph
nodes_def = {
    "research_web": (runner_research, config, "Research the latest trends in AI.", 3),
    "research_docs": (runner_research, config, "Analyze internal documentation.", 3),
    "draft_article": (runner_writer, config, "Write a blog post based on research.", 2),
    "final_review": (runner_editor, config, "Proofread and edit the article.", 1),
}

edges = [
    ("research_web", "draft_article"),
    ("research_docs", "draft_article"),
    ("draft_article", "final_review"),
]

# Run the Engine
async def main():
    engine = DAGAgentRunner(nodes_def, edges, max_concurrency=2)
    results = await engine.execute()
    print(results)

asyncio.run(main())
```

### Advanced Configuration

#### `nodes_def` Parameter Breakdown

The nodes_def dictionary is the heart of your graph. Each entry follows this tuple structure: "node_id": (AgentRunner, RunnerConfig, str, int)

| Parameter | Type       | Description                                                                           |
| :-------- | :--------- | :------------------------------------------------------------------------------------ |
| AgentRunner | AgentRunner   | The logic engine for this specific node.                                                |
| RunnerConfig  | RunnerConfig  | Runtime settings (max iterations, system prompt, etc.).                                 |
| prompt        | str           | The specific instruction for this node.                                                |
| max_retries   | int           | How many times to retry on transient API errors (optional).                             |

### Monitoring with `DAGEventObserver`

You can track the execution in real-time by implementing a custom observer:

```python
from agentic_core.dag_engine import DAGEventObserver

class MyDAGObserver(DAGEventObserver):
    def on_node_start(self, node_id, worker_id):
        print(f"Node {node_id} is now running on worker {worker_id}")

    def on_node_retry(self, node_id, count, max_r):
        print(f"Node {node_id} failed. Retry {count}/{max_r}...")

    def on_node_complete(self, node_id, status, result):
        print(f"Node {node_id} finished with status: {status}")

engine = DAGAgentRunner(nodes_def, edges, observer=MyDAGObserver())
```

## Understanding Node States

At the end of execution, execute() returns a dictionary of node_id: state.

| State            | Meaning                         |
| :--------------- | :----------------------------- |
| `SUCCESS`          | Node executed and returned a valid result. |
| `FAILED`           | Node failed and exhausted all retries (or hit a fatal error). |
| `FAILED_UPSTREAM`  | Node was never executed because one of its parents failed. |
| `PENDING`          | Node was never reached (usually indicates a disconnected graph). |

## Performance Tuning

*   `max_concurrency`: Adjust this based on your LLM rate limits. High concurrency speeds up wide DAGs but may trigger more RateLimitErrors.
*   `max_retries`: Set higher for nodes using unstable APIs or large context windows prone to timeouts.
*   Graph Design: To maximize throughput, keep the graph "wide" (more parallel nodes) rather than "deep" (long chains of dependencies).