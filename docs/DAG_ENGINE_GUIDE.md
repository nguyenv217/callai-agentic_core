# DAG Agent Runner Guide

`DAGAgentRunner` is a high-performance, state-aware asynchronous execution engine for complex agentic workflows.

## Core Concepts

The engine represents a workflow as a **Directed Acyclic Graph (DAG)** of **Nodes** (Agents) and **Edges** (Dependencies/Routing Logic).

*   **Nodes**: Each node contains an `AgentRunner` instance, a `RunnerConfig`, a prompt, and optional retry settings.
*   **Edges**: A directed link from Node A to Node B means Node B evaluates whether to execute based on Node A's completion.
*   **Shared State Bus**: A global dictionary (`shared_state`) injected into all tools (via `context["dag_state"]`) and evaluated by edges.

## Key Features

### 1. Conditional Edges (Dynamic Routing)
Edges are not just static dependencies. You can provide a `Callable[[AgentResponse, dict], bool]` to an edge. When Node A finishes, the engine evaluates the condition. If it returns `False`, the branch to Node B is pruned (`SKIPPED`), cascading downstream until an implicit OR-merge resolves it.

### 2. State-Aware Execution
Nodes do not rely solely on LLM context windows to pass complex data (like large JSONs or database IDs). Tools can mutate the `shared_state` bus, and downstream nodes can read from it natively.

### 3. Critical Path Heuristic & Async Workers
The engine automatically calculates the longest path to the leaf nodes. Nodes that unlock the most downstream work execute first across an async worker pool.

### 4. Adaptive Retries & Cascade Failures
Exponential backoff handles transient API errors. If a node fails permanently, its downstream dependents are automatically masked as `FAILED_UPSTREAM`.

### 5. Context Reducers (State Channels)
To prevent "Prompt-as-a-Database" token bloat, `DAGAgentRunner` supports custom Context Assemblers (`State Reducers`). Instead of blindly concatenating all parent outputs into the prompt, you can define a `Callable[[dict[str, AgentResponse], dict], str]` that prunes, formats, or aggregates upstream dependencies and the global state.

## Usage Guide

### Basic Setup with Conditional Edges and State Bus

```python
import asyncio
from agentic_core.engines import AgentRunner
from agentic_core.config import RunnerConfig
from agentic_core.engines.dag_engine import DAGAgentRunner
from agentic_core.llm_providers.openai import OpenAILLM
from agentic_core.tools import ToolManager
from agentic_core.memory.manager import MemoryManager
from agentic_core.models import AgentResponse

# 1. Initialize shared components
llm = OpenAILLM(api_key="your_key", model="gpt-4o")
tools = ToolManager()
memory = MemoryManager()
config = RunnerConfig()

runner = AgentRunner(llm, tools, memory)

# 2. Define the Graph Nodes
def reduce_research(parents: dict, state: dict) -> str:
    # Safely extract just the text we care about to prevent token bloat
    return f"\n\nHere is the research to review: {parents['research'].text[:500]}..."

nodes_def = {
    "research": (runner, config, "Research the latest trends in AI.", 3),
    "review": (runner, config, "Review the research. Set 'approved': True in dag_state if good.", 2, reduce_research), # 5th param is the context assembler
    "publish": (runner, config, "Publish the article.", 1),
    "revise": (runner, config, "Revise the research based on feedback.", 1),
}

# 3. Define Conditional Logic
def is_approved(resp: AgentResponse, state: dict) -> bool:
    return state.get("approved", False) == True

def needs_revision(resp: AgentResponse, state: dict) -> bool:
    return not is_approved(resp, state)

# 4. Define Edges
edges = [
    ("research", "review"),
    ("review", "publish", is_approved),    # Only runs if approved
    ("review", "revise", needs_revision)   # Only runs if rejected
]

# 5. Execute
async def main():
    global_state = {"approved": False}
    engine = DAGAgentRunner(nodes_def, edges, max_concurrency=2, shared_state=global_state)
    results = await engine.execute()
    print(results.to_dict())

asyncio.run(main())
```

### Return Type

`DAGAgentRunner.execute()` returns a `DAGResponse` object containing the node results and any fatal execution errors.

```python
class DAGNodeResponse:
    state: str
    result: AgentResponse | None
    error: BaseException | None
    error_details: str | None
    failed_by: str | None

class DAGResponse:
    nodes: dict[str, DAGNodeResponse]
    error: BaseException | None
```

### Advanced Configuration

#### `nodes_def` Parameter Breakdown

The nodes_def dictionary is the heart of your graph. Each entry follows this tuple structure: "node_id": (AgentRunner, RunnerConfig, str, int)

| Parameter | Type       | Description                                                                           |
| :-------- | :--------- | :------------------------------------------------------------------------------------ |
| `AgentRunner` | `AgentRunner`   | The logic engine for this specific node.                                                |
| `RunnerConfig`  | `RunnerConfig`  | Runtime settings (max iterations, system prompt, etc.).                                 |
| `prompt      `  | `str`           | The specific instruction for this node.                                                |
| `max_retries `  | `int`           | How many times to retry on transient API errors (optional).                             |
| `context_assembler` | `Callable` | Optional reducer `[[dict, dict], str]` to format parent outputs and prevent token bloat. |

### Monitoring with `DAGEventHandler`

You can track the execution in real-time by implementing a custom handler:

```python
from agentic_core.dag_engine import DAGEventHandler

class MyDAGHandler(DAGEventHandler):
    def on_node_start(self, node_id, worker_id):
        print(f"Node {node_id} is now running on worker {worker_id}")

    def on_node_retry(self, node_id, count, max_r):
        print(f"Node {node_id} failed. Retry {count}/{max_r}...")

    def on_node_complete(self, node_id, status, result):
        print(f"Node {node_id} finished with status: {status}")

engine = DAGAgentRunner(nodes_def, edges, handler=MyDAGHandler())
```


## Understanding Node States

| State            | Meaning                         |
| :--------------- | :----------------------------- |
| `SUCCESS`          | Node executed and returned a valid result. |
| `FAILED`           | Node failed and exhausted all retries (or hit a fatal error). |
| `FAILED_UPSTREAM`  | Node was never executed because one of its parents failed. |
| `SKIPPED`          | Node was pruned because all of its incoming conditional edges evaluated to False. |
| `PENDING`          | Node was never reached (usually indicates a disconnected graph). |

## Performance Tuning

*   `max_concurrency`: Adjust this based on your LLM rate limits. High concurrency speeds up wide DAGs but may trigger more RateLimitErrors.
*   `max_retries`: Set higher for nodes using unstable APIs or large context windows prone to timeouts.
*   **Graph Design**: To maximize throughput, keep the graph "wide" (more parallel nodes) rather than "deep" (long chains of dependencies). Use the shared state bus to prevent context pollution between unrelated nodes.
