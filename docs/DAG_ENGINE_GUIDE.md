# DAG Engine Specification

The `DAGAgentRunner` coordinates multi-agent execution topologies using a Directed Acyclic Graph (DAG). It resolves dependencies, manages concurrent execution via an `asyncio.PriorityQueue`, evaluates conditional routing logic, and maintains a mutable global state bus.

## 1. Core Components

### DAGTask
Defines the configuration for a single node in the graph.

```python
@dataclass
class DAGTask:
    runner: AgentRunner
    prompt: str
    config: RunnerConfig | None = None
    max_retries: int | None = None
    context_assembler: Callable[[dict[str, AgentResponse], dict[str, Any]], str] | None = None
```

*   `runner`: The initialized `AgentRunner` instance.
*   `prompt`: The primary instruction passed to the runner.
*   `config`: Specific `RunnerConfig` overrides for this node.
*   `max_retries`: Re-execution limit upon API transient failures.
*   `context_assembler`: Optional state reducer function (see Section 3).

### Edges
Defines the execution order and routing logic. Edges are provided as a list of tuples:
*   `Tuple[str, str]`: Unconditional edge. The child node executes after the parent node completes.
*   `Tuple[str, str, Callable[[AgentResponse, dict], bool]]`: Conditional edge. The child node executes only if the callable evaluates to `True`.

### Shared State Bus
A global `dict[str, Any]` provided during `DAGAgentRunner` initialization. 
*   It is injected into the execution context of every tool via `context["dag_state"]`.
*   Tools can mutate this dictionary to pass data out-of-band.
*   It is passed as the second argument to all Conditional Edge callables and Context Assemblers.

## 2. Conditional Routing

Conditional edges dynamically prune graph execution paths.

```python
# Signature: Callable[[AgentResponse, dict[str, Any]], bool]
def routing_condition(resp: AgentResponse, state: dict) -> bool:
    return state.get("requires_review", False)

edges = [
    ("generate", "review", routing_condition)
]
```

*   **Pruning (SKIPPED State)**: If a condition evaluates to `False`, the target node registers the parent as `skipped`. If a node finds that *all* its incoming edges evaluated to `False`, the node's state becomes `SKIPPED`, and it does not execute. This status cascades to all its strictly dependent children.
*   **OR-Merge**: If a node has multiple incoming conditional edges, and at least one evaluates to `True`, the node executes. Its `context_assembler` will only receive the `AgentResponse` objects from the branches that evaluated to `True`.

## 3. Context Reducers (State Channels)

By default, `DAGAgentRunner` concatenates the raw string outputs of all successful parent nodes and appends them to the child node's prompt. To prevent token limit exhaustion, define a `context_assembler` in the `DAGTask`.

```python
# Signature: Callable[[dict[str, AgentResponse], dict[str, Any]], str]
def reduce_context(parents: dict[str, AgentResponse], state: dict) -> str:
    parent_a_text = parents["node_a"].text
    return f"\n\nExtracted metric: {parent_a_text[:100]}"

task = DAGTask(
    runner=my_runner,
    prompt="Analyze the metric.",
    context_assembler=reduce_context
)
```

## 4. Execution State Machine

Upon `engine.execute()`, the DAG returns a `DAGResponse` object containing a mapping of `node_id` to `DAGNodeResponse`.

| Status | Description |
| :--- | :--- |
| `SUCCESS` | Node executed and returned an `AgentResponse`. |
| `FAILED` | Node encountered an exception and exhausted all retries. |
| `FAILED_UPSTREAM` | Node execution cancelled because a required parent node resolved to `FAILED`. |
| `SKIPPED` | Node execution bypassed because all incoming conditional edges evaluated to `False`. |
| `PENDING` | Default uninitialized state. |

## 5. Persistence and Telemetry

The `DAGAgentRunner` accepts an `IPersistenceProvider` and a `session_id`. If provided, the engine will invoke `save_checkpoint(session_id, state)` upon every node completion. If initialized with an existing `session_id` containing saved data, the engine will deserialize the states and resume execution from the last pending node.

Pass `StructuredTelemetryHandler` as the `handler` argument to emit deterministic JSON logs tracking node duration, token usage, and trace IDs.