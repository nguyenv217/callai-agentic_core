# Graph Engine Specification

*Note: `DAGAgentRunner` has been upgraded and renamed to `GraphAgentRunner` to reflect its capability to execute arbitrary cyclic graphs natively. `DAGAgentRunner` remains as an alias for backwards compatibility.*

The `GraphAgentRunner` coordinates multi-agent execution topologies using generalized Cyclic Graphs with Petri Net semantics. It resolves dependencies, automatically detects loops (back-edges), manages concurrent execution via an abstract `ITaskBroker` queue, evaluates conditional routing logic, and maintains an isolated, horizontally-scalable state bus.

## 1. Core Components

### DAGTask (Node Definition)
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
*   `context_assembler`: Optional state reducer function (see Section 4).

### Edges
Defines the execution order and routing logic. Edges are provided as a list of tuples:
*   `Tuple[str, str]`: Unconditional edge. The child node executes after the parent node completes.
*   `Tuple[str, str, Callable[[AgentResponse, dict], bool]]`: Conditional edge. The child node executes only if the callable evaluates to `True`.

### Distributed State Bus
A global `dict[str, Any]` provided during `GraphAgentRunner` initialization.
*   It is injected into the execution context of every tool via `context["dag_state"]`.
*   **Concurrency Safe**: Nodes do not mutate the shared reference directly. They mutate a deep-copied localized state slice. Changes are deterministically reconciled and merged under a lock at the node boundary, completely eliminating race conditions.
*   It is passed as the second argument to all Conditional Edge callables and Context Assemblers.

## 2. Cyclic Routing & Loopbacks (Petri Net Semantics)

Unlike early iterations that were restricted to Directed Acyclic Graphs (DAGs), `GraphAgentRunner` natively supports cycles, loops, and recursive consensus structures.

During `compile()`, the engine automatically categorizes edges using Depth-First Search (DFS):
*   **Forward Edges (AND-joins)**: Standard dependencies. A node waits until all incoming forward edges have resolved (either `SUCCESS` or `SKIPPED`).
*   **Back Edges (OR-joins / Loops)**: Cycles in the graph. When a conditional back-edge evaluates to `True`, it triggers a "Reset Wave". 
    *   The loop body (all nodes in the cycle) is mathematically cleared of its previous state.
    *   The target node of the back-edge is instantly queued for re-execution.
    *   Outer dependencies (forward edges from outside the loop) are preserved.

## 3. Conditional Routing

Conditional edges dynamically prune graph execution paths and control loops.

```python
# Signature: Callable[[AgentResponse, dict[str, Any]], bool]
def routing_condition(resp: AgentResponse, state: dict) -> bool:
    return state.get("requires_review", False)

edges = [
    ("generate", "review", routing_condition),
    ("review", "generate") # Back-edge automatically handled!
]
```

*   **Pruning (SKIPPED State)**: If a condition evaluates to `False`, the target node registers the parent as `skipped`. If a node finds that *all* its incoming forward edges evaluated to `False`, the node's state becomes `SKIPPED`, and it does not execute. This status cascades to all its strictly dependent children.

## 4. Context Reducers (State Channels)

By default, `GraphAgentRunner` concatenates the raw string outputs of all successful parent nodes and appends them to the child node's prompt. To prevent token limit exhaustion, define a `context_assembler`.

```python
# Signature: Callable[[dict[str, AgentResponse], dict[str, Any]], str]
def reduce_context(parents: dict[str, AgentResponse], state: dict) -> str:
    parent_a_text = parents["node_a"].text
    return f"\n\nExtracted metric: {parent_a_text[:100]}"
```

## 5. Persistence, Telemetry, and Scaling

The `GraphAgentRunner` accepts an `IPersistenceProvider` and a `session_id`. It features **Delta Checkpointing**, where the `save_node_result` is invoked incrementally upon each node's completion. This O(1) write operation eliminates massive serialization bloat in long-running orchestrations.

The execution queue is fully abstracted. By injecting a custom `ITaskBroker` (e.g., pointing to Redis or Celery), the engine transitions from a local single-process loop to a fully distributed, horizontally scaled Kubernetes orchestration.