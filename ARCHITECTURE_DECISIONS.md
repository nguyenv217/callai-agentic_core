# Architecture Decisions

## 2026-06-18: Conditional Edges in DAGAgentRunner (Control Flow Decoupling)

**Context:**
We needed a way to introduce dynamic routing (e.g., "if condition X, trigger branch Y") without coupling the execution logic inside an agent's prompt or forcing agents to use orchestrator-specific tools like `transfer_control` in DAG workflows.

**Decision:**
Implemented **Conditional Edges** natively in the `DAGAgentRunner`.
Edges can now optionally take a third parameter: a `Callable[[AgentResponse], bool]`.
When a node completes, its outbound edges evaluate this callable. 
- If `True` (or omitted), the target node receives the parent's context.
- If `False`, the branch is pruned. The target node records the parent as `skipped`. 
If a node has ALL its parent dependencies skipped, the node itself transitions to `SKIPPED` and cascades the pruning downstream. 
If a node has a mix of successful and skipped parents, it executes using only the context of the successful parents (acting as an implicit OR-merge for branched workflows).

**Rationale:**
This guarantees strict separation of concerns (Single Responsibility Principle). Agents remain "pure" computation units unaware of the global graph topology, while the graph orchestrator exclusively manages routing logic. It vastly increases the modularity and reusability of agent nodes across different swarms.