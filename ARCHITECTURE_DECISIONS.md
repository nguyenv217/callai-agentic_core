# Architecture Decisions

## 2026-06-18: Global State Bus in DAGAgentRunner & Subagent Re-Alignment

**Context:**
Conditional edges in `DAGAgentRunner` evaluated only localized `AgentResponse` objects, preventing holistic swarm routing decisions. Simultaneously, `SpawnSubAgentsTool` was utilizing `StatefulSwarmEngine` (a cyclic, transfer-based orchestrator) for task graphs defined with strict sequential edges, creating severe architectural mismatch.

**Decision:**
1.  **DAG State Bus:** We injected a global `shared_state: dict[str, Any]` into `DAGAgentRunner`. This state is propagated directly to all node runners via `RunnerConfig.extra_context["dag_state"]` (enabling mutation via tools) and is passed as the second argument to edge conditions (`Callable[[AgentResponse, dict], bool]`).
2.  **Subagent Re-Alignment:** We refactored `SpawnSubAgentsTool` to exclusively orchestrate via `DAGAgentRunner`. We removed `max_swarm_steps` as topological execution implicitly provides bound termination.

**Rationale:**
Bridging global state sharing into the DAG retains the deterministic, acyclic guarantees of topological completion while resolving the localized context limitations. Real-world swarms require stateful routing, but do not necessarily require unstructured, dynamic cycles. This refactor cleanly eliminates technical debt, simplifies the schema, and aligns tools with their explicitly designed orchestrators.