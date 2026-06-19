# Architecture Decisions

## 2026-06-19: Complete Removal of StatefulSwarmEngine

**Context:**
We previously marked `StatefulSwarmEngine` as deprecated after identifying its concurrency bottlenecks (centralized lock) and "Prompt-as-a-Database" flaws. Maintaining two orchestration engines creates cognitive overhead, splits maintenance, and represents architectural bloat.

**Decision:**
`StatefulSwarmEngine` has been completely deleted. `DAGAgentRunner` is now the sole orchestration engine for both structured pipelines and complex swarms.

**Rationale:**
With the introduction of Conditional Edges, the Global State Bus, and Context Reducers, the `DAGAgentRunner` possesses all the primitives necessary to build advanced, state-aware agentic workflows deterministically. Excising the legacy swarm engine enforces a single, mathematically sound source of truth for the framework.

## 2026-06-19: Deprecation of StatefulSwarmEngine & Introduction of DAG Context Reducers

**Context:**
The framework suffered from "State Fragmentation" and a "Prompt-as-a-Database" concurrency anti-pattern. `StatefulSwarmEngine` utilized a centralized `asyncio.Lock()` to serialize global state into raw text, which neutralized async parallelism and caused extreme token bloat. Similarly, `DAGAgentRunner` blindly concatenated parent results, leading to context window exhaustion. A reviewer accurately criticized these as severe limitations for distributed, production-grade applications.

**Decision:**
1. **Deprecated `StatefulSwarmEngine`:** Officially marked as deprecated. The framework now exclusively recommends `DAGAgentRunner` paired with Conditional Edges and the Global State Bus for routing. We removed the unnecessary `state_lock` in the legacy swarm engine since dictionary operations are synchronously safe in asyncio loops.
2. **Context Reducers (State Channels):** Upgraded `DAGAgentRunner` to accept a 5th tuple parameter in `nodes_def`: a `context_assembler` (`Callable[[dict[str, AgentResponse], dict[str, Any]], str]`). Instead of forcing raw text accumulation, developers can now write custom reducers that intelligently extract, summarize, or prune parent outputs and global state before passing it to the agent.

**Rationale:**
By shifting away from a monolithic, unstructured Swarm to a deterministically routed DAG with explicit Data Reducers, we embrace our identity as a lightweight, highly performant, infrastructure-efficient MCP micro-framework. It directly resolves the token bloat and concurrency bottlenecks while providing developers with professional-grade state manipulation controls.

## 2026-06-19: Fluent Builder Pattern for Agent Construction

**Context:**
As the framework's configuration surface area grew (MCP pathing, tools injections, custom memory strategies, custom handlers), the factory methods (`create_openai_agent`, `create_anthropic_agent`) suffered from significant argument bloat. It became unwieldy to extend configurations without breaking API signatures or providing monolithic constructors.

**Decision:**
Replaced the individual factory functions with a single, fluent `AgentBuilder`. 

**Rationale:**
The Builder pattern enforces cleaner, modular agent instantiation. Consumers can now arbitrarily stack configurations (`.with_tools()`, `.with_mcp()`, `.with_memory()`) without navigating long positional argument lists. It simplifies internal test setup and improves IDE auto-completion.

## 2026-06-18: Global State Bus in DAGAgentRunner & Subagent Re-Alignment

**Context:**
Conditional edges in `DAGAgentRunner` evaluated only localized `AgentResponse` objects, preventing holistic swarm routing decisions. Simultaneously, `SpawnSubAgentsTool` was utilizing `StatefulSwarmEngine` (a cyclic, transfer-based orchestrator) for task graphs defined with strict sequential edges, creating severe architectural mismatch.

**Decision:**
1.  **DAG State Bus:** We injected a global `shared_state: dict[str, Any]` into `DAGAgentRunner`. This state is propagated directly to all node runners via `RunnerConfig.extra_context["dag_state"]` (enabling mutation via tools) and is passed as the second argument to edge conditions (`Callable[[AgentResponse, dict], bool]`).
2.  **Subagent Re-Alignment:** We refactored `SpawnSubAgentsTool` to exclusively orchestrate via `DAGAgentRunner`. We removed `max_swarm_steps` as topological execution implicitly provides bound termination.

**Rationale:**
Bridging global state sharing into the DAG retains the deterministic, acyclic guarantees of topological completion while resolving the localized context limitations. Real-world swarms require stateful routing, but do not necessarily require unstructured, dynamic cycles. This refactor cleanly eliminates technical debt, simplifies the schema, and aligns tools with their explicitly designed orchestrators.

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