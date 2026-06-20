## 2026-06-19: Extensibility via Persistence Providers and Structured Telemetry

**Context:**
We recognized the framework natively bounded DAG execution to ephemeral memory processes. A reviewer accurately pointed out that without "Durable Execution," orchestrations facing OOM-kills or node rotations would lose hours of swarm computation. Additionally, console-printed telemetry proved useless for tracking latencies and token-bloat in production environments.

**Decision:**
1. **IPersistenceProvider:** We introduced a formal protocol interface allowing developers to pass Redis/Postgres/S3 wrappers to the `DAGAgentRunner`. The engine automatically intercepts execution to load historical checkpoints and saves state incrementally after every node completion.
2. **StructuredTelemetryHandler:** Implemented an event handler that bypasses standard text-logs in favor of JSON-formatted span records tracking `trace_id`, `node_id`, `duration_s`, and `usage`.

**Rationale:**
These decisions grant enterprise-level fault tolerance and observability without injecting heavy infrastructure dependencies into the core package. The framework remains zero-config out of the box but is trivially extensible for complex, long-running production environments.

## 2026-06-19: Transition to Horizontally Scalable Workflow Paradigm

**Context:**
The previous implementation of `DAGAgentRunner` was structurally bottlenecked, relying on a single-process `asyncio.PriorityQueue`, O(N^2) state persistence dumps, and a mutable shared dictionary across concurrent coroutines. This caused massive write amplification to databases, potential race conditions, and an inability to distribute tasks across horizontal workers or pause execution for Human-In-The-Loop (HITL) workflows.

**Decision:**
1. **Task Broker Abstraction**: Replaced `asyncio.PriorityQueue` with an injected `ITaskBroker` protocol, allowing seamless swapping to Redis, Celery, or RabbitMQ for distributed cluster orchestration.
2. **State Isolation**: Replaced concurrent mutability of `shared_state` with deep-copied isolated state channels and a deterministic change reducer at the node boundaries.
3. **Delta Checkpointing**: Expanded `IPersistenceProvider` to support incremental append-only `save_node_result()` to eliminate O(N^2) write bloat on massive sub-agent DAG graphs.
4. **First-Class Suspension**: Introduced `SUSPEND` lifecycle action and `StreamEventType.SUSPENDED` natively to the `AgentRunner`, pausing node execution safely for asynchronous external triggers (e.g., Human-in-the-loop webhooks).

**Rationale:**
These fixes break the single-process ceiling of `agentic_core`. Solo developers can still run it zero-config locally, but enterprise users can now confidently scale the orchestration across Kubernetes clusters.

## 2026-06-20: The Pre-Engineered Plugin Ecosystem

**Context:**
The core execution engine stripped out heavy infrastructure dependencies to prioritize execution speed and strict single-responsibility orchestration. However, this inflicted high developmental overhead on solo developers deploying to production (requiring manual database/queue adapters) and researchers running benchmarking pipelines (lacking evaluation frameworks).

**Decision:**
Instead of bloating the core package with SQL, Redis, and visualization servers, we established a strict Monorepo Plugin Ecosystem (`packages/persistence`, `packages/brokers`, `packages/evals`, `packages/studio`). We simultaneously injected a native `.to_mermaid()` visualizer directly into the base Graph orchestrator to instantly eliminate mental mapping friction during topology debugging.

**Rationale:**
This honors the Open-Closed principle. The execution engine remains pure and unopinionated, but standard operational hassles are solved entirely out-of-the-box via official, opt-in plugins.