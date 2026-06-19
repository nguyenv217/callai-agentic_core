## 2026-06-19: Extensibility via Persistence Providers and Structured Telemetry

**Context:**
We recognized the framework natively bounded DAG execution to ephemeral memory processes. A reviewer accurately pointed out that without "Durable Execution," orchestrations facing OOM-kills or node rotations would lose hours of swarm computation. Additionally, console-printed telemetry proved useless for tracking latencies and token-bloat in production environments.

**Decision:**
1. **IPersistenceProvider:** We introduced a formal protocol interface allowing developers to pass Redis/Postgres/S3 wrappers to the `DAGAgentRunner`. The engine automatically intercepts execution to load historical checkpoints and saves state incrementally after every node completion.
2. **StructuredTelemetryHandler:** Implemented an event handler that bypasses standard text-logs in favor of JSON-formatted span records tracking `trace_id`, `node_id`, `duration_s`, and `usage`.

**Rationale:**
These decisions grant enterprise-level fault tolerance and observability without injecting heavy infrastructure dependencies into the core package. The framework remains zero-config out of the box but is trivially extensible for complex, long-running production environments.