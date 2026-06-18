## 2026-06-19: Fluent Builder Pattern for Agent Construction

**Context:**
As the framework's configuration surface area grew (MCP pathing, tools injections, custom memory strategies, custom handlers), the factory methods (`create_openai_agent`, `create_anthropic_agent`) suffered from significant argument bloat. It became unwieldy to extend configurations without breaking API signatures or providing monolithic constructors.

**Decision:**
Replaced the individual factory functions with a single, fluent `AgentBuilder`. 

**Rationale:**
The Builder pattern enforces cleaner, modular agent instantiation. Consumers can now arbitrarily stack configurations (`.with_tools()`, `.with_mcp()`, `.with_memory()`) without navigating long positional argument lists. It simplifies internal test setup and improves IDE auto-completion.