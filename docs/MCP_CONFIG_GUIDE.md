# Model Context Protocol (MCP) Configuration

The `agentic_core` framework handles MCP server lifecycles, JSON-RPC session management, and dynamic tool schema injection natively via the `ToolManager`.

## 1. Configuration Parameters

MCP behavior is controlled via the `RunnerConfig` object during agent execution.

| Parameter | Type | Description |
|-----------|------|-------------|
| `mcp_active_servers` | `list[str]` | Exact names of the servers defined in the MCP JSON config to initialize. Server processes are spawned when the execution turn begins. |
| `mcp_preload_tools` | `list[str]` | Specific tools to load into the active LLM schema before Turn 1. Format must be `{server_name}_{tool_name}`. |
| `mcp_enable_discovery` | `bool` | If `True`, injects the `list_mcp_catalog` and `load_mcp_tool` schemas into the LLM, allowing dynamic discovery. |
| `mcp_use_loaded_tools` | `bool` | If `True` (default), tools loaded during previous turns or preloaded remain in the active schema. |

## 2. Execution Profiles

### Lazy Loading (Discovery)
Servers and tools are not loaded into memory until the LLM explicitly requests them via the discovery tools.
```python
config = RunnerConfig(
    mcp_enable_discovery=True
    # mcp_active_servers and mcp_preload_tools are None
)
```

### Eager Loading (Targeted)
The specified servers are spawned immediately, and the requested tools are injected into the LLM's system prompt prior to execution. Discovery is disabled.
```python
config = RunnerConfig(
    mcp_active_servers=["github"],
    mcp_preload_tools=["github_create_issue", "github_search_repositories"]
)
```
*Validation Rule*: If a tool is declared in `mcp_preload_tools`, its parent server must be declared in `mcp_active_servers`. Failure to do so raises a `ConfigurationError`.

### Sandboxed Discovery
Initializes specific servers but requires the LLM to load the tools it needs dynamically.
```python
config = RunnerConfig(
    mcp_active_servers=["github", "slack"], 
    mcp_enable_discovery=True
)
```

## 3. Subprocess Lifecycle Management

MCP servers execute as external processes (via STDIO). The `ToolManager` must cleanly terminate these processes to prevent zombie resource consumption.

### Async Context Manager (Required Pattern)
The framework enforces lifecycle cleanup via `__aexit__`.

```python
from agentic_core.tools import ToolManager

async def execute():
    async with ToolManager(mcp_config_path="mcp.json") as tools:
        # tools.shutdown_mcp() is automatically invoked on block exit
        pass
```

### Process Tree Termination (`psutil`)
If `psutil` is installed, the framework intercepts the creation of wrapper scripts (e.g., `npx`, `uvx`) by evaluating the differential PID tree before and after `stdio_client` initialization. During shutdown, it recursively issues SIGKILL commands to the entire subprocess tree. If `psutil` is unavailable, it relies on the default MCP SDK graceful shutdown, which may fail to terminate daemonized wrapper scripts.