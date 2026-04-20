# MCP (Model Context Protocol) Configuration Guide

The `RunnerConfig` provides flexible ways to integrate external MCP servers into your agent. You can choose to lazily load tools as the agent discovers them, eagerly load specific tools into the agent's immediate context, or create a sandboxed environment.

## Core MCP Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `mcp_active_servers` | `List[str]` | A list of server names to initialize. If omitted, no servers are booted until requested. |
| `mcp_preload_tools` | `List[str]` | Specific tools to inject directly into the agent's active schema on Turn 1. |
| `enable_mcp_discovery` | `bool` | If `True`, gives the agent tools to browse and load from your configured MCP servers dynamically. |

## Configuration Scenarios

### 1. Standard Agent (No MCP)
If you don't need external servers, simply ignore the MCP parameters. The agent will run using only the standard Python tools provided in the `tools` or `toolset` arguments.

```python
config = RunnerConfig(
    max_iterations=10,
    tools=[my_standard_tool]
)
```

### 2. Pure Discovery (Lazy Loading)
**Best for:** Open‑ended agents that need access to everything but shouldn't waste resources booting servers they might not use.

**Behavior:** The agent starts with only `list_mcp_catalog` and `load_mcp_tool`. Servers are initialized only if the agent decides to look at the catalog.

```python
config = RunnerConfig(
    enable_mcp_discovery=True
    # mcp_active_servers and mcp_preload_tools are left as None
)
```
> **Note:** This requires a valid MCP configuration file or previously added servers via `add_mcp_server()`.

### 3. Specific Tools Only (Eager Loading)
**Best for:** Purpose‑built agents where you know exactly which tools are needed (e.g., a "GitHub Issue Creator" agent).

**Behavior:** The specified hosting server is booted immediately, and the requested tools are pushed into the agent's context. Discovery is disabled by default.

```python
config = RunnerConfig(
    mcp_active_servers=["github"],
    mcp_preload_tools=["github_create_issue", "github_search_repositories"],
    # enable_mcp_discovery defaults to False
)
```
**Strict Validation:** If you specify `mcp_preload_tools`, you must also specify the hosting server in `mcp_active_servers`. Failure to do so raises a `ConfigurationError`.

### 4. The Sandbox (Allowlist + Discovery)
**Best for:** Security‑conscious environments where the agent may discover tools, but only within a strict subset of configured servers.

**Behavior:** Only the servers listed in `mcp_active_servers` are booted. The agent can query the catalog, but it will see tools belonging only to those allowed servers.

```python
config = RunnerConfig(
    mcp_active_servers=["github", "slack"],  # Ignores "database" or "local_fs"
    enable_mcp_discovery=True
)
```

### 5. The Power User (Preload + Sandboxed Discovery)
**Best for:** Complex workflows where an agent needs immediate access to primary tools and may later look up helper tools on allowed servers.

**Behavior:** Boots the allowed servers, preloads the specific tools for Turn 1, and also provides discovery tools to browse the rest of the allowed servers' catalogs.

```python
config = RunnerConfig(
    mcp_active_servers=["github", "slack"],
    mcp_preload_tools=["github_create_issue"],
    enable_mcp_discovery=True
)
```

## Common Configuration Errors
The framework is designed to fail fast to prevent silent resource drains. You will encounter a `ConfigurationError` if:

1. **Missing Hosting Server** – You specify a tool in `mcp_preload_tools` but forget to include its parent server in `mcp_active_servers`.
2. **Dead State** – You specify `mcp_active_servers` but leave both `mcp_preload_tools` as `None` and `enable_mcp_discovery` as `False`. (This boots servers but gives the agent no way to interact with them.)
3. **Missing Config** – You enable discovery or request preloads without actually setting up an `mcp_config_path` or using programmatic config injection.
