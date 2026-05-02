from __future__ import annotations
from typing import TYPE_CHECKING, Any
from dataclasses import dataclass

if TYPE_CHECKING:
    from agentic_core.tools.base import ToolSchema

import logging
logger = logging.getLogger(__name__)

class ConfigurationError(Exception):
    pass

@dataclass
class RunnerConfig:
    '''
    Configuration for the AgentRunner.

    Args:
        max_iterations (int = False): The maximum number of iterations the agent can take before failing.
        max_chars (int | None = 10000): Result of each tool call is limited to `max_chars` characters to save tokens. Set to None for no limits. Default is 10000.
        system_prompt (str | None = None): A prompt to be used as the system prompt for the agent. This overwrites any manually system prompt written to the memory manager before `run_turn()`.      
        kwargs (dict[str, Any] | None = None): Any extra arguments passing to client.ask() method, e.g. `extra_body` for OpenAI.
        tools (list[ToolSchema] | None = None): A list of tools to be used by the agent. 
        toolset (str | None = None): The name of a preconfigured `toolset` registered with `AgentRunner.tools: ToolManager`. Passing `tools` will take priority over this settings to encourage clearer tools injection. Additionally, this will attach the toolset-specific prompt to system prompt (if any).
        extra_context (dict[str, Any] | None = None): Extra context to be passed to `AgentRunner.tools.execute()`.
        mcp_active_servers (list[str] | None = None): A list of MCP server names to be used for the agent. This is useful when you only want to use a specific set of servers. It is best accompanied with `mcp_preload_tools` and `enable_mcp_discovery=False` to conserve resources.
        mcp_use_loaded_tools (bool = False): Whether to use the MCP tools loaded in `ToolManager` for this turn.
        mcp_preload_tools (list[str] | None = None): A list of MCP tool names to be preloaded for the agent. This is useful when you know what MCP tools you want to use.
        mcp_enable_discovery (bool = False): Whether to enable user to dynamically browse and load MCP tools. Recommended 'False' if `mcp_preload_tools` is specified
    '''
    max_iterations: int = 20
    max_chars: int | None = 10000
    system_prompt: str | None = None
    kwargs: dict[str, Any] | None = None
    # Tool settings
    tools: list[ToolSchema] | None = None        
    toolset: str | None = None                   
    extra_context: dict[str, Any] | None = None
    # MCP (Model Context Protocol) Settings
    mcp_use_loaded_tools: bool = True
    mcp_active_servers: list[str] | None = None  # e.g. ["github", "memory"]. Supply this before supplying mcp_preload_tools.
    mcp_preload_tools: list[str] | None = None   # e.g. ["github_create_issue"]
    mcp_enable_discovery: bool = False           

    def __post_init__(self):
        if self.max_iterations < 1: 
            raise ValueError("`max_iterations` must be >= 1")
        if self.tools and self.toolset:
            logger.warning("[RunnerConfig] Both tools and toolset were specified at the same time. Will prioritize `tools`.")

        self.toolset = self.toolset or "none"   
        mcp_preload_tools = self.mcp_preload_tools or []
        mcp_active_servers = self.mcp_active_servers or []

        # Force this convention over implicit alive MCP server discovery (prone to error?) 
        if (self.mcp_preload_tools and not self.mcp_active_servers) or (not all(any(t.startswith(sn) for sn in mcp_active_servers) for t in mcp_preload_tools)):
            raise ConfigurationError("The hosting servers of some tools in `mcp_preload_tools` are not found in `mcp_active_servers`")
        