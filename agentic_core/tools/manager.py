from __future__ import annotations
from typing import Awaitable, Callable, Set, Protocol, TYPE_CHECKING, Any
from pathlib import Path
import asyncio
import atexit
import inspect
import logging
import json

from agentic_core.constants import MCP_INITLIAZE_TIMEOUT

from ..config import ConfigurationError
from ..decisions import DecisionEvent, ToolOnPromptDecision
from ..interfaces import ToolSchema

if TYPE_CHECKING:
    from .base import BaseTool
    from ..engines.engine import RunnerConfig
    from .mcp.tools import MCPToolAdapter

logger = logging.getLogger(__name__)

class ToolExecutionController(Protocol):
    """Protocol for tool execution control."""
    on_chat_notified: Callable[[str], Awaitable[None]] | None = None
    on_prompt_respond: Callable[[Any], Awaitable[str]] | None = None
    on_prompt_confirmation: Callable[[Any], Awaitable[DecisionEvent[ToolOnPromptDecision]]] | None = None

class ToolManager:
    """
    Manages tool execution via a plugin architecture.
    Includes native, zero-config lazy loading for Model Context Protocol (MCP) servers.

    This class supports the async context manager protocol (`async with ToolManager(...) as tools:`),
    which is the recommended way to ensure that all connected MCP servers are gracefully
    shut down upon exit.
    """
    def __init__(
        self, 
        toolsets: dict[str, list[str]] | None = None,
        mcp_config_path: str | Path | None = None,
        enable_mcp_discovery: bool = True,
        extra_env: dict[str, str] | None = None,
        extra_context: dict[str, Any] | None = None,
        on_server_error: Callable[[str, Exception], None] | None = None,
        tenant_id: str = "default"
    ):
        """
        Initializes the ToolManager.

        Args:
            toolsets: A dictionary where the keys are the names of the toolsets and the values are lists of tool names.
            extra_context: Extra context to pass to tools when executed. Useful for state-aware tool implementations.
            mcp_config_path: The path to the MCP servers configuration file.
            enable_mcp_discovery: If True, the ToolManager will inject MCP discovery tools on each agent run.
            extra_env: Extra environment variables to pass to MCPClient initialization.
            on_server_error: Hook on event of a MCP server death. The function implementing this should expect `server_name` and the exception as arguments.
            tenant_id: The tenant ID to use for this ToolManager.
        """

        self.active_sessions = {}
        self.seed = 0
        self.extra_context = extra_context or {}

        self.tool_schemas: list[ToolSchema] = []
        self._plugins: dict[str, BaseTool] = {}
        
        self.tenant_id = tenant_id
        
        # --- MCP State ---
        self.mcp_config_path = mcp_config_path
        self.on_server_error = on_server_error
        self._mcp_config_dict = {}
        self._mcp_standby_registry: dict[str, MCPToolAdapter] = {}  
        self._mcp_loaded_tools: Set[BaseTool] = set()
        self._mcp_init_in_progress = False
        self.extra_env = extra_env
        self._mcp_manager = None
        
        # Discovery tools config
        self._discovery_tools = ["list_mcp_catalog", "load_mcp_tool"]
        self._loaded_discovery_tools = False
        self.enable_mcp_discovery = enable_mcp_discovery
        if enable_mcp_discovery:
            self._register_discovery_tools()
        
        # Toolsets initialization
        # Support optional prompts per toolset. `toolsets` can be a dict mapping toolset name to either a list of tool names
        # or a dict with keys 'tools' (list) and optional 'prompt' (str).
        self.toolsets: dict[str, list[str]] = {}
        self.toolset_prompts: dict[str, str] = {}
        if toolsets:
            for name, spec in toolsets.items():
                if isinstance(spec, dict):
                    tools = spec.get('tools', [])
                    prompt = spec.get('prompt')
                else:
                    tools = spec  # assume list of tool names
                    prompt = None
                self.toolsets[name] = list(tools)
                if prompt:
                    self.toolset_prompts[name] = prompt
        self.toolsets.setdefault('all', [])
        for ts in self.toolsets.values():
            self.toolsets['all'].extend(ts)
        self.toolsets['all'] = list(set(self.toolsets['all']))
        
        # Ensure cleanup of background threads on exit
        atexit.register(self.cleanup)


    def _register_discovery_tools(self):
        from .mcp.tools import ListMCPTools, LoadMCPTool
        self.register_tool(ListMCPTools(self))
        self.register_tool(LoadMCPTool(self))
        self._loaded_discovery_tools = True

    def register_tool(self, tool_instance: BaseTool, load_mcp=False):
        """Registers a standard tool plugin or a dynamically loaded MCP tool."""

        if tool_instance.name not in self._plugins:  # prevents hallucinates and registers the same tool multiple times
            self._plugins[tool_instance.name] = tool_instance
            self.tool_schemas.append(tool_instance.schema)
        
        if load_mcp:
            self._mcp_loaded_tools.add(tool_instance)
        # return self

    # ==========================================
    # MCP Tool Management
    # ==========================================
    
    async def initialize_mcp(self, allowed_servers: list[str] | None = None, extra_env: dict[str, str] | None = None) -> int:
        """Connects to MCP servers and pre-loads tool definitions into standby.
        Args:
            allowed_servers: 
                List of server names to connect to instead of connecting to all 
                configured servers. Defaults to None.
            extra_env: 
                Dictionary of extra environment variables to be used when 
                creating the connection to the MCP server. Defaults to None
        """
        # Dynamic import prevents crashing if user doesn't have MCP dependencies installed
        from .mcp.manager import MCPClientManager
        from .mcp.tools import MCPToolAdapter
        
        self._mcp_init_in_progress = True
        
        if not self._mcp_manager:
            self._mcp_manager = MCPClientManager(
                config=self._mcp_config_dict, # config takes priority over config_path
                config_path=self.mcp_config_path,
                on_server_death=self.on_server_error
            )

        initialized = await self._mcp_manager.initialize(
            allowed_servers=allowed_servers, 
            extra_env=extra_env or self.extra_env,
            tenant_id=self.tenant_id
        )

        self._mcp_init_in_progress = False

        if not initialized:
            logger.info("No MCP servers configured or available.")
            return -1
        
        # Populate standby registry
        mcp_tools = await self._mcp_manager.list_all_tools()
        for tool_def in mcp_tools:
            adapter = MCPToolAdapter(
                mcp_tool_def=tool_def,
                session=tool_def["session"],
                server_name=tool_def["server_name"]
            )
            self._mcp_standby_registry[adapter.name] = adapter
            
        # if len(self._mcp_standby_registry) > 0:
        #     self._mcp_initialized = True

        logger.info(f"Initialized {len(self._mcp_standby_registry)} MCP tools in standby mode.")
        return len(self._mcp_standby_registry)

    async def shutdown_mcp(self):
        if self._mcp_manager:
            await self._mcp_manager.close() # this triggers shutdown event for all active serveers
            self._mcp_manager = None

    async def ensure_mcp_initialized(self) -> None:
        """Pure async lazy-loader for MCP servers."""
        if not self.mcp_config_path or self._mcp_init_in_progress:
            return
            
        try:
            await asyncio.wait_for(self.initialize_mcp(), timeout=MCP_INITLIAZE_TIMEOUT)
            # self._mcp_initialized = True
            logger.debug("MCP servers initialized successfully.")
        except asyncio.TimeoutError:
            logger.error("MCP initialization timed out after 15 seconds.")
        except Exception as e:
            logger.error(f"MCP initialization failed: {e}")
    
    def add_mcp_server(self, server_name: str, command: str, args: list[str] = None, env: dict[str, str] = None, log_file: str = None):
        """
        Programmatically add an MCP server configuration.
        
        Args:
            server_name: Unique identifier for the server
            command: Executable command to start the server
            args: List of arguments for the command
            env: Environment variables for the server process
            log_file: Optional path to log stderr for server output. Useful for debugging and avoid conflicts with TUI apps.
        """
        if self._mcp_init_in_progress:
            logger.warning("MCP is initializing. Please wait or restart the program.")
            return 

        if args is None:
            args = []
        if env is None:
            env = {}
            
        if "mcpServers" not in self._mcp_config_dict:
            self._mcp_config_dict["mcpServers"] = {}
            
        self._mcp_config_dict["mcpServers"][server_name] = {
            "command": command,
            "args": args,
            "env": env,
            "log_file": log_file
        }
        
        # Reset MCP initialization state to force reload on next ensure_mcp_initialized call
        # self._mcp_initialized = False
        self._mcp_init_in_progress = False
        logger.info(f"Added MCP server '{server_name}' to programmatic config.")

    # ========================================
    # Cleanup abd Context Management
    # ========================================

    async def __aenter__(self):
        """Allows ToolManager to be used as a safe async context manager."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Gracefully shuts down MCP servers when exiting the context block."""
        await self.shutdown_mcp()
        
        # Unregister the atexit hook since we handled cleanup gracefully
        import atexit
        atexit.unregister(self.cleanup)

    def cleanup(self):
        """
        Synchronous atexit fallback to ensure background MCP tasks are stopped.
        Highly recommended to use `async with ToolManager(...) as tools:` instead.
        """
        if self._mcp_manager:
            logger.info("Cleaning up MCP Manager (atexit fallback)...")
            try:
                # Check if we are in an actively running event loop
                loop = asyncio.get_running_loop()
                if not loop.is_closed():
                    # Fire and forget if the loop is still alive
                    loop.create_task(self.shutdown_mcp())
            except RuntimeError:
                # No loop is running (standard atexit behavior).
                # Spin up a fresh, isolated loop strictly for the shutdown coroutine.
                try:
                    asyncio.run(self.shutdown_mcp())
                except Exception as e:
                    logger.debug(f"atexit asyncio.run cleanup encountered an issue: {e}")

    # ==========================================
    # PUBLIC API
    # ==========================================

    def get_tools_from_toolset(self, toolset: str = "all") -> list[ToolSchema]:
        """Get tools for a specific toolset."""
        return [t for t in self.tool_schemas if t['function']['name'] in self.toolsets.get(toolset, [])]

    def get_toolset_prompt(self, toolset: str) -> str | None:
        """Return the custom prompt associated with a toolset, if any."""
        return self.toolset_prompts.get(toolset)
    
    def get_discovery_tools(self) -> list[ToolSchema]:
        """Get discovery tools."""
        return [t for t in self.tool_schemas if t['function']['name'] in self._discovery_tools]

    def get_registered_tools(self) -> list[str]:
        """
        Get list of all registered tool names.
        
        Returns:
            List of tool names currently registered in the manager.
        """
        return list(self._plugins.keys())

    def unregister_tool(self, name: str) -> bool:
        """
        Unregister a tool by name.
        
        Args:
            name: The name of the tool to unregister.
            
        Returns:
            True if the tool was successfully unregistered, False if not found.
        """
        if name not in self._plugins:
            logger.warning(f"Tool '{name}' not found in registry.")
            return False
        
        del self._plugins[name]
        self.tool_schemas = [s for s in self.tool_schemas if s['function']['name'] != name]
        logger.info(f"Tool '{name}' unregistered.")
        return True

    def unload_mcp_tool(self, name: str) -> bool:
        """
        Unload an MCP tool from the loaded tools set.
        The tool remains in standby registry for potential reload.
        
        Args:
            name: The name of the MCP tool to unload.
            
        Returns:
            True if the tool was successfully unloaded, False if not found.
        """
        # Find and remove from loaded tools set
        tool_to_remove = None
        for tool in self._mcp_loaded_tools:
            if tool.name == name:
                tool_to_remove = tool
                break
        
        if tool_to_remove:
            self._mcp_loaded_tools.discard(tool_to_remove)
            # Also remove from plugins if present
            if name in list(self._plugins.keys()):
                del self._plugins[name]
                self.tool_schemas = [s for s in self.tool_schemas if s['function']['name'] != name]
            logger.info(f"MCP tool '{name}' unloaded from loaded tools.")
            return True
        
        logger.warning(f"MCP tool '{name}' not found in loaded tools.")
        return False

    def get_mcp_loaded_tools(self) -> list[ToolSchema]:
        return [t.schema for t in self._mcp_loaded_tools]
    
    def clear_loaded_tools(self):
        """Clears the list of loaded MCP tools."""
        self._mcp_loaded_tools.clear()

    def get_active_servers(self) -> list[str]:
        return self._mcp_manager.get_active_servers() if self._mcp_manager else []
    
    async def disconnect_mcp(self, server_names: list[str] | None = None):
        """
        Disconnect from MCP server(s) and wipe the tools out of registries.
        
        Args:
            server_names: List of server names to disconnect. If None, disconnects all servers.
                      If provided, only disconnects the specified servers.
        """
        for mcp_toolname, mcp_tool_instance in list(self._mcp_standby_registry.items()):
            if mcp_tool_instance.server_name in server_names:
                self._mcp_standby_registry.pop(mcp_toolname)
                self.unload_mcp_tool(mcp_toolname)

        if self._mcp_manager:
            await self._mcp_manager.disconnect(server_names)
        
        logger.info(f"Disconnected from MCP server(s): {server_names}")
    
    def add_toolset(self, name: str, tools: list[str], prompt: str | None = None):
        self.toolsets[name] = list(tools)
        if prompt:
            self.toolset_prompts[name] = prompt

    # ===================================
    # EXECUTION
    # ===================================

    async def execute(
        self, tool_name: str, args: dict, 
        controller: ToolExecutionController | None = None, 
        max_chars: int | None = 30000,
        extra_context: dict[str, Any] | None = None
        ) -> str:
        """Routes execution to the registered plugin (Standard or MCP). Executes asynchronously."""
        if tool_name not in self._plugins:
            return f"Error: Tool '{tool_name}' not found or not registered."
        
        try:
            # Load all servers if agent call discovery tools
            if tool_name in self._discovery_tools:
                await self.ensure_mcp_initialized()

            turn_extra_context = extra_context or self.extra_context

            context = {
                "active_sessions": self.active_sessions,
                "session_id": self.seed,
                "controller": controller,
                **turn_extra_context
            }

            plugin = self._plugins[tool_name]
            
            if inspect.iscoroutinefunction(plugin.execute):
                result = await plugin.execute(args, context)
            else:
                result = plugin.execute(args, context)

            self.active_sessions = context.get("active_sessions", self.active_sessions)

            # Context limit enforcement
            result_str = str(result)
            if max_chars:
                logger.info(f"Tool result: {result_str}")
                if len(result_str) > max_chars:
                    result_str = (
                        result_str[:max_chars] +
                        f"\n\n... [Output truncated to save context window]"
                    )

            return result_str

        except Exception as e:
            logger.exception(f"Tool execution failed for {tool_name}")
            return f"Error parsing or executing tool arguments: {e}"

    async def prepare_turn(self, config: RunnerConfig):
        """Forces MCP initialization and preloads requested tools before turn 1."""
        
        config_exists = (self.mcp_config_path or len(self._mcp_config_dict) > 0)

        if not config.mcp_preload_tools and not config.mcp_active_servers:
            if config.mcp_enable_discovery:
                if not config_exists: 
                    raise ConfigurationError(f"Please supply valid MCP server configuration to enable MCP discovery. See `docs/MCP_config.json` for further discussion on security concerns.")
                elif not self._loaded_discovery_tools:
                    self._register_discovery_tools()
            else:
                return
        
        elif config.mcp_preload_tools and not config.mcp_active_servers:
            if config_exists: 
                raise ConfigurationError("`config.mcp_preload_tools` is supplied but not `mcp_active_servers`. Please explicitly pass `mcp_active_servers` at least containing corresponding hosting servers.")
            raise ConfigurationError("No MCP configuration found.")
        
        elif not config.mcp_preload_tools and config.mcp_active_servers:
            if not config_exists:
                raise ConfigurationError("No MCP configuration found.")

            if not self._loaded_discovery_tools:
                self._register_discovery_tools()  

            await self.initialize_mcp(allowed_servers=config.mcp_active_servers) # allowlist

        else:
            # If we reached this stage, RunnerConfig __post__init__ already validated whether every suppled tool has its accompanying hosting servers supplied as well
            await self.initialize_mcp(allowed_servers=config.mcp_active_servers)

            # Preload the specific tools directly into the active set
            if config.mcp_preload_tools:
                registry = self._mcp_standby_registry
                for tool_name in config.mcp_preload_tools:
                    if tool_name in registry:
                        adapter = registry[tool_name]
                        self.register_tool(adapter, load_mcp=True)
                    else:
                        logger.warning(f"MCP tool '{tool_name}' not found in standby registry during preload.")

            if config.mcp_enable_discovery and not self._loaded_discovery_tools:
                self._register_discovery_tools()
