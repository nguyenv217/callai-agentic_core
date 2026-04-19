from __future__ import annotations
from typing import Dict, List, Any, Callable, Optional, Set, TYPE_CHECKING
import asyncio
import threading
import concurrent.futures
import atexit
import inspect
import logging
import json

if TYPE_CHECKING:
    from .base import BaseTool
    from ..engine import RunnerConfig

logger = logging.getLogger(__name__)

MCP_INIT_TIMEOUT = 30

class ToolExecutionController:
    """Protocol for tool execution control."""
    on_chat_notified: Callable[[str], None] | None = None
    on_prompt_respond: Callable[[str], str] | None = None
    on_prompt_confirmation: Callable[[str, Callable, Callable], None] | None = None


class ToolManager:
    """
    Manages tool execution via a plugin architecture.
    Includes native, zero-config lazy loading for Model Context Protocol (MCP) servers.
    """
    def __init__(
        self, 
        search_api_key=None, 
        e2b_api_key=None, 
        firecrawl_api_key=None, 
        sandbox_dir="sandbox/",
        toolsets: dict[str, list[str]] | None = None,
        mcp_config_path: Optional[str] = None
    ):
        self.search_api_key = search_api_key
        self.e2b_api_key = e2b_api_key
        self.firecrawl_api_key = firecrawl_api_key
        self.sandbox_dir = sandbox_dir

        self.active_sessions = {}
        self.seed = 0

        self.tools_schema = []
        self._plugins: dict[str, BaseTool] = {}

        # --- MCP State ---
        self.mcp_config_path = mcp_config_path
        self._mcp_standby_registry = {}  
        self._mcp_loaded_tools: Set[BaseTool] = set()
        self._mcp_initialized = False
        self._mcp_init_in_progress = False
        self._universal_tools = ["list_mcp_catalog", "load_mcp_tool"]
        # Register the universal meta-tools that allow the LLM to search/load MCPs
        from .mcp import ListMCPTools, LoadMCPTool
        self.register_tool(ListMCPTools(self))
        self.register_tool(LoadMCPTool(self))
        
        # Toolsets initialization (Base logic)
        self.toolsets = toolsets or {}
        self.toolsets.setdefault('all', [])
        for ts in self.toolsets.values():
            self.toolsets['all'].extend(ts)
        self.toolsets['all'] = list(set(self.toolsets['all']))
        
        # Ensure cleanup of background threads on exit
        atexit.register(self.cleanup)


    def register_tool(self, tool_instance: BaseTool, load_mcp=False):
        """Registers a standard tool plugin or a dynamically loaded MCP tool."""
        self._plugins[tool_instance.name] = tool_instance
        self.tools_schema.append(tool_instance.schema)
        
        if load_mcp:
            self._mcp_loaded_tools.add(tool_instance)
        # return self

    # ==========================================
    # MCP NATIVE SUPPORT (Hidden from end-user)
    # ==========================================
    
    async def initialize_mcp(self, allowed_servers: list[str] | None = None):
        """Connects to MCP servers and pre-loads tool definitions into standby."""
        # Dynamic import prevents crashing if user doesn't have MCP dependencies installed
        try:
            from .mcp import MCPClientManager, MCPToolAdapter
        except ImportError:
            logger.warning("MCP dependencies missing. Skipping MCP initialization.")
            return 0
        
        mcp_manager = MCPClientManager(config_path=self.mcp_config_path)
        initialized = await mcp_manager.initialize(allowed_servers=allowed_servers)
        
        if not initialized:
            logger.info("No MCP servers configured or available.")
            return 0
        
        # Populate standby registry
        mcp_tools = await mcp_manager.list_all_tools()
        for tool_def in mcp_tools:
            adapter = MCPToolAdapter(
                mcp_tool_def=tool_def,
                session=tool_def["session"],
                server_name=tool_def["server_name"]
            )
            self._mcp_standby_registry[adapter.name] = adapter
        
        self._mcp_manager = mcp_manager

        if len(self._mcp_standby_registry) > 0:
            self._mcp_initialized = True

        logger.info(f"Initialized {len(self._mcp_standby_registry)} MCP tools in standby mode.")
        return len(self._mcp_standby_registry)

    async def shutdown_mcp(self):
        if hasattr(self, '_mcp_manager') and self._mcp_manager:
            await self._mcp_manager.close()
            self._mcp_manager = None

    async def ensure_mcp_initialized(self) -> None:
        """Pure async lazy-loader for MCP servers."""
        if not getattr(self, 'mcp_config_path', None) or self._mcp_initialized or self._mcp_init_in_progress:
            return
            
        self._mcp_init_in_progress = True
        
        try:
            await asyncio.wait_for(self.initialize_mcp(), timeout=15.0)
            self._mcp_initialized = True
            logger.debug("MCP servers initialized successfully.")
        except asyncio.TimeoutError:
            logger.error("MCP initialization timed out after 15 seconds.")
        except Exception as e:
            logger.error(f"MCP initialization failed: {e}")
        finally:
            self._mcp_init_in_progress = False

    def cleanup(self):
        """Ensures the background MCP loop is safely killed on application exit."""
        if hasattr(self, '_mcp_loop') and self._mcp_loop.is_running():
            try:
                future = asyncio.run_coroutine_threadsafe(self.shutdown_mcp(), self._mcp_loop)
                future.result(timeout=5.0)
            except Exception:
                pass
            finally:
                self._mcp_loop.call_soon_threadsafe(self._mcp_loop.stop)
                if hasattr(self, '_mcp_thread'):
                    self._mcp_thread.join(timeout=2.0)

    # ==========================================
    # PUBLIC API
    # ==========================================

    def get_tools(self, toolset="all"):
        """Get tools for a specific toolset, triggering lazy MCP init if needed."""
        # Get base tools
        tools = [t for t in self.tools_schema if t['function']['name'] in self.toolsets.get(toolset, [])]
        
        # Always inject dynamically loaded MCP tools
        tools.extend([t.schema for t in self._mcp_loaded_tools])
        
        # Always inject universal MCP meta-tools if MCP is active
        existing_names = [t['function']['name'] for t in tools]
        for tool_name in self._universal_tools:
            if tool_name not in existing_names and tool_name in self._plugins:
                tools.append(self._plugins[tool_name].schema)
        
        return tools
    
    def clear_loaded_tools(self):
        """Clears the list of loaded MCP tools."""
        self._mcp_loaded_tools.clear()

    async def execute(self, tool_name: str, args: dict, controller: ToolExecutionController) -> str:
        """Routes execution to the registered plugin (Standard or MCP). Executes asynchronously."""
        if tool_name not in self._plugins:
            return f"Error: Tool '{tool_name}' not found or not registered."
        
        # Handle double-serialized JSON arguments cleanly
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                return f"Error: Tool arguments must be valid JSON. Received: {args}"

        try:
            param_schemas = self._plugins[tool_name].schema['function']['parameters'].get('properties', {})
    
            for key, value in args.items():
                # Check if the tool actually expects a complex type (array/object)
                expected_type = param_schemas.get(key, {}).get('type')
                
                if expected_type in ['array', 'object'] and isinstance(value, str):
                    try:
                        # Only "fix" it if the tool expects a non-string type
                        args[key] = json.loads(value)
                        logger.info(f"Fixed double-serialized {expected_type} for '{key}'")
                    except (json.JSONDecodeError, TypeError):
                        # If it's not valid JSON, the server will handle the error 
                        # during its own validation phase. (or not, and it will returns an error)
                        pass

            if tool_name in self._universal_tools:
                await self.ensure_mcp_initialized()

            context = {
                "search_api_key": self.search_api_key,
                "e2b_api_key": self.e2b_api_key,
                "active_sessions": self.active_sessions,
                "session_id": self.seed,
                "sandbox_dir": self.sandbox_dir,
                "controller": controller,
                "firecrawl_api_key": self.firecrawl_api_key
            }

            plugin = self._plugins[tool_name]
            
            if inspect.iscoroutinefunction(plugin.execute):
                result = await plugin.execute(args, context)
            else:
                result = plugin.execute(args, context)

            self.active_sessions = context.get("active_sessions", self.active_sessions)

            # Context limit enforcement
            MAX_CHARS = 10000
            result_str = str(result)
            logger.info(f"Tool result: {result_str}")
            if len(result_str) > MAX_CHARS:
                result_str = (
                    result_str[:MAX_CHARS] +
                    f"\n\n... [Output truncated to {MAX_CHARS} characters to save context window]"
                )

            return result_str

        except Exception as e:
            logger.exception(f"Tool execution failed for {tool_name}")
            return f"Error parsing or executing tool arguments: {e}"

    async def prepare_turn(self, config: RunnerConfig):
        """Forces MCP initialization and preloads requested tools before turn 1."""
        if not self.mcp_config_path:
            return

        # If the user specifically requested servers or tools, we must eagerly initialize
        if config.mcp_active_servers or config.mcp_preload_tools:
            num_tools = await self.initialize_mcp(allowed_servers=config.mcp_active_servers)

            if num_tools > 0:
                self._mcp_initialized = True
            
            # Preload the specific tools directly into the active set
            if config.mcp_preload_tools:
                registry = getattr(self, '_mcp_standby_registry', {})
                for tool_name in config.mcp_preload_tools:
                    if tool_name in registry:
                        adapter = registry[tool_name]
                        self.register_tool(adapter, load_mcp=True)
                        if 'all' in self.toolsets and tool_name not in self.toolsets['all']:
                            self.toolsets['all'].append(tool_name)
                    else:
                        logger.warning(f"MCP tool '{tool_name}' not found in standby registry during preload.")
