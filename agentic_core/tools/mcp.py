
"""
MCP (Model Context Protocol) Tool Adapter.

This module provides dynamic tool registration from external MCP servers,
bridging the gap between MCP's JSON-RPC protocol and callai's BaseTool interface.
"""
from __future__ import annotations
import asyncio
import json
from typing import Any, Callable, Dict, List, TYPE_CHECKING, TypedDict
from pathlib import Path

from ..config import ConfigurationError
from .base import BaseTool

if TYPE_CHECKING:
    from .manager import ToolManager

import logging
logger = logging.getLogger(__name__)

# Note: ListMCPTools and LoadMCPTool are NOT decorated with @tool
# because they require runtime arguments (catalog_summary, tool_manager).
# They are manually instantiated in ToolManager.initialize_mcp() instead.

class ListMCPTools(BaseTool):
    """Returns a lightweight catalog of available external MCP tools."""
    
    def __init__(self, tool_manager: ToolManager):
        self._tool_manager = tool_manager
    
    @property
    def name(self) -> str:
        return "list_mcp_catalog"
    
    @property
    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "list_mcp_catalog",
                "description": "Returns a catalog of available MCP tools grouped by server. Use this to discover external tools. Pass a specific server_name to see its full list.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "server_name": {
                            "type": "string",
                            "description": "The specific MCP server domain (e.g., 'github', 'playwright') to expand and list all of its tools. Don't supply this argument to list all servers."
                        }
                    },
                    "required": []
                }
            }
        }
    
    def execute(self, args: dict, context: dict) -> str:
        server_filter = args.get("server_name")
        registry = getattr(self._tool_manager, '_mcp_standby_registry', {})
        
        if not registry:
            return "No MCP tools available in standby registry."
            
        # Group tools by their source server
        servers = {}
        for tool_name, adapter in registry.items():
            server = getattr(adapter, '_server_name', 'unknown')
            if server not in servers:
                servers[server] = []
            servers[server].append(adapter)
            
        lines = []
        
        if server_filter:
            # The agent requested a specific server's full catalog
            if server_filter not in servers:
                return f"Error: Server '{server_filter}' not found. Available servers: {list(servers.keys())}"
            
            lines.append(f"--- Available tools for server '{server_filter}' ({len(servers[server_filter])} total) ---")
            for adapter in servers[server_filter]:
                desc = adapter.schema['function'].get('description', 'No description')
                lines.append(f"- {adapter.name}: {desc}")
        else:
            # Default view: Overview with truncation
            lines.append("--- Available MCP Servers and Tools Overview ---")
            for server, adapters in servers.items():
                lines.append(f"\n[{server}] ({len(adapters)} tools total):")
                
                # Show up to 3 tools as a preview
                preview_limit = 3
                for adapter in adapters[:preview_limit]:
                    # Grab description and truncate it to a single line for neatness
                    raw_desc = adapter.schema['function'].get('description', 'No description')
                    short_desc = raw_desc.split('\n')[0][:100]
                    if len(raw_desc) > 100: 
                        short_desc += "..."
                    lines.append(f"  - {adapter.name}: {short_desc}")
                
                if len(adapters) > preview_limit:
                    lines.append(f"  ... and {len(adapters) - preview_limit} more. -> Call list_mcp_catalog with server_name='{server}' to view all.")
                    
        lines.append("\nTo load tools into active context, call 'load_mcp_tool' providing a list of exact tool names in the 'tool_names' array.")
        return "\n".join(lines)


class LoadMCPTool(BaseTool):
    """Moves specified tool schemas from standby registry into active execution context."""
    
    def __init__(self, tool_manager: ToolManager):
        self._tool_manager = tool_manager
    
    @property
    def name(self) -> str:
        return "load_mcp_tool"
    
    @property
    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "load_mcp_tool",
                "description": "Loads one or more MCP tools from the catalog into the active toolset, making them available for use.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tool_names": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "A list of exact names of the tools to load (from the catalog)."
                        }
                    },
                    "required": ["tool_names"]
                }
            }
        }
    
    def execute(self, args: dict, context: dict) -> str:
        tool_names = args.get("tool_names")
        
        # Fallback in case the LLM passes a single string or uses the old argument name
        if not tool_names and "tool_name" in args:
            tool_names = [args["tool_name"]]
        elif isinstance(tool_names, str):
            tool_names = [tool_names]
            
        if not tool_names or not isinstance(tool_names, list):
            return "Error: 'tool_names' (array of strings) is required."
        
        registry = getattr(self._tool_manager, '_mcp_standby_registry', {})
        
        results = []
        loaded = []
        
        for tool_name in tool_names:
            if tool_name not in registry:
                results.append(f"Error: Tool '{tool_name}' not found in catalog.")
                continue
            
            adapter = registry[tool_name]
            self._tool_manager.register_tool(adapter, load_mcp=True)
            
            loaded.append(tool_name)
            
        if loaded:
            results.append(f"Success: Loaded {len(loaded)} tool(s): {', '.join(loaded)}")
            
        return "\n".join(results)


class MCPToolAdapter(BaseTool):
    """
    A dynamic subclass of BaseTool that acts as a proxy to MCP server tools.
    Its execute method forwards calls via JSON-RPC to the respective MCP server.
    """
    
    def __init__(
        self, 
        mcp_tool_def: Dict[str, Any], 
        session: Any, 
        server_name: str,
        timeout: float = 30.0,
        clean_schema: bool = True
    ):
        """
        Initialize the MCP tool adapter.
        
        Args:
            mcp_tool_def: The tool definition from MCP server (contains name, description, inputSchema)
            session: The MCP ClientSession connected to the server
            server_name: Name identifier for the MCP server (used for namespacing)
            timeout: Timeout in seconds for tool execution
        """
        super().__init__()
        self._session = session
        self._server_name = server_name
        self._timeout = timeout
        
        # Prefix the tool name to prevent collisions across servers
        # e.g., "sqlite_query" instead of just "query"
        self._name = f"{server_name}_{mcp_tool_def.get('name', 'unnamed')}"
        
        if clean_schema:
            cleansed_schema = {k: v for k, v in mcp_tool_def.get('inputSchema', {}).items() if k in ['type', 'required', 'properties']}
        else: cleansed_schema=mcp_tool_def.get('inputSchema', {})

        # Map MCP schema to OpenAI function calling schema format
        self._schema = {
            "type": "function",
            "function": {
                "name": self._name,
                "description": mcp_tool_def.get('description', ''),
                "parameters": cleansed_schema
            }
        }
    
    @property
    def name(self) -> str:
        """The function name expected by the LLM."""
        return self._name
    
    @property
    def schema(self) -> dict:
        """The JSON schema associated with this tool."""
        return self._schema
    
    async def execute(self, args: dict, context: dict) -> str:
        """
        Execute the tool by calling the MCP server.
        
        Args:
            args: The arguments to pass to the tool
            context: The execution context (controller, callbacks, etc.)
            
        Returns:
            str: The result from the MCP server formatted as a string
        """
        import anyio
        # Extract the actual tool name (without server prefix) for the MCP call
        actual_tool_name = self._name.replace(f"{self._server_name}_", "", 1)
        
        try:
            # Apply timeout to prevent indefinite hangs
            logger.info(f"Executing tool {actual_tool_name} in mcp {self._server_name}")
            result = await asyncio.wait_for(
                self._session.call_tool(actual_tool_name, arguments=args),
                timeout=self._timeout
            )
            
            # Format MCP ToolResult array into string
            # MCP returns content as a list of Content objects (text, image, etc.)
            formatted_results = []
            for content in result.content:
                if hasattr(content, 'text'):
                    formatted_results.append(content.text)
                elif isinstance(content, dict):
                    formatted_results.append(json.dumps(content))
                else:
                    formatted_results.append(str(content))
            
            return "\n".join(formatted_results)
            
        except anyio.ClosedResourceError:
            logger.error(f"MCP server {self._server_name} disconnected unexpectedly.")
            return f"Error: The external server process for '{self._server_name}' crashed or disconnected. Please check the server logs and restart the tool."
        except asyncio.TimeoutError:
            return f"Error: Timeout after {self._timeout}s"
        except Exception as e:
            logger.exception(f"Exception at MCP tool {self._name}")
            return f"Error: {str(e)}"

class _MCPSessionProxy:
    def __init__(self, request_queue: asyncio.Queue, loop: asyncio.AbstractEventLoop, server_name: str):
        self.request_queue = request_queue
        self.loop = loop  # The loop where the background task lives
        self.server_name = server_name

    async def _send_to_actor(self, action, payload):
        """Safely injects a command into the actor's loop from ANY thread/loop."""
        import asyncio
        import concurrent.futures
        
        # 1. Create a thread-safe future instead of an asyncio.Future
        thread_safe_fut = concurrent.futures.Future()
        
        def _inject():
            # This runs INSIDE the actor's loop
            self.request_queue.put_nowait((action, payload, thread_safe_fut))
            
        # 2. Inject the task into the background loop
        self.loop.call_soon_threadsafe(_inject)
        
        # 3. Safely await the cross-loop future in the caller's loop
        return await asyncio.wrap_future(thread_safe_fut)

    async def list_tools(self):
        logger.info(f"[{self.server_name}] PROXY: Sending 'list_tools' to actor loop...")
        return await self._send_to_actor("list_tools", None)

    async def call_tool(self, name: str, arguments: dict = None):
        logger.info(f"[{self.server_name}] PROXY: Sending 'call_tool' ({name}) to actor loop...")
        return await self._send_to_actor("call_tool", {"name": name, "arguments": arguments or {}})

class _MCPSession(TypedDict):
    name: str
    session: _MCPSessionProxy 
    shutdown_event: asyncio.Event
    task: asyncio.Task
    # alive: bool # to be refactored for threaded keep-alive -> no need for this field anymoree
    # last_health_checked: int

class MCPClientManager:
    """
    Manages lifecycle of connections to external MCP servers.
    Uses the official Python MCP SDK for stdio-based connections.
    """
    
    def __init__(self, config_path: str | Path | None = None, config: Dict[str, Any] | None = None, on_server_death: Callable[[str, Exception], Any] | None = None):
        """
        Initialize the MCP client manager.
        
        Args:
            config_path: Path to the MCP servers configuration file (Optional)
            config: Direct MCP configuration dictionary (Optional)
            on_server_death: Hook on event of server death (Optional)
        """
        self.config_path = config_path
        self.config = config
        self.sessions: List[_MCPSession] = []
        self.on_server_death = on_server_death

    def load_config(self) -> Dict[str, Any]:
        """
        Load MCP server configuration from JSON file if path is provided.
        
        Returns:
            Dict containing mcpServers configuration
        """
        if self.config:
            return self.config

        if not self.config_path:
            logger.info("No MCP config path or dictionary provided.")
            return {"mcpServers": {}}

        if isinstance(self.config_path, str):
            config_file = Path(self.config_path)
        logger.info(f"Resolved path {config_file.resolve()}")

        if not config_file.exists():
            logger.error("Could not find MCP server configuration file.")
            raise ConfigurationError("MCP server configuration file not found")
        
        logger.info(f" Loading config from: {config_file}")
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        
        return self.config
    
    def get_active_servers(self):
        return [s['name'] for s in self.sessions]
    
    async def initialize(self, allowed_servers: list[str] | None = None, extra_env: dict[str, str] | None = None) -> bool:
        """
        Initialize connections to configured MCP servers.
        
        Args:
            allowed_servers: If provided, only these servers will be initialized.
            
        Returns:
            bool: True if at least one connection was established
        """
        config = self.load_config()
        
        if not config.get("mcpServers"): 
            raise ConfigurationError("No servers configured. Check your config file if you supplied `mcp_config_path`.")
            
        # Filter servers if a whitelist is provided
        servers_to_start = config['mcpServers']
        if allowed_servers is not None:
            servers_to_start = {
                k: v for k, v in servers_to_start.items() 
                if k in allowed_servers
            }

        # Filter out started and alive servers
        active_servers_name = [s['name'] for s in self.sessions]
        servers_to_start = {
            k: v for k, v in servers_to_start.items() 
            if not k in active_servers_name
        }
        
        if not servers_to_start:
            logger.info("No allowed servers matched configured servers.")
            return False

        logger.info(f" Found {len(servers_to_start)} server(s) to connect: {list(servers_to_start.keys())}")
        
        try:
            server_names = list(servers_to_start.keys())
            results = await asyncio.gather(
                *(self._connect_to_server(name, cfg, extra_env) for name, cfg in servers_to_start.items()),
                return_exceptions=True
            )

            for name, result in zip(server_names, results):
                if isinstance(result, Exception):
                    logger.error(f"Skipping server '{name}' due to error: {result}")

            return len(self.sessions) > 0
            
        except ImportError:
            raise ConfigurationError("MCP SDK not installed. Run: pip install mcp")
    
    async def _connect_to_server(self, server_name: str, server_config: Dict[str, Any], extra_env: dict[str, str] | None):
        from mcp.client.stdio import stdio_client
        from mcp import ClientSession, StdioServerParameters
        import re, os, shutil, asyncio
        
        raw_command = server_config.get("command", "python")
        args = server_config.get("args", [])
        
        command = shutil.which(raw_command)
        if not command:
            logger.error(f"ERROR: Could not find '{raw_command}' in system PATH.")
            return False

        # Provide only what is strictly necessary to run subprocesses
        safe_env_keys = ["PATH", "HOME", "USERPROFILE", "SystemRoot", "APPDATA", "LOCALAPPDATA"]
        extra_env = extra_env or {}
        env = {**{k: os.environ[k] for k in safe_env_keys if k in os.environ}, **extra_env}

        server_env = server_config.get("env", {})

        for key, value in server_env.items():
            if isinstance(value, str) and value.strip().startswith("${") and value.endswith("}"):
                match = re.match(r"\${(.*?)\}", value)
                if match:
                    env_var = match.group(1)
                    if env_var in os.environ:
                        env[key] = os.environ[env_var]
                    else:
                        print(f"Config error: Environment variable '{env_var}' not found")
                        return False
            else:
                env[key] = value
        
        logger.info(f"Starting server '{server_name}'...")
        
        shutdown_event = asyncio.Event()
        init_event = asyncio.Event()
        session_ref = {"error": None}
        
        # Create the bridge and proxy (UPDATE THIS LINE)
        current_loop = asyncio.get_running_loop()
        request_queue = asyncio.Queue()
        
        # Pass the loop to the proxy so it knows where to send signals
        session_proxy = _MCPSessionProxy(request_queue, current_loop, server_name)

        async def server_task():
            try:
                server_params = StdioServerParameters(command=command, args=args, env=env)
                async with stdio_client(server_params) as (read_stream, write_stream):
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        init_event.set()
                        
                        logger.info(f"[{server_name}] ACTOR: Background task ready and polling queue.")
                        
                        while not shutdown_event.is_set():
                            try:
                                action, payload, fut = await asyncio.wait_for(request_queue.get(), timeout=0.5)
                                try:
                                    logger.info(f"[{server_name}] ACTOR: Pulled '{action}' from queue.")
                                    if action == "list_tools":
                                        res = await session.list_tools()
                                        if not fut.done(): fut.set_result(res)
                                    elif action == "call_tool":
                                        tool_name = payload["name"]
                                        args_dict = payload.get("arguments", {})
                                        logger.info(f"[{server_name}] ACTOR: Executing '{tool_name}' via MCP...")
                                        
                                        # The actual call to the Node server
                                        res = await session.call_tool(tool_name, arguments=args_dict)
                                        
                                        logger.info(f"[{server_name}] ACTOR: Successfully executed '{tool_name}'!")
                                        if not fut.done(): fut.set_result(res)
                                        
                                except Exception as e:
                                    logger.error(f"[{server_name}] ACTOR ERROR: {type(e).__name__} - {e}")
                                    if not fut.done(): fut.set_exception(e)
                                finally:
                                    logger.info(f"[{server_name}] ACTOR: Marking queue task as done.")
                                    request_queue.task_done()
                                    
                            except asyncio.TimeoutError:
                                pass # Normal heartbeat, loop around
                            except Exception as queue_err:
                                logger.error(f"[{server_name}] ACTOR QUEUE ERROR: {queue_err}")
                                
            except Exception as e:
                logger.exception(f"[{server_name}] Server task died unexpectedly: {e}")

                if self.on_server_death:
                    self.on_server_death(server_name, e)
                
                session_ref["error"] = e
                if not init_event.is_set():
                    init_event.set()

        task = asyncio.create_task(server_task(), name=f"mcp_server_{server_name}")
        
        await asyncio.wait_for(init_event.wait(), timeout=60.0)
        
        if session_ref["error"]:
            raise session_ref["error"]

        self.sessions.append({
            "name": server_name,
            "session": session_proxy,  # Inject the Proxy instead of the raw session!
            "shutdown_event": shutdown_event,
            "task": task,
            # "alive": True
        })
        logger.info(f"Session mapped for '{server_name}'")
    
    async def list_all_tools(self) -> List[Dict[str, Any]]:
        """
        List all tools from all connected MCP servers.

        Returns:
            List of tool definitions with server info
        """

        all_tools = []

        sessions = [(session["name"], session["session"]) for session in self.sessions]
        
        results = await asyncio.gather(
            *(self._fetch_tools_info(session) for session in sessions),
            return_exceptions=True
        )

        for session_name, result in zip(sessions, results):
            if isinstance(result, Exception):
                logger.exception(f"Error listing tools from server '{session_name[0]}': {result}")
            else:
                all_tools.extend(result)

        return all_tools

    async def _fetch_tools_info(self, session_info: tuple[str, _MCPSessionProxy]):
        server_name, session = session_info
        tools_result = await session.list_tools()

        if hasattr(tools_result, 'tools'):
            tools = tools_result.tools
        else:
            tools = tools_result

        return [
            {
                "server_name": server_name,
                "session": session,
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.inputSchema
            }
            for tool in tools
        ]
    
    async def close(self):
        """Close all MCP server connections gracefully."""
        import asyncio
        
        for session_info in self.sessions:
            server_name = session_info.get("name", "unknown")
            try:
                # 1. Trigger the event to break the wait loop inside the task
                shutdown_event = session_info.get("shutdown_event")
                if shutdown_event:
                    shutdown_event.set()
                
                # 2. Wait for the task to exit its async with blocks cleanly
                task = session_info.get("task")
                if task:
                    await asyncio.wait_for(task, timeout=5.0)
                    
                logger.info(f"Closed connection to '{server_name}'")
            except asyncio.TimeoutError:
                logger.warning(f"Timeout while closing '{server_name}'. Force killing.")
            except Exception:
                logger.exception(f"Error closing MCP session for '{server_name}'")
        
        self.sessions.clear()
