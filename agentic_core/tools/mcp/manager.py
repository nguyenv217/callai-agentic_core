import asyncio
import json
import logging
from typing import Any, Callable, Dict, List, Tuple, TypedDict
from pathlib import Path

from ...config import ConfigurationError

try:
    from mcp.client.stdio import stdio_client
    from mcp import ClientSession, StdioServerParameters
except ImportError:
    raise ConfigurationError("Python `mcp` package is not installed. Please install with: `pip install mcp`")


logger = logging.getLogger(__name__)

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
    ref_count: int

class GlobalMCPRegistry:
    """
    A singleton registry that manages a pool of active MCP sessions.
    Uses reference counting to ensure that multiple managers can share the same
    underlying MCP server process, maximizing resource efficiency.
    """
    _instance: 'GlobalMCPRegistry' = None
    _lock = asyncio.Lock()
    _sessions: Dict[Tuple, _MCPSession] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GlobalMCPRegistry, cls).__new__(cls)
        return cls._instance

    @staticmethod
    def _get_identity_key(server_config: Dict[str, Any]) -> Tuple:
        """Generates a hashable identity key from the server configuration."""
        command = server_config.get("command", "python")
        args = tuple(server_config.get("args", []))
        # Sort environment items to ensure consistent hashing
        env = tuple(sorted(server_config.get("env", {}).items()))
        return (command, args, env)

    async def acquire(
        self,
        server_name: str,
        server_config: Dict[str, Any],
        extra_env: Dict[str, str] | None,
        on_server_death: Callable[[str, Exception], Any] | None
    ) -> _MCPSession:
        """
        Acquires a session for the given configuration.
        Returns an existing session if available, or creates a new one.
        """
        identity_key = self._get_identity_key(server_config)

        async with self._lock:
            if identity_key in self._sessions:
                logger.info(f"[Registry] Reusing existing session for '{server_name}' (Identity: {identity_key})")
                self._sessions[identity_key]['ref_count'] += 1
                return self._sessions[identity_key]

            logger.info(f"[Registry] Creating new session for '{server_name}' (Identity: {identity_key})")
        import re, os, shutil, asyncio
        
        raw_command = server_config.get("command", "python")
        args = server_config.get("args", [])
        command = shutil.which(raw_command)
        if not command:
                raise RuntimeError(f"Could not find '{raw_command}' in system PATH.")
        safe_env_keys = ["PATH", "HOME", "USERPROFILE", "SystemRoot", "APPDATA", "LOCALAPPDATA"]
        env = {**{k: os.environ[k] for k in safe_env_keys if k in os.environ}, **(extra_env or {})}
        server_env = server_config.get("env", {})
        for key, value in server_env.items():
            if isinstance(value, str) and value.strip().startswith("${") and value.endswith("}"):
                match = re.match(r"\${(.*?)\}", value)
                if match:
                    env_var = match.group(1)
                    if env_var in os.environ:
                        env[key] = os.environ[env_var]
                    else:
                            raise ConfigurationError(f"Environment variable '{env_var}' not found")
                else:
                    env[key] = value
        
        shutdown_event = asyncio.Event()
        init_event = asyncio.Event()
        session_ref = {"error": None}
        current_loop = asyncio.get_running_loop()
        request_queue = asyncio.Queue()
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

                if on_server_death:
                    on_server_death(server_name, e)
                
                session_ref["error"] = e
                if not init_event.is_set():
                    init_event.set()

        task = asyncio.create_task(server_task(), name=f"mcp_server_{server_name}")
        
        await asyncio.wait_for(init_event.wait(), timeout=60.0)
        
        if session_ref["error"]:
            raise session_ref["error"]

        new_session: _MCPSession = {
        "name": server_name,
            "session": session_proxy,
        "shutdown_event": shutdown_event,
        "task": task,
            "ref_count": 1
        }
        self._sessions[identity_key] = new_session
        return new_session

    async def release(self, identity_key: Tuple):
        """Decrements ref count and shuts down session if last client releases it."""
        async with self._lock:
            if identity_key in self._sessions:
                self._sessions[identity_key]['ref_count'] -= 1
                ref_count = self._sessions[identity_key]['ref_count']

                if ref_count <= 0:
                    logger.info(f"[Registry] Ref count zero. Shutting down session for {identity_key}")
                    session_info = self._sessions.pop(identity_key)
                    shutdown_event = session_info['shutdown_event']
                    task = session_info['task']
                    shutdown_event.set()
                    try:
                        await asyncio.wait_for(task, timeout=5.0)
                    except asyncio.TimeoutError:
                                logger.warning(f"[Registry] Shutdown timeout for {identity_key}. Force killing task.")
                                task.cancel()
                    except Exception:
                                pass
                else:
                    logger.info(f"[Registry] Session for {identity_key} still active. Ref count: {ref_count}")

    async def clear(self):
        """Clears all sessions in the registry."""
        async with self._lock:
            for identity_key in list(self._sessions.keys()):
                session_info = self._sessions.pop(identity_key)
                shutdown_event = session_info['shutdown_event']
                task = session_info['task']
                shutdown_event.set()
                try:
                    await asyncio.wait_for(task, timeout=1.0)
                except Exception:
                    task.cancel()
            logger.info("[Registry] All sessions cleared.")

class MCPClientManager:
    """
    Manages lifecycle of connections to external MCP servers.
    Uses a GlobalMCPRegistry to share sessions across multiple managers
    when server configurations are identical.
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
        self.on_server_death = on_server_death
        self.sessions: List[dict] = []
        self._registry = GlobalMCPRegistry()

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
        else:
            config_file = self.config_path

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
            raise ConfigurationError("MCP SDK not installed. Run: `pip install mcp`")

    async def _connect_to_server(self, server_name: str, server_config: Dict[str, Any], extra_env: dict[str, str] | None):
        identity_key = self._registry._get_identity_key(server_config)

        session_info = await self._registry.acquire(
            server_name,
            server_config,
            extra_env,
            self.on_server_death
        )
        self.sessions.append({
            **session_info,
            "identity_key": identity_key,
            "name": server_name
        })
        logger.info(f"Session mapped for '{server_name}' (Shared ref: {session_info['ref_count']})")

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
        """Close all MCP server connections gracefully, releasing shared sessions."""
        import asyncio
        
        for session_info in self.sessions:
            server_name = session_info.get("name", "unknown")
            identity_key = session_info.get("identity_key")
            try:
                if identity_key:
                    await self._registry.release(identity_key)
                logger.info(f"Released connection to '{server_name}'")
            except Exception:
                logger.exception(f"Error releasing MCP session for '{server_name}'")

        self.sessions.clear()

