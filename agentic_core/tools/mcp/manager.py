import asyncio
import json
import logging
from typing import Any, Callable, Tuple, TypedDict
from pathlib import Path

from ...config import ConfigurationError

try:
    from mcp.client.stdio import stdio_client
    from mcp import ClientSession, StdioServerParameters
except ImportError:
    raise ConfigurationError("Python `mcp` package is not installed. Please install with: `pip install mcp`")

logger = logging.getLogger(__name__)


# =================================
# Custom Process Management
# =================================
import os
import atexit

_MCP_CLEANUP_WITH_PSUTIL = os.getenv("MCP_CLEANUP_WITH_PSUTIL", False)

def kill_process_tree(pid: int, expected_create_time: float = None):
    """Gracefully terminates a process and all its descendants with PID-recycle protection."""
    import psutil
    import contextlib
    with contextlib.suppress(psutil.NoSuchProcess):
        parent = psutil.Process(pid)
        
        # If birth times don't match, this is a recycled PID
        if expected_create_time and parent.create_time() != expected_create_time:
            logger.debug(f"PID {pid} was recycled. Skipping tree termination.")
            return 
            
        children = parent.children(recursive=True)
        
        for child in children:
            with contextlib.suppress(psutil.NoSuchProcess):
                child.kill()
        
        parent.kill()

_ACTIVE_MCP_PIDS: set[tuple[int, float]] = set()

def _emergency_cleanup():
    """Fallback to when event loop is abruptly closed and context is never switched back to `server_tasks` to perform recursive cleanup"""
    for pid, birth_time in list(_ACTIVE_MCP_PIDS):
        logger.debug(f"Emergency cleanup for PID {pid}")
        kill_process_tree(pid, expected_create_time=birth_time)

atexit.register(_emergency_cleanup)


# =============================
# MCP implementations
# =============================

class MCPTimeoutError(Exception):
    """Exception raised when an MCP operation times out."""
    pass

class _MCPSessionProxy:
    def __init__(self, request_queue: asyncio.Queue, loop: asyncio.AbstractEventLoop, server_name: str):
        self.request_queue = request_queue
        self.loop = loop  # The loop where the background task lives
        self.server_name = server_name

    async def _send_to_actor(self, action, payload):
        """Safely injects a command into the actor's loop from ANY thread/loop."""
        import concurrent.futures
        thread_safe_fut = concurrent.futures.Future()
        def _inject():
            self.request_queue.put_nowait((action, payload, thread_safe_fut))
        self.loop.call_soon_threadsafe(_inject)
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
    _sessions: dict[Tuple, _MCPSession] = {}
    _locks: dict[Tuple, asyncio.Lock] = {}
    _global_lock = asyncio.Lock()
    _failed_sessions: set[Tuple] = set() # Track failed identity keys

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GlobalMCPRegistry, cls).__new__(cls)
        return cls._instance
    
    @staticmethod
    def _get_identity_key(server_config: dict[str, Any], tenant_id: str = "default") -> Tuple:
        """Identity key hash from (command, args, eng) and a tenant_id to prevent cross-tenant poisoning"""
        command = server_config.get("command", "python")
        raw_args = server_config.get("args", [])
        if isinstance(raw_args, str):
            raw_args = [raw_args]
        args = tuple(raw_args)
        env = tuple(sorted(server_config.get("env", {}).items()))
        return (tenant_id, command, args, env)

    async def _get_lock_for_identity(self, identity_key: Tuple) -> asyncio.Lock:
        async with self._global_lock:
            if identity_key not in self._locks:
                self._locks[identity_key] = asyncio.Lock()
            return self._locks[identity_key]

    async def acquire(
        self,
        server_name: str,
        server_config: dict[str, Any],
        extra_env: dict[str, str] | None,
        on_server_death: Callable[[str, Exception], Any] | None,
        tenant_id: str = "default"
    ) -> _MCPSession:
        """
        Acquires a session for the given configuration.
        Returns an existing session if available, or creates a new one.
        """
        identity_key = self._get_identity_key(server_config, tenant_id)

        if identity_key in self._failed_sessions:
            raise RuntimeError(f"Server '{server_name}' previously failed to initialize. Blocking retry.")
        
        lock = await self._get_lock_for_identity(identity_key)
        async with lock:
            if identity_key in self._sessions:
                logger.info(f"[Registry] Reusing existing session for '{server_name}' (Identity: {identity_key})")
                self._sessions[identity_key]['ref_count'] += 1
                return self._sessions[identity_key]

            logger.info(f"[Registry] Creating new session for '{server_name}' (Identity: {identity_key})")
            import re, os, shutil
            raw_command = server_config.get("command", "python")
            args = server_config.get("args", [])
            command = shutil.which(raw_command)
            if not command: 
                raise RuntimeError(f"Could not find {raw_command}")

            safe_env_keys = ["PATH", "HOME", "USERPROFILE", "SystemRoot", "APPDATA", "LOCALAPPDATA"]
            env = {k: os.environ[k] for k in safe_env_keys if k in os.environ}
            if extra_env:
                for k, v in extra_env.items():
                    # Validate keys are alphanumeric + underscore
                    if not k.replace("_", "").isalnum():
                        logger.warning(f"Dropping unsafe extra_env key: {k}")
                        continue
                    env[k] = str(v)
            
            # Inject environment variables in the command and args
            server_env = server_config.get("env", {})
            for key, value in server_env.items():
                if isinstance(value, str) and value.strip().startswith("${") and value.endswith("}"):
                    match = re.match(r"\${(.*?)\}", value)
                    if match:
                        env_var = match.group(1)
                        val = (extra_env or {}).get(env_var)
                        env[key] = val if val is not None else os.environ.get(env_var, value)
                    else:
                        env[key] = value
            
            shutdown_event = asyncio.Event()
            init_event = asyncio.Event()
            session_ref = {"error": None}
            current_loop = asyncio.get_running_loop()
            request_queue = asyncio.Queue()
            session_proxy = _MCPSessionProxy(request_queue, current_loop, server_name)

            async def server_task():
                err_stream = None
                try:
                    log_file = server_config.get("log_file")
                    if log_file:
                        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
                        err_stream = open(log_file, "a", encoding="utf-8")
                    else:
                        # Provide an explicit devnull handle if no log file is set
                        err_stream = open(os.devnull, "w")

                    server_params = StdioServerParameters(command=command, args=args, env=env)
                    
                    async with stdio_client(server_params, errlog=err_stream) as (read_stream, write_stream):
                        try:   
                            if _MCP_CLEANUP_WITH_PSUTIL: # hack: we snapshot before and after initalizing mcp client, then can grab the PID of the server subprocess
                                import psutil
                                current_process = psutil.Process(os.getpid())
                                children_before = set(current_process.children(recursive=False)) 

                            async with ClientSession(read_stream, write_stream) as session:
                                if _MCP_CLEANUP_WITH_PSUTIL:
                                    children_after = set(current_process.children(recursive=False))
                                    new_children = list(children_after - children_before)

                                    if new_children:
                                        wrapper_proc = new_children[0]
                                        session_ref["pid"] = wrapper_proc.pid
                                        session_ref["create_time"] = wrapper_proc.create_time() # record this time to prevents PID from being recycled to innocent tasks and the shutdown event being scheduled 
                                        _ACTIVE_MCP_PIDS.add((session_ref["pid"], session_ref["create_time"]))
                                        logger.debug(f"[{server_name}] Captured wrapper PID: {session_ref['pid']}")

                                await session.initialize()
                                init_event.set()

                                logger.info(f"[{server_name}] ACTOR: Background task ready and polling queue.")
                                while not shutdown_event.is_set():
                                    try:
                                        action, payload, fut = await asyncio.wait_for(request_queue.get(), timeout=0.5)
                                        try:
                                            if action == "list_tools":
                                                res = await session.list_tools()
                                                if not fut.done(): fut.set_result(res)
                                            elif action == "call_tool":
                                                res = await session.call_tool(payload["name"], arguments=payload.get("arguments"))
                                                logger.info(f"[{server_name}] ACTOR: Successfully executed '{payload['name']}'")
                                                if not fut.done(): fut.set_result(res)
                                        except Exception as e:
                                            logger.exception(f"[{server_name}] ACTOR ERROR: {type(e).__name__} - {e}")
                                            if not fut.done(): fut.set_exception(e)
                                        finally:
                                            request_queue.task_done()
                                    except asyncio.TimeoutError: 
                                        # Normal heartbeat
                                        pass
                        finally:
                            if _MCP_CLEANUP_WITH_PSUTIL:
                                pid_to_kill = session_ref.get("pid")
                                birth_time = session_ref.get("create_time")
                                if pid_to_kill:
                                    logger.debug(f"[{server_name}] Executing tree-killer on PID {pid_to_kill}")
                                    kill_process_tree(pid_to_kill, expected_create_time=birth_time)
                                    _ACTIVE_MCP_PIDS.discard((pid_to_kill, birth_time)) # cancellation happened, so no need for sync cleanup hook

                except Exception as e:
                    logger.exception(f"[{server_name}] Server task died unexpectedly: {e}")
                    if on_server_death: 
                        on_server_death(server_name, e)
                    session_ref["error"] = e
                    if not init_event.is_set():
                        init_event.set()

                finally:
                    if err_stream:
                        try:
                            err_stream.close()
                        except Exception:
                            pass

            task = asyncio.create_task(server_task(), name=f"mcp_server_{server_name}")
            try:
                await asyncio.wait_for(init_event.wait(), timeout=60.0)
            except asyncio.TimeoutError:
                task.cancel()
                self._failed_sessions.add(identity_key)
                raise MCPTimeoutError(f"Server '{server_name}' failed to initialize within 60 seconds (timeout).")

            if session_ref["error"]: 
                task.cancel() # ensure the task is killed if it died with an error
                self._failed_sessions.add(identity_key)
                raise session_ref["error"]

            new_session = {"name": server_name, "session": session_proxy, "shutdown_event": shutdown_event, "task": task, "ref_count": 1}
            self._sessions[identity_key] = new_session
        return new_session

    async def _shutdown_server(self, identity_key):
        session_info = self._sessions.pop(identity_key)
        session_info['shutdown_event'].set()
        task = session_info['task']
        try: 
            await asyncio.wait_for(task, timeout=2.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, RuntimeError):
                pass
        except Exception as e:
            logger.debug(f"[Registry] Unexpected error shutting down {identity_key}: {e}")
            task.cancel()

    async def release(self, identity_key: Tuple):
        """Decrements ref count and shuts down session if last client releases it."""
        lock = await self._get_lock_for_identity(identity_key)
        async with lock:
            if identity_key in self._sessions:
                self._sessions[identity_key]['ref_count'] -= 1
                if self._sessions[identity_key]['ref_count'] <= 0:
                    logger.info(f"[Registry] Ref count zero. Shutting down session for {identity_key}")
                    await self._shutdown_server(identity_key)
                else:
                    logger.info(f"[Registry] Session for {identity_key} still active. Ref count: {self._sessions[identity_key]['ref_count']}")
        
    async def clear(self):
        """Clears all sessions in the registry concurrently."""
        async with self._global_lock:
            identity_keys = list(self._sessions.keys())

        if not identity_keys:
            logger.warning("All sessions already cleared.")
            return

        async def _shutdown_session(identity_key: Tuple):
            lock = await self._get_lock_for_identity(identity_key)
            async with lock:
                if identity_key in self._sessions:
                    logger.info(f"[Registry] Clearing session for {identity_key}")
                    await self._shutdown_server(identity_key)

        # Dispatch all shutdowns concurrently
        await asyncio.gather(*[_shutdown_session(key) for key in identity_keys], return_exceptions=True)

class MCPClientManager:
    """
    Manages lifecycle of connections to external MCP servers.
    Uses a GlobalMCPRegistry to share sessions across multiple managers
    when server configurations are identical.
    """

    def __init__(self, config_path: str | Path | None = None, config: dict[str, Any] | None = None, on_server_death: Callable[[str, Exception], Any] | None = None):
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
        self.sessions: list[dict] = []
        self._registry = GlobalMCPRegistry()

    def load_config(self) -> dict[str, Any]:
        """
        Load MCP server configuration from JSON file if path is provided.

        Returns:
            dict containing mcpServers configuration
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

    async def initialize(self, allowed_servers: list[str] | None = None, extra_env: dict[str, str] | None = None, tenant_id: str = "default") -> bool:
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
                *(self._connect_to_server(name, cfg, extra_env, tenant_id) for name, cfg in servers_to_start.items()),
                return_exceptions=True
            )

            for name, result in zip(server_names, results):
                if isinstance(result, Exception):
                    logger.error(f"Skipping server '{name}' due to error: {result}")

            return len(self.sessions) > 0

        except ImportError:
            raise ConfigurationError("MCP SDK not installed. Run: `pip install mcp`")

    async def _connect_to_server(self, server_name: str, server_config: dict[str, Any], extra_env: dict[str, str] | None, tenant_id: str = "default"):
        identity_key = self._registry._get_identity_key(server_config, tenant_id)

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

    async def list_all_tools(self) -> list[dict[str, Any]]:
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
    
    async def disconnect(self, server_names: list[str] | None = None):
        """
        Disconnect from MCP server(s).
        
        Args:
            server_names: 
                List of server names to disconnect. If None, disconnects all servers.
                If provided, only disconnects the specified servers.
        """
        if not self.sessions:
            logger.info("No active MCP sessions to disconnect.")
            return

        if server_names is None:
            # Disconnect all servers (same as close())
            await self.close()
            return

        # Filter sessions to disconnect only specified servers
        sessions_to_disconnect = [
            s for s in self.sessions 
            if s.get("name") in server_names
        ]
        
        remaining_sessions = [
            s for s in self.sessions 
            if s.get("name") not in server_names
        ]

        for session_info in sessions_to_disconnect:
            server_name = session_info.get("name", "unknown")
            identity_key = session_info.get("identity_key")
            try:
                if identity_key:
                    await self._registry.release(identity_key)
                logger.info(f"Disconnected from '{server_name}'")
            except Exception:
                logger.exception(f"Error disconnecting MCP session for '{server_name}'")

        # Update sessions list to keep only remaining servers
        self.sessions = remaining_sessions

    async def close(self):
        """Close all MCP server connections gracefully, releasing shared sessions."""
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