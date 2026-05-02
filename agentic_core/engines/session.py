# agentic_core/engines/session.py
import asyncio
import time
import logging
from .engine import AgentRunner

logger = logging.getLogger(__name__)

class SessionManager:
    """
    Manages cached AgentRunner instances to preserve memory and MCP connections.
    Includes an automatic TTL (Time-To-Live) cleanup mechanism to prevent memory leaks.
    """
    def __init__(self, ttl_seconds: int = 3600, cleanup_interval: int = 600):
        # Store dicts containing both the runner and its last accessed time
        self._sessions: dict[str, dict] = {} 
        self._lock = None
        self.ttl_seconds = ttl_seconds
        self.cleanup_interval = cleanup_interval
        self._cleanup_task: asyncio.Task | None = None

    def _get_key(self, tenant_id: str, session_id: str) -> str:
        return f"{tenant_id}::{session_id}"

    async def _ensure_cleanup_task(self):
        """Ensures the background cleanup task is running within the current event loop."""
        if self._cleanup_task is None or self._cleanup_task.done():
            loop = asyncio.get_running_loop()
            self._cleanup_task = loop.create_task(self._cleanup_loop())

    async def _cleanup_loop(self):
        """Background loop that periodically sweeps for stale sessions."""
        while True:
            await asyncio.sleep(self.cleanup_interval)
            await self.cleanup_stale_sessions()

    async def cleanup_stale_sessions(self):
        """Identifies and removes sessions that have exceeded the TTL."""
        if self._lock is None:
            self._lock = asyncio.Lock()
            
        now = time.time()
        stale_keys = []
        
        async with self._lock:
            for key, data in self._sessions.items():
                if now - data["last_accessed"] > self.ttl_seconds:
                    stale_keys.append(key)
        
        for key in stale_keys:
            logger.info(f"SessionManager: TTL expired, cleaning up session {key}")
            
            # Re-acquire lock briefly to pop the session safely
            async with self._lock:
                if key in self._sessions:
                    data = self._sessions.pop(key)
                else:
                    continue
            
            # Shut down MCP servers outside the lock to avoid blocking the whole cache
            runner = data["runner"]
            if hasattr(runner, "tools") and hasattr(runner.tools, "shutdown_mcp"):
                try:
                    # Apply a timeout so a hanging MCP server doesn't stall the cleanup loop
                    await asyncio.wait_for(runner.tools.shutdown_mcp(), timeout=10.0)
                except Exception as e:
                    logger.error(f"Error shutting down MCP for stale session {key}: {e}")

    async def get_runner(self, session_id: str, creator_func, tenant_id: str = "default") -> AgentRunner:
        if self._lock is None:
            self._lock = asyncio.Lock()
            
        await self._ensure_cleanup_task()
        cache_key = self._get_key(tenant_id, session_id)
        
        async with self._lock:
            if cache_key in self._sessions:
                # Update the timestamp on access
                self._sessions[cache_key]["last_accessed"] = time.time()
                return self._sessions[cache_key]["runner"]
            
            runner = await creator_func()
            self._sessions[cache_key] = {
                "runner": runner,
                "last_accessed": time.time()
            }
            return runner

    async def remove_session(self, session_id: str, tenant_id: str = "default"):
        """Manually removes a session and shuts down its dependencies."""
        if self._lock is None:
            self._lock = asyncio.Lock()
            
        cache_key = self._get_key(tenant_id, session_id)
        
        async with self._lock:
            if cache_key in self._sessions:
                data = self._sessions.pop(cache_key)
                runner = data["runner"]
                if hasattr(runner, "tools") and hasattr(runner.tools, "shutdown_mcp"):
                    await runner.tools.shutdown_mcp()

# Global singleton for the convenience chat() function
# Defaults: 1 hour (3600s) TTL, sweeping every 10 minutes (600s)
global_session_manager = SessionManager(ttl_seconds=3600, cleanup_interval=600)