import asyncio
from .engine import AgentRunner

class SessionManager:
    """
    Manages cached AgentRunner instances to preserve memory and MCP connections.
    """
    def __init__(self):
        self._sessions: dict[str, AgentRunner] = {}
        self._lock = None

    def _get_key(self, tenant_id: str, session_id: str) -> str:
        return f"{tenant_id}::{session_id}"

    async def get_runner(self, session_id: str, creator_func, tenant_id: str = "default") -> AgentRunner:
        if self._lock is None:
            self._lock = asyncio.Lock()

        cache_key = self._get_key(tenant_id, session_id)

        async with self._lock:
            if session_id in self._sessions:
                return self._sessions[cache_key]

            runner = await creator_func()
            self._sessions[cache_key] = runner
            return runner

    async def remove_session(self, session_id: str, tenant_id: str = "default"):
        if self._lock is None:
            self._lock = asyncio.Lock()
            
        cache_key = self._get_key(tenant_id, session_id)
        
        async with self._lock:
            if cache_key in self._sessions:
                runner = self._sessions.pop(cache_key)
                if hasattr(runner, "tools") and hasattr(runner.tools, "shutdown_mcp"):
                    await runner.tools.shutdown_mcp()

# Global singleton for the convenience chat() function
global_session_manager = SessionManager()

