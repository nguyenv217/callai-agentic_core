import asyncio
from .engine import AgentRunner

class SessionManager:
    """
    Manages cached AgentRunner instances to preserve memory and MCP connections.
    """
    def __init__(self):
        self._sessions: dict[str, AgentRunner] = {}
        self._lock = None

    async def get_runner(self, session_id: str, creator_func) -> AgentRunner:
        if self._lock is None:
            self._lock = asyncio.Lock()

        async with self._lock:
            if session_id in self._sessions:
                return self._sessions[session_id]

            runner = await creator_func()
            self._sessions[session_id] = runner
            return runner

    async def remove_session(self, session_id: str):
        if self._lock is None:
            self._lock = asyncio.Lock()

        async with self._lock:
            if session_id in self._sessions:
                runner = self._sessions.pop(session_id)
                # ToolManager is handled by AgentRunner usually, but we should try to close it.
                if hasattr(runner, "tools") and hasattr(runner.tools, "shutdown_mcp"):
                    await runner.tools.shutdown_mcp()

# Global singleton for the convenience chat() function
global_session_manager = SessionManager()

