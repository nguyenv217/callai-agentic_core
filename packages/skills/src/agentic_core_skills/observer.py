import logging
from typing import Any
from agentic_core.handlers.base import AgentEventHandler
from agentic_core.decisions import ErrorContext, DecisionEvent, ToolStartAction
from agentic_core.interfaces import AgentResponse
from .extractor import SkillExtractor

logger = logging.getLogger(__name__)

class AutoSkillObserver(AgentEventHandler):
    """
    Passively monitors agent execution over a session.
    Accumulates errors and tracks excessive tool usage/inefficiencies.
    """
    def __init__(self, extractor: SkillExtractor, error_threshold: int = 3, tool_call_threshold: int = 6, base_handler: AgentEventHandler | None = None):
        self.extractor = extractor
        self.error_threshold = error_threshold
        self.tool_call_threshold = tool_call_threshold
        self.base_handler = base_handler
        self._session_error_count = 0
        self._session_tool_call_count = 0
        self._task_completed = False

    async def on_turn_start(self) -> None:
        if self.base_handler: await self.base_handler.on_turn_start()

    async def on_iteration_start(self, iteration: int, max_iterations: int) -> None:
        if self.base_handler: await self.base_handler.on_iteration_start(iteration, max_iterations)

    async def on_llm_progress(self, info: str) -> None:
        if self.base_handler: await self.base_handler.on_llm_progress(info)

    async def on_tool_call_session_start(self, reasoning_text: str, tool_calls: list, iteration: int, max_iterations: int) -> None:
        if self.base_handler: await self.base_handler.on_tool_call_session_start(reasoning_text, tool_calls, iteration, max_iterations)

    async def on_tool_start(self, tool_name: str, tool_id: str, tool_arg: str | dict | None = None) -> DecisionEvent[ToolStartAction]:
        self._session_tool_call_count += 1
        if self.base_handler:
            return await self.base_handler.on_tool_start(tool_name, tool_id, tool_arg)
        return await super().on_tool_start(tool_name, tool_id, tool_arg)

    async def on_tool_complete(self, tool_name: str, tool_id: str, success: bool, result: str) -> None:
        if not success:
            self._session_error_count += 1
        if tool_name == "announce_finish" and success:
            self._task_completed = True
        if self.base_handler: await self.base_handler.on_tool_complete(tool_name, tool_id, success, result)

    async def on_error(self, error_context: ErrorContext) -> Any:
        self._session_error_count += 1
        if self.base_handler:
            return await self.base_handler.on_error(error_context)
        return await super().on_error(error_context)

    async def on_final_iteration(self) -> Any:
        if self.base_handler: return await self.base_handler.on_final_iteration()
        return await super().on_final_iteration()

    async def on_turn_complete(self, response: AgentResponse) -> None:
        if self.base_handler: await self.base_handler.on_turn_complete(response)