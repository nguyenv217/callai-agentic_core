import asyncio
import logging
from typing import Any
from agentic_core.handlers.base import AgentEventHandler
from agentic_core.decisions import ErrorContext, DecisionEvent, ToolStartAction
from agentic_core.interfaces import AgentResponse
from .extractor import SkillExtractor

logger = logging.getLogger(__name__)

class AutoSkillObserver(AgentEventHandler):
    """
    Passively monitors agent execution. If a turn encounters multiple errors but
    ultimately succeeds, it triggers the SkillExtractor asynchronously to condense
    the experience into a reusable skill.
    """
    def __init__(self, extractor: SkillExtractor, error_threshold: int = 2, base_handler: AgentEventHandler | None = None):
        self.extractor = extractor
        self.error_threshold = error_threshold
        self.base_handler = base_handler
        self._trace_log: list[str] = []
        self._error_count = 0

    async def on_turn_start(self) -> None:
        self._trace_log.append("[TURN START]")
        if self.base_handler: await self.base_handler.on_turn_start()

    async def on_iteration_start(self, iteration: int, max_iterations: int) -> None:
        if self.base_handler: await self.base_handler.on_iteration_start(iteration, max_iterations)

    async def on_llm_progress(self, info: str) -> None:
        if self.base_handler: await self.base_handler.on_llm_progress(info)

    async def on_tool_call_session_start(self, reasoning_text: str, tool_calls: list, iteration: int, max_iterations: int) -> None:
        if self.base_handler: await self.base_handler.on_tool_call_session_start(reasoning_text, tool_calls, iteration, max_iterations)

    async def on_final_iteration(self) -> Any:
        if self.base_handler: return await self.base_handler.on_final_iteration()
        return await super().on_final_iteration()

    async def on_tool_start(self, tool_name: str, tool_id: str, tool_arg: str | dict | None = None) -> DecisionEvent[ToolStartAction]:
        self._trace_log.append(f"[TOOL START] {tool_name} | Args: {tool_arg}")
        if self.base_handler:
            return await self.base_handler.on_tool_start(tool_name, tool_id, tool_arg)
        return await super().on_tool_start(tool_name, tool_id, tool_arg)

    async def on_tool_complete(self, tool_name: str, tool_id: str, success: bool, result: str) -> None:
        status = "SUCCESS" if success else "FAILURE"
        self._trace_log.append(f"[TOOL COMPLETE] {tool_name} | Status: {status} | Result: {result}")
        if not success:
            self._error_count += 1
        if self.base_handler: await self.base_handler.on_tool_complete(tool_name, tool_id, success, result)

    async def on_error(self, error_context: ErrorContext) -> Any:
        self._error_count += 1
        self._trace_log.append(f"[ERROR] {error_context.error.__class__.__name__}: {error_context.error}")
        if self.base_handler:
            return await self.base_handler.on_error(error_context)
        return await super().on_error(error_context)

    async def on_turn_complete(self, response: AgentResponse) -> None:
        self._trace_log.append(f"[FINAL RESPONSE] {response.text}")
        
        # If the agent struggled (hit the error threshold) but managed to complete without a fatal iteration error
        if self._error_count >= self.error_threshold and not response.error:
            logger.info(f"Agent recovered from {self._error_count} errors. Triggering autonomous skill extraction.")
            trace_str = "\n".join(self._trace_log)
            asyncio.create_task(self.extractor.extract_skill(trace_str))
            
        self._trace_log.clear()
        self._error_count = 0
        if self.base_handler: await self.base_handler.on_turn_complete(response)
