import logging

from .base import AgentEventHandler
from ..decisions import (
    ToolStartDecision,
    DecisionEvent,
    ErrorContext,
    ErrorAction,
    ErrorDecision,
)

logger = logging.getLogger(__name__)


class SilentHandler(AgentEventHandler):
    """A no-op handler that prints nothing."""

    async def on_turn_start(self) -> None:
        pass

    async def on_iteration_start(self, iteration: int, max_iterations: int) -> None:
        pass

    async def on_llm_progress(self, info: str) -> None:
        pass

    async def on_tool_start(self, tool_name, tool_id, tool_arg):
        return DecisionEvent(ToolStartDecision.CONTINUE())

    async def on_tool_complete(self, tool_name: str, tool_id: str, success: bool, result: str) -> None:
        pass

    async def on_turn_complete(self, response: dict) -> None:
        pass

    async def on_error(self, error_context: ErrorContext) -> DecisionEvent[ErrorAction]:
        return DecisionEvent(action=ErrorDecision.ABANDON())

