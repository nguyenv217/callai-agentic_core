"""
Handler base interface.
"""
from abc import ABC

from agentic_core.interfaces import AgentResponse
from ..decisions import (
    DecisionEvent, 
    LastIterationAction, 
    LastIterationDecision, 
    ToolStartAction, 
    ToolStartDecision,
    ErrorContext,
    ErrorAction,
    ErrorDecision
)

class AgentEventHandler(ABC):
    """Base class for observing agent events."""
    
    async def on_turn_start(self) -> None: 
        pass

    async def on_iteration_start(self, iteration: int, max_iterations: int) -> None: 
        pass

    async def on_llm_progress(self, info: str) -> None: 
        pass

    async def on_tool_call_session_start(self, reasoning_text: str, tool_calls: list, iteration: int, max_iterations: int):
        pass

    async def on_tool_start(self, tool_name: str, tool_id: str, tool_arg: str | dict | None = None) -> DecisionEvent[ToolStartAction]: 
        return DecisionEvent(action=ToolStartDecision.CONTINUE())

    async def on_tool_complete(self, tool_name: str, tool_id: str, success: bool, result: str) -> None: 
        pass

    async def on_turn_complete(self, response: AgentResponse) -> None: 
        pass

    async def on_error(self, error_context: ErrorContext) -> DecisionEvent[ErrorAction]:
        """
        Handle errors with granular decision making.
        
        Args:
            error_context: Context information about the error including type, exception, 
                          tool info, iteration details, and retry state.
        
        Returns:
            DecisionEvent with an ErrorDecision action to control flow.
            Default implementation returns ABANDON for all errors.
        """
        return DecisionEvent(action=ErrorDecision.ABANDON())
    
    async def on_final_iteration(self) -> DecisionEvent[LastIterationAction]: 
        return DecisionEvent(action=LastIterationDecision.CONTINUE())
