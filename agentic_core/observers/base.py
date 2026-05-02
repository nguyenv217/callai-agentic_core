"""
Observer base interface.
"""
from abc import ABC
from ..decisions import DecisionEvent, LastIterationAction, LastIterationDecision, ToolStartAction, ToolStartDecision

class AgentEventObserver(ABC):
    """Base class for observing agent events."""
    
    def on_turn_start(self) -> None: 
        pass

    def on_iteration_start(self, iteration: int, max_iterations: int) -> None: 
        pass

    def on_llm_progress(self, info: str) -> None: 
        pass

    def on_tool_call_session_start(self, reasoning_text: str, tool_calls: list, iteration: int, max_iterations: int):
        pass

    def on_tool_start(self, tool_name: str, tool_id: str, tool_arg: str | dict | None = None) -> DecisionEvent[ToolStartAction]: 
        return DecisionEvent(action=ToolStartDecision.CONTINUE())

    def on_tool_complete(self, tool_name: str, tool_id: str, success: bool, result: str) -> None: 
        pass

    def on_turn_complete(self, response: dict) -> None: 
        pass

    def on_error(self, error: str) -> None: 
        pass
    
    def on_final_iteration(self) -> DecisionEvent[LastIterationAction]: 
        return DecisionEvent(action=LastIterationDecision.CONTINUE())