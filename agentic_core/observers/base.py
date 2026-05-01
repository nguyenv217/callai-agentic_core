"""
Observer base interface.
"""
from abc import ABC
from enum import Enum, auto
from ..interfaces import DecisionEvent

class ToolStartDecision(Enum):
    """
    Decision before each tool execution.
    
    Options:
        CONTINUE: proceed with execution
        SKIP: skip this tool only
        ABANDON: halt all execution and return final response immediately 
        SKIP_WITH_MSG: skip this tool, BUT leave a message for the agent as the tool result
        BREAK_WITH_MSG: skip every tool in this turn, BUT leave a message for the agent as the tool result
    """
    CONTINUE = auto(),   
    SKIP = auto()      
    ABANDON = auto()   
    SKIP_WITH_MSG = auto()  
    BREAK_WITH_MSG = auto() 

    @property
    def required_message(self):
        return self in [ToolStartDecision.SKIP_WITH_MSG, ToolStartDecision.BREAK_WITH_MSG]

class LastIterationDecision(Enum):
    """
    Decision after the last tool execution.
    
    Options:
        CONTINUE: proceed with the last iteration (agent may continue calling tools until iteration budget is depleted)
        LEAVE_MESSAGE: leave a final message for the agent and continue with the last iteration
        ABANDON: return immediately
        EXTEND: extends the max iteration budget by `max_iterations_count` (if not supplied/is None, defaults to current config's `max_iterations`) 
    """
    CONTINUE = auto()
    LEAVE_MESSAGE = auto()
    ABANDON = auto()
    EXTEND = auto()

    @property
    def required_message(self):
        return self == LastIterationDecision.LEAVE_MESSAGE
    
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

    def on_tool_start(self, tool_name: str, tool_id: str, tool_arg: str | dict | None = None) -> DecisionEvent[ToolStartDecision]: 
        return DecisionEvent(action=ToolStartDecision.CONTINUE)

    def on_tool_complete(self, tool_name: str, tool_id: str, success: bool, result: str) -> None: 
        pass

    def on_turn_complete(self, response: dict) -> None: 
        pass

    def on_error(self, error: str) -> None: 
        pass
    
    def on_final_iteration(self) -> DecisionEvent[LastIterationDecision]: 
        return DecisionEvent(action=LastIterationDecision.CONTINUE)