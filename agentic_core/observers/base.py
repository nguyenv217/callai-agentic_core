"""
Observer base interface.
"""
from abc import ABC
from typing import Generic, TypeVar
from enum import Enum
from dataclasses import dataclass

class ToolStartDecision(Enum):
    """
    Decision before each tool execution.
    
    Options:
        CONTINUE: proceed with execution
        SKIP: skip this tool only
        ABANDON: skip every tool in this turn
        SKIP_WITH_MSG: skip this tool, BUT leave a message for the agent as the tool result
        ABANDON_WITH_MSG: skip every tool in this turn, BUT leave a message for the agent as the tool result
    """
    CONTINUE = 0   # proceed with execution
    SKIP = 1       # skip this tool only
    ABANDON = 2    # skip every tool in this turn
    SKIP_WITH_MSG = 3   # skip this tool, BUT leave a message for the agent as the tool result
    ABANDON_WITH_MSG = 4 # skip every tool in this turn and leave a message for the agent as the tool result

class LastIterationDecision(Enum):
    """
    Decision after the last tool execution.
    
    Options:
        CONTINUE: proceed with the last iteration (agent may continue calling tools until iteration budget is depleted)
        LEAVE_MESSAGE: leave a final message for the agent
        ABANDON: return immediately
    """
    CONTINUE = 0
    LEAVE_MESSAGE = 1
    ABANDON = 2

Action = TypeVar("Action", ToolStartDecision, LastIterationDecision)

@dataclass
class DecisionEvent(Generic[Action]):
    """Event for observing the decision made by an agent."""    
    action: Action
    message: str | None = None

    def __post_init__(self):
        if isinstance(self.action, ToolStartDecision) and (self.action in [ToolStartDecision.SKIP_WITH_MSG, ToolStartDecision.ABANDON_WITH_MSG] and self.message is None):
            raise ValueError("ToolStartDecision.SKIP_WITH_MSG and ToolStartDecision.ABANDON_WITH_MSG cannot be used without a message")
        if isinstance(self.action, LastIterationDecision) and (self.action in [LastIterationDecision.LEAVE_MESSAGE] and self.message is None):
            raise ValueError("LastIterationDecision.LEAVE_MESSAGE cannot be used without a message")

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