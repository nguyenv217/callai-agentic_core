from dataclasses import dataclass, field
from typing import Generic, TypeVar, Protocol, Any, List, Dict, Optional

class DecisionAction(Protocol):
    @property
    def required_message(self) -> bool: ...
    @property
    def name(self) -> str: ...

ActionT = TypeVar("ActionT", bound=DecisionAction)

@dataclass
class DecisionEvent(Generic[ActionT]):
    """Event for observing the decision made by an agent."""    
    action: ActionT
    message: str | None = None

    def __post_init__(self):
        if self.action.required_message and self.message is None:
            error_prefix = f"{type(self.action).__name__}.{self.action.name}"
            raise ValueError(f"{error_prefix} cannot be used without a message")

# --- New Structured Responses and Exceptions ---

class AgenticError(Exception):
    """Base exception for agentic_core"""
    pass

class MCPConnectionError(AgenticError):
    """Raised when connection to an MCP server fails"""
    pass

class ContextLimitExceededError(AgenticError):
    """Raised when the conversation context exceeds the LLM limit"""
    pass

@dataclass
class AgentResponse:
    """Structured response from an agent turn."""
    text: str = ""
    reasoning: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    usage: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "reasoning": self.reasoning,
            "tool_calls": self.tool_calls,
            "usage": self.usage,
            "error": self.error
        }
