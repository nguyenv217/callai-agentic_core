from dataclasses import dataclass, field
from typing import Generic, TypeVar, Protocol, Any
from enum import Enum

# ===================================================
# Decision Events
# ===================================================

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

# ===================================================
# Exceptions 
# ===================================================

class AgenticError(Exception):
    """Base exception for agentic_core"""
    pass

class MCPConnectionError(AgenticError):
    """Raised when connection to an MCP server fails"""
    pass

class ContextLimitExceededError(AgenticError):
    """Raised when the conversation context exceeds the LLM limit"""
    pass

class ProviderAuthenticationError(AgenticError):
    """Raised when the provider authentication fails"""
    pass

class ProviderRateLimitError(AgenticError):
    """Raised when the provider rate limit is exceeded"""
    pass

class ProviderTimeoutError(AgenticError):
    """Raised when the provider request times out"""
    pass

# ===================================================
# Structured Responses
# ===================================================

@dataclass
class AgentResponse:
    """Structured response from an agent turn."""
    text: str = ""
    reasoning: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "reasoning": self.reasoning,
            "tool_calls": self.tool_calls,
            "usage": self.usage,
            "error": self.error
        }

@dataclass
class DAGNodeResponse:
    """Response for a single node in a DAG."""
    state: str
    result: AgentResponse | None
    error: BaseException | None # due to the nature of swarm we must store this
    error_details: str | None = None
    failed_by: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "result": self.result,
            "error_details": self.error_details,
            "failed_by": self.failed_by
        }

@dataclass
class DAGResponse:
    """Structured response from a DAG execution."""
    nodes: dict[str, DAGNodeResponse] = field(default_factory=dict)
    error: BaseException | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": {node_id: node_resp.to_dict() for node_id, node_resp in self.nodes.items()},
            "error": self.error
        }

class StreamEventType(Enum):
    """Types of events that can be streamed from an agent turn."""
    TEXT = "text"
    REASONING = "reasoning"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    FINAL_RESPONSE = "final_response"
    ERROR = "error"

@dataclass
class StreamEvent:
    """Event yielded during a streaming agent turn."""
    type: StreamEventType
    content: Any



