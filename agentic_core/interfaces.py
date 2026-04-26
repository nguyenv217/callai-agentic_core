from dataclasses import dataclass, field
from typing import Generic, TypeVar, Protocol, Any

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

# --- Structured Responses and Exceptions ---

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
    result: Any
    error_details: str | None = None
    failed_by: str | None = None

    def to_dict(self) -> dict[str, Any]:
        res = self.result
        if hasattr(res, 'to_dict'):
            res = res.to_dict()
        return {
            "state": self.state,
            "result": res,
            "error_details": self.error_details,
            "failed_by": self.failed_by
        }

@dataclass
class DAGResponse:
    """Structured response from a DAG execution."""
    nodes: dict[str, DAGNodeResponse] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": {node_id: node_resp.to_dict() for node_id, node_resp in self.nodes.items()}
        }

