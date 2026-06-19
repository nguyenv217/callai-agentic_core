from dataclasses import dataclass, field
from typing import Any
from enum import Enum
from .interfaces import ToolResponse

@dataclass
class AgentResponse:
    """Structured response from an agent turn."""
    text: str = ""
    reasoning: str = ""
    tool_calls: list[ToolResponse] = field(default_factory=list)
    usage: dict[str, Any]  | None = None
    error: BaseException | None = None

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
    """
    Response for a single node in a DAG.
    """
    state: str
    result: AgentResponse | None
    error: BaseException | None = None
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
    ERROR = "error"
    FINAL_RESPONSE = "final_response"
    SUSPENDED = "suspended"

@dataclass
class StreamEvent:
    """Event yielded during a streaming agent turn."""
    type: StreamEventType
    content: Any = None
    error: BaseException | None = None
