from dataclasses import dataclass, field
from typing import Any
from enum import Enum


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

class IterationLimitReachedError(Exception):
    """Raised when the agent exceeds the maximum number"""
    pass

# ===================================================
# Structured Responses
# ===================================================

@dataclass
class AgentResponse:
    """Structured response from an agent turn.
    
    Attributes:
        text (str): The text response from the agent
        reasoning (str): The reasoning behind the response
        tool_calls (list[dict[str, Any]]: List of tool calls
        usage (dict[str, Any]): Usage information from the inference API
        error (`BaseException` | None): Error occured during execution
    """
    text: str = ""
    reasoning: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
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

    Attributes:
        state (str): One of the following: "PENDING", "READY", "RUNNING", "SUCCESS", "FAILED", "FAILED_UPSTREAM", "RETRYING"
        result (`AgentResponse` | None): The result of the node execution if successful.
        error (`BaseException` | None): Error occured during execution
        error_details (str | None): Error details if any
        failed_by (str | None): Id of parent node failed upstreamed 
        
    """
    state: str
    result: AgentResponse | None
    error: BaseException | None = None # due to the nature of swarm we must store this
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

@dataclass
class StreamEvent:
    """Event yielded during a streaming agent turn."""
    type: StreamEventType
    content: Any
    error: BaseException | None = None



