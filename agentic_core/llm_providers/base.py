"""
LLM Provider base interfaces.
"""
from typing import Any, Protocol, AsyncIterator
from dataclasses import dataclass

from agentic_core.interfaces import ToolResponse

@dataclass
class LLMResponse:
    """
    Response from an LLM provider.

    Attributes:
        text: The text response from the LLM.
        tool_calls: List of tool calls (if any).
        usage: Token usage information.
        reasoning: Optional reasoning trace (if supported by the LLM/provider).
        finish_reason: Indicating why stream stopped. This is useful for signalling state transitions during streaming for the engine. Generally, users do not need to make use of it.
    """
    text: str | None = None
    tool_calls: list[ToolResponse] | None = None
    usage: dict | None = None
    reasoning: str | None = None
    finish_reason: str | None = None

class ILLMClient(Protocol):
    """
    Protocol for LLM clients.
    
    Note: tools should be a list of JSON schemas (list[dict]), NOT a ToolManager object.
    """
    
    def ask(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs
    ) -> AsyncIterator[LLMResponse]: 
        """
        Send a request to the LLM.
        
        Args:
            messages: Conversation history as list of message dicts.
            tools: A list of JSON schemas for tools (NOT the ToolManager object).
                   Pass None or empty list if no tools are available.
        
        Yields:
            LLMResponse objects with the model's response.
        """
        ...