"""
LLM Provider base interfaces.
"""
from typing import Any, Protocol, AsyncIterator
from dataclasses import dataclass

@dataclass
class LLMResponse:
    """Response from an LLM provider."""
    success: bool
    text: str | None = None
    tool_calls: list[dict] | None = None
    usage: dict | None = None
    error: str | None = None
    reasoning: str | None = None

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