"""
LLM Provider base interfaces.
"""
from typing import Iterator, List, Dict, Any, Protocol, AsyncIterator
from dataclasses import dataclass

@dataclass
class LLMResponse:
    """Response from an LLM provider."""
    success: bool
    text: str | None = None
    tool_calls: List[Dict] | None = None
    usage: Dict | None = None
    error: str | None = None
    reasoning: str | None = None


class ILLMClient(Protocol):
    """
    Protocol for LLM clients.
    
    Note: tools should be a list of JSON schemas (list[dict]), NOT a ToolManager object.
    """
    
    def ask(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] | None = None,
        **kwargs
    ) -> Iterator[LLMResponse] | AsyncIterator[LLMResponse]: # THIS WILL BE STANDARDIZED LATER TO ASYNC, please migrate
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