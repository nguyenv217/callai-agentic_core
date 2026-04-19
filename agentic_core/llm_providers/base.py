"""
LLM Provider base interfaces.
"""
from typing import Iterator, List, Dict, Any, Protocol


class LLMResponse:
    """Response from an LLM provider."""
    
    def __init__(
        self,
        success: bool,
        text: str | None = None,
        tool_calls: List[Dict] | None = None,
        usage: Dict | None = None,
        error: str | None = None
    ):
        self.success = success
        self.text = text
        self.tool_calls = tool_calls
        self.usage = usage
        self.error = error


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
    ) -> Iterator[LLMResponse]:
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