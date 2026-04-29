"""
Ollama LLM Provider.
"""
from typing import AsyncIterator, Any
from .base import ILLMClient, LLMResponse


class OllamaLLM(ILLMClient):
    """Ollama local LLM adapter."""
    
    def __init__(
        self, 
        model: str = "llama3.1",
        base_url: str = "http://localhost:11434",
        **kwargs
    ):
        try:
            from ollama import AsyncClient
        except ImportError:
            raise ImportError("Please install ollama: pip install ollama")
        
        self.model = model
        self.base_url = base_url
        self.client = AsyncClient(host=base_url)
        self.extra_kwargs = kwargs
    
    async def ask(
        self, 
        messages: list[dict[str, Any]], 
        tools: list[dict[str, Any]] | None = None, 
        **kwargs
    ) -> AsyncIterator[LLMResponse]:
        """
        Send a chat request to Ollama.
        
        Args:
            messages: Conversation history.
            tools: A list of JSON schemas for tools (NOT the ToolManager object).
        """
        req_kwargs = {
            "model": self.model,
            "messages": messages,
            **self.extra_kwargs,
            **kwargs
        }
        
        # To ollama format
        if tools:
            req_kwargs["tools"] = [{"type": "function", "function": t["function"]} for t in tools]

        response = await self.client.chat(**req_kwargs)
        
        msg = response["message"]
        yield LLMResponse(
            text=msg.get("content", ""),
            tool_calls=msg.get("tool_calls", []),
            usage={},
        )