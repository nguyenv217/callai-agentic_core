"""
Ollama LLM Provider.
"""
from typing import AsyncIterator, List, Dict, Any
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
        messages: List[Dict[str, Any]], 
        tools: List[Dict[str, Any]] | None = None, 
        **kwargs
    ) -> AsyncIterator[LLMResponse]:
        """
        Send a chat request to Ollama.
        
        Args:
            messages: Conversation history.
            tools: A list of JSON schemas for tools (NOT the ToolManager object).
        """
        try:
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
                success=True,
                text=msg.get("content", ""),
                tool_calls=msg.get("tool_calls", []),
                usage={},
                error=None
            )
        except Exception as e:
            yield LLMResponse(success=False, text=None, tool_calls=None, usage=None, error=str(e))