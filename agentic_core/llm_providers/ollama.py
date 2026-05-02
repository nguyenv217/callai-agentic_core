"""
Ollama LLM Provider.
"""
from typing import AsyncIterator
from .base import ILLMClient, LLMResponse
from ..interfaces import Message, ToolSchema

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
        messages: list[Message], 
        tools: list[ToolSchema] | None = None, 
        stream: bool = True,
        **kwargs
    ) -> AsyncIterator[LLMResponse]:
        """
        Send a chat request to Ollama.
        
        Args:
            messages: Conversation history.
            tools: A list of JSON schemas for tools
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

        if stream:
            stream_response = await self.client.chat(stream=True, **req_kwargs)
            async for chunk in stream_response:
                msg = chunk.message
                if msg.tool_calls:
                    tool_calls = []
                    for tc in msg.tool_calls:
                            # ollama doesnt take id nor type, simply assumes `function``
                            tool_calls.append({
                                "id": "",
                                "type":"function",
                                "function": {
                                    "name": tc.function.name if tc.function else "",
                                    "arguments": tc.function.arguments if tc.function else ""
                                }
                            })
                    yield LLMResponse(
                        text=msg.content or "",
                        reasoning=msg.thinking or "",
                        tool_calls=tool_calls,
                    finish_reason="tool_calls", # no accumulation needed
                        usage={}
                    )
                else:                            
                    yield LLMResponse(
                        text=msg.content or "",
                        reasoning=msg.thinking or "",
                        usage={}
                    )
            return

        response = await self.client.chat(stream=False, **req_kwargs)
        
        msg = response.message
        yield LLMResponse(
            text=msg.content,
            reasoning=msg.thinking,
            tool_calls=[
                {
                    "id": "", 
                    "type": "function", 
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                } for tc in msg.tool_calls
            ] if msg.tool_calls else [],
            usage={},
        )