"""
Anthropic LLM Provider.
"""
from typing import AsyncIterator, Any

from anthropic import AsyncStream
from .base import ILLMClient, LLMResponse

class AnthropicLLM(ILLMClient):
    """Anthropic Claude adapter."""
    
    def __init__(
        self, 
        api_key: str, 
        model: str,
        **kwargs
    ):
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            raise ImportError("Please install anthropic: pip install anthropic")
        
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model
        self.extra_kwargs = kwargs
    
    async def ask(
        self, 
        messages: list[dict[str, Any]], 
        tools: list[dict[str, Any]] | None = None, 
        stream: bool = False,
        **kwargs
    ) -> AsyncIterator[LLMResponse]:
        """
        Send a message request to Anthropic.
        
        Args:
            messages: Conversation history.
            tools: A list of JSON schemas for tools (NOT the ToolManager object).
        """
        # Convert messages to Anthropic format
        system = None
        anthropic_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system = msg.get("content", "")
            else:
                anthropic_messages.append(msg)
        
        # Build request kwargs dynamically
        req_kwargs = {
            "model": self.model,
            "messages": anthropic_messages,
            **self.extra_kwargs,
            **kwargs
        }
        
        # Only attach system and tools if they exist
        if system:
            req_kwargs["system"] = system
        
        if tools:
            # Convert OpenAI-style tool schema to Anthropic format
            anthropic_tools = [
                {
                    "name": t["function"]["name"],
                    "description": t["function"]["description"],
                    "input_schema": t["function"]["parameters"]
                }
                for t in tools
            ]
            req_kwargs["tools"] = anthropic_tools
    
        
        if stream:
            async with self.client.messages.stream(**req_kwargs) as event_stream:
                async for event in event_stream:
                    if event.type == "text":
                        yield LLMResponse(text=event.text)

                    if event.type == "thinking":
                        yield LLMResponse(reasoning=event.thinking)
                    
                    elif event.type == "content_block_stop":
                        block = event.content_block
                        if block.type == "tool_use":
                            yield LLMResponse(
                                tool_calls=[{
                                    "id": block.id,
                                    "name": block.name,
                                    "arguments": block.input 
                                }]
                            )

            return
        
        response = await self.client.messages.create(**req_kwargs)
        text_content = ""
        tool_calls = []
        
        for block in response.content:
            if block.type == "text":
                text_content += block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "name": block.name,
                    "id": block.id,
                    "arguments": block.input
                })
        
        yield LLMResponse(
            text=text_content,
            tool_calls=tool_calls,
            usage={
                "prompt_tokens": response.usage.input_tokens, 
                "completion_tokens": response.usage.output_tokens
            },
        )