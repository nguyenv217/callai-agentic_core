"""
Anthropic LLM Provider.
"""
from typing import AsyncIterator, Any

from .base import ILLMClient, LLMResponse
from ..interfaces import Message, ProviderAuthenticationError, ProviderRateLimitError, ProviderTimeoutError, ToolSchema

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
        messages: list[Message], 
        tools: list[ToolSchema] | None = None, 
        stream: bool = False,
        **kwargs
    ) -> AsyncIterator[LLMResponse]:
        """
        Send a message request to Anthropic.
        
        Args:
            messages: Conversation history.
            tools: A list of JSON schemas for tools (NOT the `ToolManager` object).
        """
        from anthropic.types import RateLimitError, AuthenticationError, GatewayTimeoutError

        system = None
        anthropic_messages = []
        
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            
            if role == "system":
                system = content
            elif role == "tool":
                # Anthropic requires tool results to be role: "user" with a `tool_result` content block.
                anthropic_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.get("tool_call_id"),
                        "content": str(content)
                    }]
                })
            else:
                anthropic_messages.append({"role": role, "content": content})

        req_kwargs = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": kwargs.get("max_tokens", 4096),
            **self.extra_kwargs,
            **kwargs
        }
        
        if system:
            req_kwargs["system"] = system
        
        if tools:
            req_kwargs["tools"] = [
                {
                    "name": t["function"]["name"],
                    "description": t["function"]["description"],
                    "input_schema": t["function"]["parameters"]
                }
                for t in tools
            ]
        
        try:
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
                                        "type": "function",
                                        "function": {
                                            "name": block.name,
                                            "arguments": block.input
                                        }
                                    }],
                                    finish_reason="tool_calls" # no accumulation needed
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
    
        except AuthenticationError:
            raise ProviderAuthenticationError(f"openai client: Invalid API key for {self.model}")
        except RateLimitError:
            raise ProviderRateLimitError(f"openai client: Rate limit exceeded for {self.model}")
        except GatewayTimeoutError:
            raise ProviderTimeoutError(f"openai client: Request timed out for {self.model}")
