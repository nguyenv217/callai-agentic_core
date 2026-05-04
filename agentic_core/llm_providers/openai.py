"""
OpenAI LLM Provider.
"""
from typing import Any, AsyncIterator

from ..config import ConfigurationError
from .base import ILLMClient, LLMResponse
from ..interfaces import ProviderAuthenticationError, ProviderRateLimitError, ProviderTimeoutError, Message, ToolSchema

_OPENAI_IMPORTED=True
try:
    from openai import AsyncOpenAI, AuthenticationError, RateLimitError, BadRequestError, APIConnectionError, APITimeoutError
except ImportError:
    _OPENAI_IMPORTED=False

class OpenAILLM(ILLMClient):
    """OpenAI GPT adapter."""
    
    def __init__(
        self, 
        model: str,
        api_key: str | None = None, 
        base_url: str = "https://api.openai.com/v1",
        client: AsyncOpenAI | None = None,
        timeout: float = 30,
        **kwargs
    ):
        if not _OPENAI_IMPORTED:
            raise ImportError("Missing dependency `openai`. Please install via `pip install openai`")
        
        if client:
            self.client = client
        elif api_key:
            self.client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        else:
            raise ConfigurationError("Please pass either a valid api_key as 'api_key' or a configured `AsyncOpenAI` client instance as 'client'")
            
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
        Send a chat completion request to OpenAI asynchronously.
        
        Args:
            messages: Conversation history.
            tools: A list of JSON schemas for tools (NOT the ToolManager object).
            stream: If True, yields chunks as they arrive for streaming compatibility.
            **kwargs: Extra body parameters to pass to the OpenAI API.
        
        Yields:
            AsyncIterator[LLMResponse]: An iterator of LLMResponse objects.
        """
        try:
            # Build request kwargs dynamically to avoid SDK validation errors when tools=None is passed
            req_kwargs = {
                "model": self.model,
                "messages": messages,
                **self.extra_kwargs,
                **kwargs
            }
            
            if tools:
                req_kwargs["tools"] = tools

            accumulated_tools = {}
            active_index_map = {}
            
            if stream:
                stream_response = await self.client.chat.completions.create(stream=True, **req_kwargs)
            
                async for chunk in stream_response:
                    if not chunk.choices:
                        continue
                    
                    choice = chunk.choices[0]
                    delta = choice.delta

                    if delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            api_idx = tc_delta.index if tc_delta.index else 0

                            if api_idx not in active_index_map:
                                active_index_map[api_idx] = len(accumulated_tools)
                            else:
                                virtual_idx = active_index_map[api_idx]
                                existing_tool = accumulated_tools[virtual_idx]
                                
                                # if API reused the index for a new tool
                                if tc_delta.id and existing_tool["id"] and tc_delta.id != existing_tool["id"]:
                                    active_index_map[api_idx] = len(accumulated_tools)

                            idx = active_index_map[api_idx]

                            # If this is a new tool call index, initialize it
                            if idx not in accumulated_tools:
                                accumulated_tools[idx] = {
                                    "id": "",
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""}
                                }
                            
                            target = accumulated_tools[idx]
                            
                            # Update ID if present (usually only in the first chunk for this index)
                            if tc_delta.id:
                                target["id"] = tc_delta.id
                            
                            if tc_delta.function:
                                if tc_delta.function.name:
                                    target["function"]["name"] += tc_delta.function.name
                                if tc_delta.function.arguments:
                                    target["function"]["arguments"] += tc_delta.function.arguments

                    current_tool_calls = list(accumulated_tools.values())
                    
                    yield LLMResponse(
                        text=delta.content or "",
                        reasoning=getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None) or getattr(delta, "thinking", None) or "",
                        tool_calls=current_tool_calls,
                        usage=chunk.usage or {}, # only yielded at last chunk
                    )

                return

            response = await self.client.chat.completions.create(stream=False,**req_kwargs)

            msg = response.choices[0].message
            yield LLMResponse(
                text=msg.content or "",
                reasoning=getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or getattr(msg, "thinking", None) or "",
                tool_calls=[tc.model_dump() for tc in msg.tool_calls] if msg.tool_calls else [],
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens
                }
            )

        except AuthenticationError:
            raise ProviderAuthenticationError(f"openai client: Invalid API key for {self.model}")
        except RateLimitError:
            raise ProviderRateLimitError(f"openai client: Rate limit exceeded for {self.model}")
        except APITimeoutError:
            raise ProviderTimeoutError(f"openai client: Request timed out for {self.model}")
