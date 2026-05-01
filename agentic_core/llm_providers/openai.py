"""
OpenAI LLM Provider.
"""
from typing import Any, AsyncIterator

from openai import AsyncStream

from ..config import ConfigurationError
from .base import ILLMClient, LLMResponse
from ..interfaces import ProviderAuthenticationError, ProviderRateLimitError, ProviderTimeoutError

_OPENAI_IMPORTED=True
try:
    from openai import AsyncOpenAI, AuthenticationError, RateLimitError, BadRequestError, APIConnectionError, APITimeoutError
    from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
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
        messages: list[dict[str, Any]], 
        tools: list[dict[str, Any]] | None = None, 
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
            # Build request kwargs dynamically to avoid SDK validation errors
            # when tools=None is passed (which causes 400 Bad Request)
            req_kwargs = {
                "model": self.model,
                "messages": messages,
                **self.extra_kwargs,
                **kwargs
            }
            
            # Only attach tools if actually present - prevents "None" validation errors
            if tools:
                req_kwargs["tools"] = tools

            # Handle streaming mode for compatibility with stream_engine.py
            if stream:
                req_kwargs["stream"] = True
                stream_response: AsyncStream[ChatCompletionChunk] = await self.client.chat.completions.create(**req_kwargs)
            
                async for chunk in stream_response:
                    if not chunk.choices:
                        continue
                    
                    delta = chunk.choices[0].delta

                    # Collect tool calls as they appear
                    tool_calls = []
                    if delta.tool_calls:
                        for tc in delta.tool_calls:
                            tool_calls.append({
                                "id": tc.id or "",
                                "type": tc.type or "function",
                                "function": {
                                    "name": tc.function.name if tc.function else "",
                                    "arguments": tc.function.arguments if tc.function else ""
                                }
                            })

                    yield LLMResponse(
                        text=delta.content or "",
                        reasoning=getattr(delta, "reasoning_content", "") or "",
                        tool_calls=tool_calls,
                        usage={}
                    )

                return

            response = await self.client.chat.completions.create(**req_kwargs)

            msg = response.choices[0].message
            yield LLMResponse(
                text=msg.content or "",
                reasoning= getattr(msg, "reasoning_content", "") or getattr(msg, "reasoning", ""),
                tool_calls=[tc.model_dump() for tc in msg.tool_calls] if msg.tool_calls else [],
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens
                }
            )

        except AuthenticationError:
            raise ProviderAuthenticationError
        except RateLimitError:
            raise ProviderRateLimitError
        except APITimeoutError:
            raise ProviderTimeoutError
