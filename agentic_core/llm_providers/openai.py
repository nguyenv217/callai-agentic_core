"""
OpenAI LLM Provider.
"""
from typing import List, Dict, Any, Iterator
from .base import ILLMClient, LLMResponse

_openai_imported=True
try:
    from openai import OpenAI
except ImportError:
    _openai_imported=False

class OpenAILLM(ILLMClient):
    """OpenAI GPT adapter."""
    
    def __init__(
        self, 
        api_key: str | None = None, 
        model: str = "gpt-4o",
        base_url: str = "https://api.openai.com/v1",
        client: OpenAI | None = None,
        timeout: float = 30,
        **kwargs
    ):
        if not _openai_imported:
            raise ImportError("Please install openai: pip install openai")
        
        if client:
            self.client = client
        elif api_key:
            self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        else:
            raise RuntimeError("Please pass either a valid api_key as 'api_key' or a configured OpenAI client instance as 'client")
            
        self.model = model
        self.extra_kwargs = kwargs
    
    def ask(
        self, 
        messages: List[Dict[str, Any]], 
        tools: List[Dict[str, Any]] | None = None, 
        **kwargs
    ) -> Iterator[LLMResponse]:
        """
        Send a chat completion request to OpenAI.
        
        Args:
            messages: Conversation history.
            tools: A list of JSON schemas for tools (NOT the ToolManager object).
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

            # Make the API call
            response = self.client.chat.completions.create(**req_kwargs)
            
            msg = response.choices[0].message
            yield LLMResponse(
                success=True,
                text=msg.content or "",
                tool_calls=[tc.model_dump() for tc in msg.tool_calls] if msg.tool_calls else [],
                usage={
                    "prompt_tokens": response.usage.prompt_tokens, 
                    "completion_tokens": response.usage.completion_tokens
                },
                error=None
            )

        except Exception as e:
            # We import here to ensure the core doesn't crash if openai isn't installed
            # but a user accidentally instantiates the provider
            try:
                import openai
            except ImportError:
                yield LLMResponse(success=False, error="OpenAI library is not installed.", text=None)
                return
                
            error_msg = str(e)
            if isinstance(e, openai.AuthenticationError):
                error_msg = f"FATAL AUTH ERROR: Invalid or missing OpenAI API key. ({e})"
            elif isinstance(e, openai.RateLimitError):
                error_msg = f"RATE LIMIT: OpenAI quota exceeded or rate limited. ({e})"
            elif isinstance(e, openai.APIConnectionError):
                error_msg = f"NETWORK ERROR: Failed to connect to OpenAI API. ({e})"
            elif isinstance(e, openai.BadRequestError):
                error_msg = f"BAD REQUEST: Invalid parameters or context window exceeded. ({e})"
                
            yield LLMResponse(
                success=False, 
                text=None, 
                tool_calls=[], 
                usage={}, 
                error=error_msg
            )