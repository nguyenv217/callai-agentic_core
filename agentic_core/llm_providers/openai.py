"""
OpenAI LLM Provider.
"""
from typing import List, Dict, Any, Iterator
from .base import ILLMClient, LLMResponse


class OpenAILLM(ILLMClient):
    """OpenAI GPT adapter."""
    
    def __init__(
        self, 
        api_key: str, 
        model: str = "gpt-4o",
        base_url: str = "https://api.openai.com/v1",
        **kwargs
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
        
        self.client = OpenAI(api_key=api_key, base_url=base_url)
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
            yield LLMResponse(success=False, text=None, tool_calls=None, usage=None, error=str(e))