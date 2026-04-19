"""
Anthropic LLM Provider.
"""
from typing import List, Dict, Any, Iterator
from .base import ILLMClient, LLMResponse


class AnthropicLLM(ILLMClient):
    """Anthropic Claude adapter."""
    
    def __init__(
        self, 
        api_key: str, 
        model: str = "claude-3-5-sonnet-20241022",
        **kwargs
    ):
        try:
            import agentic_core.llm_providers.anthropic as anthropic  # type: ignore
        except ImportError:
            raise ImportError("Please install anthropic: pip install anthropic")
        
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.extra_kwargs = kwargs
    
    def ask(
        self, 
        messages: List[Dict[str, Any]], 
        tools: List[Dict[str, Any]] | None = None, 
        **kwargs
    ) -> Iterator[LLMResponse]:
        """
        Send a message request to Anthropic.
        
        Args:
            messages: Conversation history.
            tools: A list of JSON schemas for tools (NOT the ToolManager object).
        """
        try:
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

            response = self.client.messages.create(**req_kwargs)
            
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
                success=True,
                text=text_content,
                tool_calls=tool_calls,
                usage={
                    "prompt_tokens": response.usage.input_tokens, 
                    "completion_tokens": response.usage.output_tokens
                },
                error=None
            )
        except Exception as e:
            yield LLMResponse(success=False, text=None, tool_calls=None, usage=None, error=str(e))