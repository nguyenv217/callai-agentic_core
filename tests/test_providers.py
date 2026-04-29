import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from agentic_core.llm_providers.anthropic import AnthropicLLM
from agentic_core.llm_providers.ollama import OllamaLLM
from agentic_core.llm_providers.base import LLMResponse

async def _collect_async_tolist(resp_iter) -> list[LLMResponse]:
    return [item async for item in resp_iter]

@pytest.mark.asyncio
async def test_anthropic_successful_tool_call():
    with patch("anthropic.AsyncAnthropic") as MockAnthropic:
        mock_client = MockAnthropic.return_value
        
        # Mocking Anthropic's block response structure
        mock_text_block = MagicMock(type="text", text="Here is the weather.")
        mock_tool_block = MagicMock(type="tool_use", id="tool_123", input={"location": "Tokyo"})
        mock_tool_block.name = "get_weather"
        
        mock_response = MagicMock()
        mock_response.content = [mock_text_block, mock_tool_block]
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 20
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        provider = AnthropicLLM(api_key="fake-key", model="claude-3-5-sonnet")
        provider.client = mock_client # Override with our mock
        
        schema = [{"function": {"name": "get_weather", "description": "", "parameters": {}}}]
        responses = await _collect_async_tolist(provider.ask(messages=[{"role": "user", "content": "Weather?"}], tools=schema))

        assert responses[0].text == "Here is the weather."
        assert len(responses[0].tool_calls) == 1
        assert responses[0].tool_calls[0]["name"] == "get_weather"

@pytest.mark.asyncio
async def test_ollama_error_handling():
    with patch("ollama.AsyncClient") as MockOllama:
        mock_client = MockOllama.return_value
        mock_client.chat = AsyncMock(side_effect=Exception("Connection refused"))

        provider = OllamaLLM(model="llama3.1")
        provider.client = mock_client
        
        with pytest.raises(Exception, match="Connection refused"):
            responses = await _collect_async_tolist(provider.ask(messages=[{"role": "user", "content": "Hi"}]))
        