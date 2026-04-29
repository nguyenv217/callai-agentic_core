from typing import AsyncIterator

import pytest
import pytest_asyncio
from unittest.mock import MagicMock
from unittest.mock import AsyncMock
from agentic_core.interfaces import ProviderAuthenticationError, ProviderRateLimitError
from agentic_core.llm_providers.base import LLMResponse
from agentic_core.llm_providers.openai import OpenAILLM

# Skip if openai is not installed in the test environment
pytest.importorskip("openai")
import openai


async def _collect_async_tolist(resp_iter: AsyncIterator[LLMResponse]) -> list[LLMResponse]:
    results = []
    async for item in resp_iter:
        results.append(item)

    return results


@pytest_asyncio.fixture
def mock_openai_client():
    """Fixture to mock the OpenAI client instance."""
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock()
    return mock_client


@pytest.mark.asyncio
async def test_openai_auth_error_handling(mock_openai_client):
    """Verifies that AuthenticationError raises ProviderAuthenticationError."""
    # Setup the mock to raise an Auth Error
    mock_error = openai.AuthenticationError(
        message="Invalid API key", 
        response=MagicMock(), 
        body=None
    )
    mock_openai_client.chat.completions.create.side_effect = mock_error

    provider = OpenAILLM(client=mock_openai_client, model="gpt-4")
    with pytest.raises(ProviderAuthenticationError):
        await _collect_async_tolist(provider.ask(messages=[{"role": "user", "content": "Hello"}]))


@pytest.mark.asyncio
async def test_openai_rate_limit_handling(mock_openai_client):
    """Verifies that RateLimitError raises ProviderRateLimitError."""
    mock_openai_client.chat.completions.create.side_effect = openai.RateLimitError(
        message="You have hit your rate limit.",
        response=MagicMock(),
        body=None
    )

    provider = OpenAILLM(client=mock_openai_client, model="gpt-4")
    with pytest.raises(ProviderRateLimitError):
        await _collect_async_tolist(provider.ask(messages=[{"role": "user", "content": "Hello"}]))

@pytest.mark.asyncio
async def test_openai_successful_tool_call(mock_openai_client):
    """Verifies that a valid response with tool calls is correctly deserialized."""
    mock_message = MagicMock()
    mock_message.content = None
    
    mock_tool_call = MagicMock()
    mock_tool_call.model_dump.return_value = {
        "id": "call_123",
        "type": "function",
        "function": {"name": "get_weather", "arguments": '{"location": "Tokyo"}'}
    }
    mock_message.tool_calls = [mock_tool_call]

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 20
    
    mock_openai_client.chat.completions.create.return_value = mock_response

    provider = OpenAILLM(client=mock_openai_client, model="gpt-4")
    responses = await _collect_async_tolist(provider.ask(messages=[{"role": "user", "content": "Weather?"}]))

    print(responses)

    assert len(responses[0].tool_calls) == 1
    assert responses[0].tool_calls[0]["id"] == "call_123"
    assert responses[0].usage["completion_tokens"] == 20
