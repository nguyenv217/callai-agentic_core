import pytest
from unittest.mock import patch, MagicMock
from agentic_core.llm_providers.openai import OpenAILLM

# Skip if openai is not installed in the test environment
pytest.importorskip("openai")
import openai

@pytest.fixture
def mock_openai_client():
    """Fixture to mock the OpenAI client instance."""
    with patch("agentic_core.llm_providers.openai.OpenAILLM") as mock_client:
        yield mock_client.return_value

def test_openai_auth_error_handling(mock_openai_client):
    """Verifies that AuthenticationError is caught and mapped correctly."""
    # Setup the mock to raise an Auth Error
    mock_error = openai.AuthenticationError(
        message="Invalid API key", 
        response=MagicMock(), 
        body=None
    )
    mock_openai_client.chat.completions.create.side_effect = mock_error

    provider = OpenAILLM(api_key="bad_key")
    # provider.ask yields an iterator
    responses = list(provider.ask(messages=[{"role": "user", "content": "Hello"}]))

    assert len(responses) == 1
    assert responses[0].success is False
    assert "FATAL AUTH ERROR" in responses[0].error

def test_openai_rate_limit_handling(mock_openai_client):
    """Verifies that RateLimitError is handled safely."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = openai.RateLimitError(
        message="You have hit your rate limit.",
        response=MagicMock(),
        body=None
    )

    provider = OpenAILLM(client=mock_client)
    responses = list(provider.ask(messages=[{"role": "user", "content": "Hello"}]))
    
    assert "RATE LIMIT" in responses[0].error

def test_openai_successful_tool_call(mock_openai_client):
    """Verifies that a valid response with tool calls is correctly deserialized."""
    # Create a mock response object mirroring the OpenAI SDK structure
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


    provider = OpenAILLM(client=mock_openai_client)
    responses = list(provider.ask(messages=[{"role": "user", "content": "Weather?"}]))

    assert responses[0].success is True
    assert len(responses[0].tool_calls) == 1
    assert responses[0].tool_calls[0]["id"] == "call_123"
    assert responses[0].usage["completion_tokens"] == 20