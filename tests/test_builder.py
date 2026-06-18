import pytest
from agentic_core.agents.builder import AgentBuilder, chat
from agentic_core.llm_providers import AnthropicLLM, OllamaLLM

def test_create_anthropic_agent():
    """Test Anthropic agent creation and default properties."""
    agent = AgentBuilder().with_provider_anthropic(api_key="test_key").with_system_prompt("Test Claude").build()
    assert isinstance(agent.llm, AnthropicLLM)
    assert agent.llm.model == "claude-3-5-sonnet-20241022"
    assert agent.memory.system_prompt['content'] == "Test Claude"

def test_create_ollama_agent():
    """Test Ollama agent creation and default properties."""
    agent = AgentBuilder().with_provider_ollama(model="llama3.1").with_system_prompt("Test Ollama").build()
    assert isinstance(agent.llm, OllamaLLM)
    assert agent.llm.model == "llama3.1"
    assert agent.memory.system_prompt['content'] == "Test Ollama"

@pytest.mark.asyncio
async def test_chat_invalid_provider():
    """Test that the chat convenience function rejects unknown providers."""
    with pytest.raises(ValueError, match="Unknown provider: random. Use 'openai', 'anthropic', or 'ollama'"):
        await chat(message="Hello", provider="random")
