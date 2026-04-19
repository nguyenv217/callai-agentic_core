import pytest
from agentic_core.memory.manager import MemoryManager
from agentic_core.agents.builder import create_ollama_agent
from agentic_core.engine import AgentRunner

def test_memory_manager_initialization():
    """Ensure the memory manager initializes and handles system prompts correctly."""
    memory = MemoryManager(max_messages=50, max_chars=80000)
    memory.set_system_prompt("You are a test agent.")
    
    # Check internal state
    assert memory.system_prompt["role"] == "system"
    assert memory.system_prompt["content"] == "You are a test agent."
    assert len(memory.messages) == 0  # Standard messages list should be empty
    
    # Check history generation (system prompt + messages)
    history = memory.get_history()
    assert len(history) == 1
    assert history[0]["content"] == "You are a test agent."

def test_memory_manager_message_limits():
    """Ensure message pruning logic triggers."""
    memory = MemoryManager(max_messages=2)
    memory.add_message({"role": "user", "content": "Hello"})
    memory.add_message({"role": "assistant", "content": "Hi there"})
    
    # Adding a 3rd message should trigger _enforce_message_limit()
    memory.add_message({"role": "user", "content": "Another message"})
    
    # Depending on pruning pair logic, it should never exceed max_messages
    assert len(memory.messages) <= 2

def test_agent_factory_initialization():
    """Ensure the factory functions successfully wire up the Engine, LLM, and Memory."""
    # Using Ollama here as it requires no API keys to instantiate
    agent = create_ollama_agent(
        model="test-model",
        system_prompt="Factory test persona"
    )
    
    # Validate the wiring
    assert isinstance(agent, AgentRunner)
    assert agent.llm.model == "test-model"
    assert agent.memory.system_prompt["content"] == "Factory test persona"
    assert agent.tools is not None