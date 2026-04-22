import pytest
import json
from agentic_core.memory.manager import MemoryManager
from agentic_core.memory.strategies import DefaultTruncationStrategy

def test_default_truncation_strategy_text():
    strategy = DefaultTruncationStrategy(text_threshold=10)
    messages = [{"role": "user", "content": "This is a very long message that should be truncated"}]
    # Max chars set very low to force truncation
    truncated = strategy.truncate(messages, max_chars=15)
    
    assert "[LONG TEXT TRUNCATED]" in truncated[0]["content"]
    assert len(truncated[0]["content"]) <= 10 + len("\n... [LONG TEXT TRUNCATED] ...")

def test_default_truncation_strategy_json_array():
    strategy = DefaultTruncationStrategy(tool_threshold=20)
    # Create a long JSON array
    data = [{"id": i, "val": "some data"} for i in range(10)]
    messages = [{"role": "tool", "content": json.dumps(data)}]
    
    truncated = strategy.truncate(messages, max_chars=50)
    
    assert "[ARRAY TRUNCATED]" in truncated[0]["content"]
    # Check that we kept some elements (the strategy keeps first 3)
    truncated_data = truncated[0]["content"].split("\n... [ARRAY TRUNCATED] ...")[0]
    parsed_data = json.loads(truncated_data)
    assert len(parsed_data) == 3

def test_memory_manager_integration():
    # Use a strategy with a low threshold to ensure truncation occurs
    strategy = DefaultTruncationStrategy(text_threshold=50)
    manager = MemoryManager(max_chars=100, strategy=strategy)
    
    # Add a long message
    long_content = "A" * 200
    manager.add_message({"role": "user", "content": long_content})
    
    # This should trigger the strategy
    manager.enforce_context_limits()
    
    history = manager.get_history()
    assert len(history[0]["content"]) < 200
    assert "[LONG TEXT TRUNCATED]" in history[0]["content"]
    assert "[LONG TEXT TRUNCATED]" in history[0]["content"]

def test_custom_strategy():
    class AlwaysClearStrategy:
        def truncate(self, messages, max_chars):
            return [] # Extreme strategy: clear everything
            
    manager = MemoryManager(strategy=AlwaysClearStrategy())
    manager.add_message({"role": "user", "content": "Hello"})
    manager.enforce_context_limits()
    
    assert len(manager.messages) == 0

def test_message_limit_preservation():
    # Test that _enforce_message_limit still works alongside strategy
    manager = MemoryManager(max_messages=2, max_chars=1000)
    manager.add_message({"role": "user", "content": "1"})
    manager.add_message({"role": "assistant", "content": "2"})
    manager.add_message({"role": "user", "content": "3"})
    manager.add_message({"role": "assistant", "content": "4"})
    
    manager.enforce_context_limits()
    
    # Should only have 2 messages left
    assert len(manager.messages) == 2
    assert manager.messages[0]["content"] == "3"
    assert manager.messages[1]["content"] == "4"
