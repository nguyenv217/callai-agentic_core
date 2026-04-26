
import asyncio
import pytest
from agentic_core.agents.builder import chat
from agentic_core.interfaces import AgentResponse
from agentic_core.engines.session import global_session_manager

@pytest.mark.asyncio
async def test_session_persistence():
    # Mocking providers to avoid API keys
    # We can use a dummy runner or a mock
    from unittest.mock import MagicMock

    # We'll just check if the SessionManager stores the runner
    session_id = "test_user_123"

    # Create a dummy runner
    mock_runner = MagicMock()

    # Use the session manager manually
    async def creator():
        return mock_runner

    runner1 = await global_session_manager.get_runner(session_id, creator)
    runner2 = await global_session_manager.get_runner(session_id, creator)

    assert runner1 is runner2
    assert runner1 == mock_runner

@pytest.mark.asyncio
async def test_chat_response_type():
    # This test requires a provider, but we can mock the agent inside chat
    # Actually, let's just test that the return type of chat is AgentResponse
    # by mocking the runner.
    from unittest.mock import AsyncMock, MagicMock
    from agentic_core.engines.engine import AgentRunner
    from agentic_core.interfaces import AgentResponse

    mock_runner = MagicMock(spec=AgentRunner)
    mock_runner.run_turn = AsyncMock(return_value=AgentResponse(text="Hello world"))

    result = await chat("Hi", runner=mock_runner)

    assert isinstance(result, AgentResponse)
    assert result.text == "Hello world"

@pytest.mark.asyncio
async def test_mcp_registry_granular_locks():
    from agentic_core.tools.mcp.manager import GlobalMCPRegistry
    registry = GlobalMCPRegistry()

    config1 = {"command": "python", "args": ["server1.py"], "env": {}}
    config2 = {"command": "python", "args": ["server2.py"], "env": {}}

    lock1 = await registry._get_lock_for_identity(registry._get_identity_key(config1))
    lock2 = await registry._get_lock_for_identity(registry._get_identity_key(config2))

    assert lock1 is not lock2
