
import pytest
from agentic_core.engines.session import global_session_manager

@pytest.mark.asyncio
async def test_session_persistence():
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
async def test_mcp_registry_granular_locks():
    from agentic_core.tools.mcp.manager import GlobalMCPRegistry
    registry = GlobalMCPRegistry()

    config1 = {"command": "python", "args": ["server1.py"], "env": {}}
    config2 = {"command": "python", "args": ["server2.py"], "env": {}}

    lock1 = await registry._get_lock_for_identity(registry._get_identity_key(config1))
    lock2 = await registry._get_lock_for_identity(registry._get_identity_key(config2))

    assert lock1 is not lock2
