import json
import os
import asyncio
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from agentic_core.tools.manager import ToolManager
from agentic_core.config import RunnerConfig
from agentic_core.tools.mcp.manager import MCPClientManager, GlobalMCPRegistry

@pytest.fixture(autouse=True)
def clear_mcp_registry():
    """Ensure GlobalMCPRegistry is cleared before and after each test."""
    registry = GlobalMCPRegistry()
    registry._sessions.clear()
    yield
    registry._sessions.clear()

@pytest.fixture
def mock_mcp_manager():
    """Mocks the MCP initialization to prevent actual subprocess spawning."""
    with patch("agentic_core.tools.mcp.manager.MCPClientManager") as mock_cls:
        instance = mock_cls.return_value
        instance.initialize = AsyncMock(return_value=True)
        # Mock the tools returned by the server
        instance.list_all_tools = AsyncMock(return_value=[
            {
                "server_name": "mock_github",
                "session": AsyncMock(),
                "name": "create_issue",
                "description": "Creates a github issue",
                "inputSchema": {"type": "object", "properties": {"title": {"type": "string"}}}
            }
        ])
        instance.close = AsyncMock()
        yield instance

@pytest.mark.asyncio
async def test_tool_manager_lazy_mcp_init(mock_mcp_manager):
    """
    Verifies that ToolManager defers MCP initialization until prepare_turn is called
    with a config that explicitly requires MCP tools.
    """
    # 1. Initialize manager with a config path (triggers MCP readiness, but shouldn't boot yet)
    manager = ToolManager(mcp_config_path="dummy_config.json")

    assert len(manager._mcp_standby_registry) == 0

    # 2. Prepare turn with an explicit MCP preload request
    config = RunnerConfig(mcp_preload_tools=["mock_github_create_issue"], mcp_active_servers=["mock_github"])
    await manager.prepare_turn(config)

    assert len(manager._mcp_standby_registry) > 0

    mock_mcp_manager.initialize.assert_awaited_once()
    mock_mcp_manager.list_all_tools.assert_awaited_once()

    # Verify the tool was moved to standby AND actively loaded
    assert "mock_github_create_issue" in manager._mcp_standby_registry

    active_loaded_schemas = manager.get_mcp_loaded_tools()
    active_tool_names = [t['function']['name'] for t in active_loaded_schemas]
    assert "mock_github_create_issue" in active_tool_names

@pytest.mark.asyncio
async def test_load_mcp_tool_execution(mock_mcp_manager):
    """
    Verifies the universal 'load_mcp_tool' can dynamically move a tool
    from the standby registry into the active execution context.
    """
    manager = ToolManager(mcp_config_path="dummy_config.json")
    config = RunnerConfig(mcp_active_servers=["mock_github"], mcp_enable_discovery=True)

    # Eagerly initialize to populate the standby registry
    await manager.prepare_turn(config)

    # Ensure it is in standby, but NOT in loaded tools yet
    assert "mock_github_create_issue" in manager._mcp_standby_registry
    assert manager._mcp_loaded_tools is not None
    loaded_names = [t.name for t in manager._mcp_loaded_tools]
    assert "mock_github_create_issue" not in loaded_names

    # Mock the controller context
    mock_controller = MagicMock()

    # Execute the universal load_mcp_tool meta-tool
    args = {"tool_names": ["mock_github_create_issue"]}
    result = await manager.execute("load_mcp_tool", args, mock_controller)

    # Verify execution result and state change
    assert "Success: Loaded 1 tool(s)" in result
    loaded_names = [t.name for t in manager._mcp_loaded_tools]
    assert "mock_github_create_issue" in loaded_names

# We will start the mock server as a subprocess in a fixture
import subprocess

@pytest.fixture(scope="module")
def mcp_server_process():
    # Path to the mock server we created
    server_path = os.path.abspath("tests/mock_mcp_server.py")

    # Start the server using python.
    # FastMCP.run() by default uses stdio if no transport is specified.
    process = subprocess.Popen(
        ["python3", server_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    yield process
    process.terminate()

@pytest.fixture
def mcp_config_file(tmp_path):
    config_dir = tmp_path / "mcp"
    config_dir.mkdir()
    config_file = config_dir / "config.json"

    # We need to tell the MCPClientManager how to start the server.
    # Since we are testing the logic of MCPClientManager, we point it
    # to the script we just wrote.
    config = {
        "mcpServers": {
            "test_server": {
                "command": "python3",
                "args": [os.path.abspath("tests/mock_mcp_server.py")]
            }
        }
    }
    config_file.write_text(json.dumps(config))
    return config_file

@pytest.mark.asyncio
async def test_mcp_integration_flow(mcp_config_file):
    # 1. Setup ToolManager with the temp config
    manager = ToolManager(mcp_config_path=str(mcp_config_file))

    # 2. Prepare turn to initialize MCP servers
    # We want to active 'test_server'
    config = RunnerConfig(mcp_active_servers=["test_server"], mcp_enable_discovery=True)
    await manager.prepare_turn(config)

    # 3. Verify tools were discovered and put in standby registry
    # Expected tools from mock_mcp_server.py: echo, add
    # Prefixed names: test_server_echo, test_server_add
    standby = manager._mcp_standby_registry
    assert "test_server_echo" in standby
    assert "test_server_add" in standby

    # 4. Test the 'list_mcp_catalog' meta-tool
    result = await manager.execute("list_mcp_catalog", {"server_name": "test_server"}, {})
    assert "test_server_echo" in result
    assert "test_server_add" in result

    # 5. Test 'load_mcp_tool' meta-tool
    load_result = await manager.execute("load_mcp_tool", {"tool_names": ["test_server_echo"]})
    assert "Success: Loaded 1 tool(s)" in load_result

    # 6. Verify tool is now in loaded tools
    loaded_names = [t.name for t in manager._mcp_loaded_tools]
    assert "test_server_echo" in loaded_names

    # 7. Actually EXECUTE the loaded MCP tool
    execution_result = await manager.execute("test_server_echo", {"text": "Hello MCP!"}, {})
    assert "Echo: Hello MCP!" in execution_result

@pytest.mark.asyncio
async def test_mcp_invalid_server(tmp_path):
    config_file = tmp_path / "invalid_config.json"
    config = {
        "mcpServers": {
            "bad_server": {
                "command": "non_existent_command_12345",
                "args": []
            }
        }
    }
    config_file.write_text(json.dumps(config))

    manager = ToolManager(mcp_config_path=str(config_file))
    config = RunnerConfig(mcp_active_servers=["bad_server"], mcp_enable_discovery=True)

    # This should not crash the whole system but log error and return False
    success = await manager.prepare_turn(config)
    # Since prepare_turn is just a wrapper, we check the manager state
    assert len(manager._mcp_standby_registry) == 0

@pytest.mark.asyncio
async def test_mcp_session_sharing():
    """Tests that two managers with identical config share the same session."""

    config = {
        "mcpServers": {
            "test_server": {
                "command": "python",
                "args": ["-c", "print('hello')"],
                "env": {"TEST_VAR": "value"}
            }
        }
    }

    with patch("agentic_core.tools.mcp.manager.stdio_client") as mock_stdio, \
         patch("agentic_core.tools.mcp.manager.ClientSession") as mock_session, \
         patch("shutil.which", return_value="/usr/bin/python"):
        
        mock_stdio.return_value.__aenter__.return_value = (MagicMock(), MagicMock())
        mock_session_instance = mock_session.return_value.__aenter__.return_value
        
        async def mock_init(): await asyncio.sleep(0)
        mock_session_instance.initialize = mock_init
        
        async def mock_list(): return MagicMock(tools=[])
        mock_session_instance.list_tools = mock_list
        
        async def mock_call(n, arguments): return MagicMock(content=[])
        mock_session_instance.call_tool = mock_call

        manager1 = MCPClientManager(config=config)
        manager2 = MCPClientManager(config=config)

        await manager1.initialize()
        await manager2.initialize()

        assert manager1.sessions[0]['session'] == manager2.sessions[0]['session']
        
        registry = GlobalMCPRegistry()
        identity_key = registry._get_identity_key(config["mcpServers"]["test_server"])
        assert identity_key in registry._sessions
        assert registry._sessions[identity_key]['ref_count'] == 2

        await manager1.close()
        assert registry._sessions[identity_key]['ref_count'] == 1

        await manager2.close()
        assert identity_key not in registry._sessions

@pytest.mark.asyncio
async def test_mcp_session_isolation():
    """Tests that different configurations result in different sessions."""

    config1 = {
        "mcpServers": {
            "server1": {
                "command": "python",
                "args": ["--arg1"],
            }
        }
    }
    config2 = {
        "mcpServers": {
            "server2": {
                "command": "python",
                "args": ["--arg2"],
            }
        }
    }

    with patch("agentic_core.tools.mcp.manager.stdio_client") as mock_stdio, \
        patch("agentic_core.tools.mcp.manager.ClientSession") as mock_session, \
        patch("shutil.which", return_value="/usr/bin/python"):

        mock_stdio.return_value.__aenter__.return_value = (MagicMock(), MagicMock())
        mock_session_instance = mock_session.return_value.__aenter__.return_value

        async def mock_init(): await asyncio.sleep(0)
        mock_session_instance.initialize = mock_init

        async def mock_list(): return MagicMock(tools=[])
        mock_session_instance.list_tools = mock_list

        async def mock_call(n, arguments): return MagicMock(content=[])
        mock_session_instance.call_tool = mock_call

        manager1 = MCPClientManager(config=config1)
        manager2 = MCPClientManager(config=config2)

        await manager1.initialize()
        await manager2.initialize()

        assert manager1.sessions[0]['session'] != manager2.sessions[0]['session']

        registry = GlobalMCPRegistry()
        assert len(registry._sessions) == 2

        await manager1.close()
        await manager2.close()
        assert len(registry._sessions) == 0

@pytest.mark.asyncio
async def test_mcp_different_env_isolation():
    """Tests that same command but different env results in different sessions."""

    config1 = {
        "mcpServers": {
            "server1": {
                "command": "python",
                "env": {"VAR": "A"}
            }
        }
    }
    config2 = {
        "mcpServers": {
            "server2": {
                "command": "python",
                "env": {"VAR": "B"}
            }
        }
    }

    with patch("agentic_core.tools.mcp.manager.stdio_client") as mock_stdio, \
         patch("agentic_core.tools.mcp.manager.ClientSession") as mock_session, \
         patch("shutil.which", return_value="/usr/bin/python"):

        mock_stdio.return_value.__aenter__.return_value = (MagicMock(), MagicMock())
        mock_session_instance = mock_session.return_value.__aenter__.return_value

        async def mock_init(): await asyncio.sleep(0)
        mock_session_instance.initialize = mock_init

        async def mock_list(): return MagicMock(tools=[])
        mock_session_instance.list_tools = mock_list

        async def mock_call(n, arguments): return MagicMock(content=[])
        mock_session_instance.call_tool = mock_call

        manager1 = MCPClientManager(config=config1)
        manager2 = MCPClientManager(config=config2)

        await manager1.initialize()
        await manager2.initialize()

        assert manager1.sessions[0]['session'] != manager2.sessions[0]['session']

        registry = GlobalMCPRegistry()
        assert len(registry._sessions) == 2

        await manager1.close()
        await manager2.close()
        assert len(registry._sessions) == 0


@pytest.mark.asyncio
async def test_mcp_disconnect_specific_server():
    """Tests disconnecting a specific server while keeping others connected."""

    config = {
        "mcpServers": {
            "server1": {
                "command": "python",
                "args": ["--server1"],
            },
            "server2": {
                "command": "python",
                "args": ["--server2"],
            }
        }
    }

    with patch("agentic_core.tools.mcp.manager.stdio_client") as mock_stdio, \
         patch("agentic_core.tools.mcp.manager.ClientSession") as mock_session, \
         patch("shutil.which", return_value="/usr/bin/python"):

        mock_stdio.return_value.__aenter__.return_value = (MagicMock(), MagicMock())
        mock_session_instance = mock_session.return_value.__aenter__.return_value

        async def mock_init(): await asyncio.sleep(0)
        mock_session_instance.initialize = mock_init

        async def mock_list(): return MagicMock(tools=[])
        mock_session_instance.list_tools = mock_list

        async def mock_call(n, arguments): return MagicMock(content=[])
        mock_session_instance.call_tool = mock_call

        manager = MCPClientManager(config=config)
        await manager.initialize()

        # Both servers should be connected
        assert len(manager.sessions) == 2
        active = manager.get_active_servers()
        assert "server1" in active
        assert "server2" in active

        # Disconnect only server1
        await manager.disconnect(["server1"])

        # Only server2 should remain
        assert len(manager.sessions) == 1
        active = manager.get_active_servers()
        assert "server1" not in active
        assert "server2" in active

        await manager.close()


@pytest.mark.asyncio
async def test_mcp_disconnect_all_servers():
    """Tests disconnecting all servers using disconnect(None)."""

    config = {
        "mcpServers": {
            "server1": {
                "command": "python",
                "args": ["--server1"],
            }
        }
    }

    with patch("agentic_core.tools.mcp.manager.stdio_client") as mock_stdio, \
         patch("agentic_core.tools.mcp.manager.ClientSession") as mock_session, \
         patch("shutil.which", return_value="/usr/bin/python"):

        mock_stdio.return_value.__aenter__.return_value = (MagicMock(), MagicMock())
        mock_session_instance = mock_session.return_value.__aenter__.return_value

        async def mock_init(): await asyncio.sleep(0)
        mock_session_instance.initialize = mock_init

        async def mock_list(): return MagicMock(tools=[])
        mock_session_instance.list_tools = mock_list

        async def mock_call(n, arguments): return MagicMock(content=[])
        mock_session_instance.call_tool = mock_call

        manager = MCPClientManager(config=config)
        await manager.initialize()

        # Server should be connected
        assert len(manager.sessions) == 1
        assert len(manager.get_active_servers()) == 1

        # Disconnect all (None)
        await manager.disconnect()

        # All should be disconnected
        assert len(manager.sessions) == 0
        assert len(manager.get_active_servers()) == 0


@pytest.mark.asyncio
async def test_tool_manager_disconnect_mcp_all():
    """Tests ToolManager.disconnect_mcp() disconnects all servers."""

    # Create a mock MCPClientManager
    mock_mcp_mgr = MagicMock()
    mock_mcp_mgr.initialize = AsyncMock(return_value=True)
    mock_mcp_mgr.list_all_tools = AsyncMock(return_value=[])
    mock_mcp_mgr.get_active_servers = MagicMock(return_value=["test_server"])
    mock_mcp_mgr.disconnect = AsyncMock()

    # Patch where it's imported from
    with patch("agentic_core.tools.mcp.manager.MCPClientManager", return_value=mock_mcp_mgr, create=True):
        manager = ToolManager()
        # Add MCP server programmatically (instead of using invalid mcp_config_dict parameter)
        manager.add_mcp_server("test_server", "python", args=["--test"])
        
        # Initialize MCP
        await manager.initialize_mcp()
        
        # Verify servers are active
        assert len(manager.get_active_servers()) > 0
        
        # Disconnect all
        await manager.disconnect_mcp()
        
        # Should have called disconnect on mock
        mock_mcp_mgr.disconnect.assert_called_once()
        
        # Verify get_active_servers returns empty
        mock_mcp_mgr.get_active_servers.return_value = []
        assert len(manager.get_active_servers()) == 0


@pytest.mark.asyncio
async def test_tool_manager_disconnect_mcp_specific():
    """Tests ToolManager.disconnect_mcp() disconnects specific servers."""

    # Create a mock MCPClientManager
    mock_mcp_mgr = MagicMock()
    mock_mcp_mgr.initialize = AsyncMock(return_value=True)
    mock_mcp_mgr.list_all_tools = AsyncMock(return_value=[])
    mock_mcp_mgr.get_active_servers = MagicMock(side_effect=[
        ["server1", "server2"],
        ["server2"]
    ])
    mock_mcp_mgr.disconnect = AsyncMock()

    # Patch where it's imported from
    with patch("agentic_core.tools.mcp.manager.MCPClientManager", return_value=mock_mcp_mgr, create=True):
        manager = ToolManager()
        # Add MCP servers programmatically
        manager.add_mcp_server("server1", "python", args=["--s1"])
        manager.add_mcp_server("server2", "python", args=["--s2"])
        
        # Initialize MCP
        await manager.initialize_mcp()
        
        # Both servers should be active
        active = manager.get_active_servers()
        assert "server1" in active
        assert "server2" in active
        
        # Disconnect only server1
        await manager.disconnect_mcp(["server1"])
        
        # Should have called disconnect with server1
        mock_mcp_mgr.disconnect.assert_called_once_with(["server1"])
        
        # Only server2 should remain
        remaining = manager.get_active_servers()
        assert "server2" in remaining

