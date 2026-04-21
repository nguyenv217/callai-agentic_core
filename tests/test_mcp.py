import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from agentic_core.tools.manager import ToolManager
from agentic_core.engine import RunnerConfig

@pytest.fixture
def mock_mcp_manager():
    """Mocks the MCP initialization to prevent actual subprocess spawning."""
    with patch("agentic_core.tools.mcp.MCPClientManager") as mock_cls:
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