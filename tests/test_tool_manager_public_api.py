import pytest
from unittest.mock import MagicMock, patch
from agentic_core.tools.manager import ToolManager
from agentic_core.tools.base import BaseTool


class MockTool(BaseTool):
    """Mock tool for testing purposes."""
    def __init__(self, name: str = "mock_tool"):
        super().__init__()
        self._name = name
        self._schema = {
            "type": "function",
            "function": {
                "name": name,
                "description": "A mock tool for testing",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }
    
    def execute(self, args: dict, context: dict) -> str:
        return f"Executed {self._name}"


@pytest.mark.asyncio
async def test_get_registered_tools_empty():
    """
    Verifies that get_registered_tools returns empty list when no tools registered.
    """
    manager = ToolManager(enable_mcp_discovery=False)
    tools = manager.get_registered_tools()
    assert tools == []

@pytest.mark.asyncio
async def test_get_registered_tools_with_plugins():
    """
    Verifies that get_registered_tools returns all registered tool names.
    """
    manager = ToolManager(enable_mcp_discovery=False)
    
    # Register some mock tools
    tool1 = MockTool("tool_one")
    tool2 = MockTool("tool_two")
    manager.register_tool(tool1)
    manager.register_tool(tool2)
    
    tools = manager.get_registered_tools()
    
    assert "tool_one" in tools
    assert "tool_two" in tools
    assert len(tools) == 2


@pytest.mark.asyncio
async def test_unregister_tool_success():
    """
    Verifies that unregister_tool successfully removes a tool.
    """
    manager = ToolManager(enable_mcp_discovery=False)
    
    tool = MockTool("test_tool")
    manager.register_tool(tool)
    
    # Verify tool is registered
    assert "test_tool" in manager.get_registered_tools()
    
    # Unregister the tool
    result = manager.unregister_tool("test_tool")
    
    assert result is True
    assert "test_tool" not in manager.get_registered_tools()


@pytest.mark.asyncio
async def test_unregister_tool_not_found():
    """
    Verifies that unregister_tool returns False for non-existent tool.
    """
    manager = ToolManager(enable_mcp_discovery=False)
    
    result = manager.unregister_tool("non_existent_tool")
    
    assert result is False


@pytest.mark.asyncio
async def test_unregister_tool_removes_from_schema():
    """
    Verifies that unregister_tool also removes tool from tools_schema.
    """
    manager = ToolManager(enable_mcp_discovery=False)
    
    tool = MockTool("schema_tool")
    manager.register_tool(tool)
    
    # Verify schema exists
    assert any(s['function']['name'] == "schema_tool" for s in manager.tools_schema)
    
    # Unregister
    manager.unregister_tool("schema_tool")
    
    # Verify schema removed
    assert not any(s['function']['name'] == "schema_tool" for s in manager.tools_schema)


@pytest.mark.asyncio
async def test_unload_mcp_tool_success():
    """
    Verifies that unload_mcp_tool removes MCP tool from loaded set.
    """
    manager = ToolManager(enable_mcp_discovery=False)
    
    # Manually add a tool to _mcp_loaded_tools
    mcp_tool = MockTool("mcp_test_tool")
    manager._mcp_loaded_tools.add(mcp_tool)
    manager.register_tool(mcp_tool, load_mcp=True)
    
    # Verify tool is in loaded set
    assert any(t.name == "mcp_test_tool" for t in manager._mcp_loaded_tools)
    
    # Unload the MCP tool
    result = manager.unload_mcp_tool("mcp_test_tool")
    
    assert result is True
    assert not any(t.name == "mcp_test_tool" for t in manager._mcp_loaded_tools)


@pytest.mark.asyncio
async def test_unload_mcp_tool_not_found():
    """
    Verifies that unload_mcp_tool returns False for non-existent tool.
    """
    manager = ToolManager(enable_mcp_discovery=False)
    
    result = manager.unload_mcp_tool("non_existent_mcp_tool")
    
    assert result is False


@pytest.mark.asyncio
async def test_unload_mcp_tool_removes_from_plugins():
    """
    Verifies that unload_mcp_tool also removes from plugins if present.
    """
    manager = ToolManager(enable_mcp_discovery=False)
    
    mcp_tool = MockTool("mcp_plugin_tool")
    manager._mcp_loaded_tools.add(mcp_tool)
    manager.register_tool(mcp_tool, load_mcp=True)
    
    # Verify tool is in plugins
    assert "mcp_plugin_tool" in manager._plugins
    
    # Unload
    result = manager.unload_mcp_tool("mcp_plugin_tool")
    
    assert result is True
    assert "mcp_plugin_tool" not in manager._plugins


@pytest.mark.asyncio
async def test_get_mcp_loaded_tools():
    """
    Verifies that get_mcp_loaded_tools returns schemas of loaded MCP tools.
    """
    manager = ToolManager(enable_mcp_discovery=False)
    
    mcp_tool1 = MockTool("loaded_mcp_1")
    mcp_tool2 = MockTool("loaded_mcp_2")
    
    manager._mcp_loaded_tools.add(mcp_tool1)
    manager._mcp_loaded_tools.add(mcp_tool2)
    
    loaded = manager.get_mcp_loaded_tools()
    
    assert len(loaded) == 2
    tool_names = [t['function']['name'] for t in loaded]
    assert "loaded_mcp_1" in tool_names
    assert "loaded_mcp_2" in tool_names


@pytest.mark.asyncio
async def test_clear_loaded_tools():
    """
    Verifies that clear_loaded_tools clears all loaded MCP tools.
    """
    manager = ToolManager(enable_mcp_discovery=False)
    
    mcp_tool = MockTool("to_clear")
    manager._mcp_loaded_tools.add(mcp_tool)
    
    assert len(manager._mcp_loaded_tools) > 0
    
    manager.clear_loaded_tools()
    
    assert len(manager._mcp_loaded_tools) == 0
