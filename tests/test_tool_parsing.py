import pytest
from unittest.mock import MagicMock
from agentic_core.tools.manager import ToolManager
from agentic_core.tools.base import BaseTool

class DummyTool(BaseTool):
    def __init__(self):
        self._name = "dummy"
        self._schema = {
            "type": "function",
            "function": {
                "name": "dummy",
                "parameters": {"properties": {"data": {"type": "object"}}}
            }
        }
    def execute(self, args, context):
        return args.get("data")

@pytest.mark.asyncio
async def test_tool_manager_json_auto_fix():
    manager = ToolManager()
    manager.register_tool(DummyTool())
    
    # 1. Valid double-serialized JSON should be parsed into a dict automatically
    valid_json_str = '{"nested": "value"}'
    res1 = await manager.execute("dummy", {"data": valid_json_str}, controller=MagicMock())
    assert res1 == "{'nested': 'value'}" # Python dict string representation
    
    # 2. Invalid JSON should fail silently and pass the raw string downstream
    invalid_json_str = '{bad_json: missing_quotes}'
    res2 = await manager.execute("dummy", {"data": invalid_json_str}, controller=MagicMock())
    assert res2 == "{bad_json: missing_quotes}" # Remained a string