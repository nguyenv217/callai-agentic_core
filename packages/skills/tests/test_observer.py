import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from agentic_core.decisions import ErrorContext
from agentic_core.models import AgentResponse
from agentic_core_skills.observer import AutoSkillObserver
from agentic_core.handlers.base import AgentEventHandler

@pytest.mark.asyncio
async def test_auto_skill_observer_accumulates_errors():
    mock_extractor = MagicMock()
    mock_base_handler = AsyncMock(spec=AgentEventHandler)
    observer = AutoSkillObserver(extractor=mock_extractor, error_threshold=2, tool_call_threshold=6, base_handler=mock_base_handler)
    
    await observer.on_tool_complete("test_tool", "1", False, "Failed")
    await observer.on_error(ErrorContext(error=ValueError("Error 2")))
    
    assert observer._session_error_count == 2
    
    await observer.on_tool_complete("announce_finish", "2", True, "Done")
    assert observer._task_completed is True

@pytest.mark.asyncio
async def test_auto_skill_observer_accumulates_tool_calls():
    mock_extractor = MagicMock()
    observer = AutoSkillObserver(extractor=mock_extractor, error_threshold=3, tool_call_threshold=2)
    
    await observer.on_tool_start("test_tool", "1", "{}")
    await observer.on_tool_start("test_tool_2", "2", "{}")
    
    assert observer._session_tool_call_count == 2
