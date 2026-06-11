import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from agentic_core.decisions import ErrorContext
from agentic_core.interfaces import AgentResponse
from agentic_core_skills.observer import AutoSkillObserver
from agentic_core.handlers.base import AgentEventHandler

@pytest.mark.asyncio
async def test_auto_skill_observer_accumulates_errors():
    mock_extractor = MagicMock()
    mock_base_handler = AsyncMock(spec=AgentEventHandler)
    observer = AutoSkillObserver(extractor=mock_extractor, error_threshold=2, base_handler=mock_base_handler)
    
    await observer.on_tool_complete("test_tool", "1", False, "Failed")
    await observer.on_error(ErrorContext(error=ValueError("Error 2")))
    
    assert observer._session_error_count == 2
    
    await observer.on_tool_complete("announce_finish", "2", True, "Done")
    assert observer._task_completed is True
