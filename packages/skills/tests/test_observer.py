import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from agentic_core.decisions import ErrorContext
from agentic_core.interfaces import AgentResponse
from agentic_core_skills.observer import AutoSkillObserver
from agentic_core.handlers.base import AgentEventHandler

@pytest.mark.asyncio
async def test_auto_skill_observer_triggers_extraction():
    mock_extractor = MagicMock()
    mock_extractor.extract_skill = AsyncMock(return_value=True)
    
    mock_base_handler = AsyncMock(spec=AgentEventHandler)
    observer = AutoSkillObserver(extractor=mock_extractor, error_threshold=2, base_handler=mock_base_handler)
    
    await observer.on_turn_start()
    
    err_ctx1 = ErrorContext(error=ValueError("Error 1"))
    err_ctx2 = ErrorContext(error=ValueError("Error 2"))
    await observer.on_error(err_ctx1)
    await observer.on_error(err_ctx2)
    
    resp = AgentResponse(text="Success")
    await observer.on_turn_complete(resp)
    
    # Allow background asyncio.create_task to execute
    await asyncio.sleep(0.01)
    
    mock_extractor.extract_skill.assert_awaited_once()
    trace = mock_extractor.extract_skill.call_args[0][0]
    assert "Error 1" in trace
    assert "Error 2" in trace
    mock_base_handler.on_turn_complete.assert_awaited_once_with(resp)

@pytest.mark.asyncio
async def test_auto_skill_observer_below_threshold():
    mock_extractor = MagicMock()
    mock_extractor.extract_skill = AsyncMock(return_value=True)
    
    observer = AutoSkillObserver(extractor=mock_extractor, error_threshold=2)
    
    await observer.on_turn_start()
    await observer.on_error(ErrorContext(error=ValueError("Error 1")))
    await observer.on_turn_complete(AgentResponse(text="Success"))
    
    await asyncio.sleep(0.01)
    mock_extractor.extract_skill.assert_not_awaited()

@pytest.mark.asyncio
async def test_auto_skill_observer_fatal_error():
    mock_extractor = MagicMock()
    mock_extractor.extract_skill = AsyncMock(return_value=True)
    
    observer = AutoSkillObserver(extractor=mock_extractor, error_threshold=1)
    
    await observer.on_turn_start()
    await observer.on_error(ErrorContext(error=ValueError("Error 1")))
    
    resp = AgentResponse(text="Failed", error=RuntimeError("Fatal"))
    await observer.on_turn_complete(resp)
    
    await asyncio.sleep(0.01)
    mock_extractor.extract_skill.assert_not_awaited()
