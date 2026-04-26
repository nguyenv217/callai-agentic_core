import asyncio
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from agentic_core.tools.manager import ToolManager

@pytest.mark.asyncio
async def test_tool_manager_async_context_manager():
    """
    Verifies that ToolManager works as an async context manager,
    returning itself on enter and calling shutdown_mcp on exit.
    """
    # Patch shutdown_mcp to verify it's called
    with patch.object(ToolManager, 'shutdown_mcp', new_callable=AsyncMock) as mock_shutdown:
        async with ToolManager() as manager:
            assert manager is not None
            assert isinstance(manager, ToolManager)
            # Shutdown should NOT be called yet
            mock_shutdown.assert_not_awaited()
        
        # After exiting the block, shutdown_mcp must be awaited
        mock_shutdown.assert_awaited_once()

@pytest.mark.asyncio
async def test_tool_manager_async_context_manager_exception():
    """
    Verifies that ToolManager still shuts down MCP servers even if
    an exception occurs inside the context block.
    """
    with patch.object(ToolManager, 'shutdown_mcp', new_callable=AsyncMock) as mock_shutdown:
        try:
            async with ToolManager() as manager:
                raise RuntimeError("Something went wrong")
        except RuntimeError:
            pass
        
        # shutdown_mcp must be called regardless of the exception
        mock_shutdown.assert_awaited_once()

@pytest.mark.asyncio
async def test_tool_manager_cleanup_fallback():
    """
    Verifies that the synchronous cleanup method correctly triggers 
    shutdown_mcp.
    """
    manager = ToolManager()
    manager._mcp_manager = MagicMock() # Ensure cleanup logic is entered

    # We mock shutdown_mcp as an AsyncMock
    with patch.object(manager, 'shutdown_mcp', new_callable=AsyncMock) as mock_shutdown:
        # Case 1: Running loop exists
        with patch("asyncio.get_running_loop") as mock_get_loop:
            mock_loop = MagicMock()
            mock_loop.is_closed.return_value = False
            mock_get_loop.return_value = mock_loop

            manager.cleanup()

            # Verify create_task was called
            mock_loop.create_task.assert_called_once()
            # Verify the argument is a coroutine (the result of calling the mock)
            args, _ = mock_loop.create_task.call_args
            assert asyncio.iscoroutine(args[0])

        # Case 2: No running loop (triggers asyncio.run)
        with patch("asyncio.get_running_loop", side_effect=RuntimeError("No loop")):
            with patch("asyncio.run", new_callable=MagicMock) as mock_run:
                manager.cleanup()

                # Verify asyncio.run was called
                mock_run.assert_called_once()
                # Verify the argument is a coroutine
                args, _ = mock_run.call_args
                assert asyncio.iscoroutine(args[0])

@pytest.mark.asyncio
async def test_tool_manager_atexit_unregistration():
    """
    Verifies that the atexit hook is unregistered when the 
    async context manager is used, avoiding double-cleanup.
    """
    with patch("atexit.unregister") as mock_unregister:
        async with ToolManager() as manager:
            pass
        
        # Verify that atexit.unregister(manager.cleanup) was called
        mock_unregister.assert_called_once_with(manager.cleanup)

