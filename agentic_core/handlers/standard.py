"""
Standard handler implementations.
"""
import asyncio
import inspect
import logging

from .base import AgentEventHandler
from ..decisions import ToolStartAction, ToolStartDecision, DecisionEvent, ErrorContext, ErrorAction, ErrorDecision
from ..tools.protocols import ToolExecutionController
from ..interfaces import (
    ProviderRateLimitError, 
    ProviderAuthenticationError, 
    ProviderTimeoutError,
    IterationLimitReachedError
)

logger = logging.getLogger(__name__)

class SilentHandler(AgentEventHandler):
    """A no-op handler that prints nothing. Use this if you don't care about events."""
    
    async def on_turn_start(self) -> None: pass
    async def on_iteration_start(self, iteration: int, max_iterations: int) -> None: pass
    async def on_llm_progress(self, info: str) -> None: pass
    async def on_tool_start(self, tool_name, tool_id, tool_args): return DecisionEvent(ToolStartDecision.CONTINUE())
    async def on_tool_complete(self, tool_name: str, tool_id: str, success: bool, result: str) -> None: pass
    async def on_turn_complete(self, response: dict) -> None: pass
    async def on_error(self, error_context: ErrorContext) -> DecisionEvent[ErrorAction]: 
        return DecisionEvent(action=ErrorDecision.ABANDON())


class PrintHandler(AgentEventHandler, ToolExecutionController):
    """A handler that prints everything - great for debugging."""
    
    async def on_turn_start(self) -> None:
        print("📍 [TURN START]")
    
    async def on_iteration_start(self, iteration: int, max_iterations: int) -> None:
        print(f"🔄 [ITERATION {iteration}/{max_iterations}]")
    
    async def on_llm_progress(self, info: str) -> None:
        print(f"💬 [LLM]: {info[:200]}")

    async def on_tool_call_session_start(self, reasoning_text, tool_calls, iteration, max_iterations):
        print(f"💡 [ITERATION {iteration}/{max_iterations}]: {reasoning_text[:500]}")
    
    async def on_tool_start(self, tool_name: str, tool_id: str, tool_args: str | dict | None) -> DecisionEvent[ToolStartAction]:
        print(f"🔧 [TOOL START]: {tool_name}")
        return DecisionEvent(ToolStartDecision.CONTINUE())
    
    async def on_tool_complete(self, tool_name: str, tool_id: str, success: bool, result: str) -> None:
        status = "✅" if success else "❌"
        print(f"{status} [TOOL COMPLETE]: {tool_name} -> {str(result)[:500]}")
    
    async def on_turn_complete(self, response: dict) -> None:
        print(f"🏁 [TURN COMPLETE]")
    
    async def on_error(self, error_context: ErrorContext) -> DecisionEvent[ErrorAction]:
        """Handle errors with contextual decision making."""
        error = error_context.error
        
        ctx_parts = []
        if error_context.tool_name:
            ctx_parts.append(f"tool={error_context.tool_name}")
        if error_context.retry_count > 0:
            ctx_parts.append(f"retry={error_context.retry_count}/{error_context.max_retries}")
        
        ctx_str = f" ({', '.join(ctx_parts)})" if ctx_parts else ""
        print(f"❗ [ERROR{ctx_str}]: {error.__class__.__name__} - {str(error)}")
        
        return DecisionEvent(action=ErrorDecision.ABANDON())

    async def on_prompt_respond(self, prompt: str) -> str:
        return await asyncio.to_thread(input, prompt)
    
    async def on_prompt_confirmation(self, prompt, on_yes, on_no):
        response = await asyncio.to_thread(input, prompt)
        if response.strip().lower() in ['y', 'yes']: 
            if inspect.iscoroutinefunction(on_yes): await on_yes()
            else: on_yes()
        elif on_no: 
            if inspect.iscoroutinefunction(on_no): await on_no()
            else: on_no()


class SmartRetryHandler(AgentEventHandler):
    """
    A handler that implements intelligent retry logic based on native Python exceptions.
    """
    def __init__(
        self, 
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        on_retry_callback=None
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.on_retry_callback = on_retry_callback
    
    async def on_error(self, error_context: ErrorContext) -> DecisionEvent[ErrorAction]:
        error = error_context.error
        retry_count = error_context.retry_count
        active_max_retries = error_context.max_retries if error_context.max_retries > 0 else self.max_retries
        
        # Transient API Errors - Retry with backoff
        if isinstance(error, (ProviderRateLimitError, ProviderTimeoutError, ConnectionError)):
            if retry_count < active_max_retries:
                if self.on_retry_callback:
                    await self.on_retry_callback(error_context, retry_count, self.base_delay)
                return DecisionEvent(action=ErrorDecision.RETRY(
                    delay=self.base_delay,
                    exponential_base=2.0
                ))
            return DecisionEvent(action=ErrorDecision.ABANDON())
            
        # Authentication or Unrecoverable - Abort immediately
        elif isinstance(error, ProviderAuthenticationError):
            return DecisionEvent(action=ErrorDecision.ABANDON())
            
        # Limits Reached - Resolve by injecting an escalation message
        elif isinstance(error, IterationLimitReachedError):
            return DecisionEvent(action=ErrorDecision.RESOLVE_WITH(
                msg=f"Iteration limit reached: {str(error)}. Please finalize your response."
            ))
            
        # Tool execution errors - Skip to keep the loop alive
        elif error_context.tool_name is not None:
            return DecisionEvent(action=ErrorDecision.SKIP())
            
        # Context limits or other fatal provider limits
        elif "limit" in str(error).lower() or "context" in str(error).lower():
            if retry_count < 1:
                return DecisionEvent(action=ErrorDecision.RESOLVE_WITH(
                    msg=f"System error: {str(error)}. Please synthesize immediately."
                ))
            return DecisionEvent(action=ErrorDecision.ABANDON())
            
        # Unknown errors - One quick retry, then abandon
        else:
            if retry_count < 1:
                return DecisionEvent(action=ErrorDecision.RETRY(delay=0.0))
            return DecisionEvent(action=ErrorDecision.ABANDON())