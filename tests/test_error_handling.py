"""
Tests for condensed error handling hooks and graph routing decisions.
"""
import pytest
from unittest.mock import AsyncMock

from agentic_core.decisions import (
    DecisionEvent,
    ErrorDecision,
    ErrorContext,
    GraphRoutingDecision,
)
from agentic_core.handlers.standard import SilentHandler, SmartRetryHandler
from agentic_core.handlers.dag import DAGSmartRetryHandler, DAGCascadeOnErrorHandler
from agentic_core.interfaces import (
    ProviderRateLimitError,
    ProviderAuthenticationError,
)


# ===================================================
# Test Condensed ErrorContext
# ===================================================

class TestErrorContext:
    def test_error_context_creation(self):
        error = ValueError("test error")
        context = ErrorContext(
            error=error,
            tool_name="test_tool",
            retry_count=0,
            max_retries=3,
            engine_state={"iteration": 1, "max_iterations": 10}
        )
        
        assert context.error == error
        assert context.tool_name == "test_tool"
        assert context.retry_count == 0
        assert context.max_retries == 3
        assert context.engine_state["iteration"] == 1
    
    def test_error_context_defaults(self):
        error = RuntimeError("runtime error")
        context = ErrorContext(error=error)
        
        assert context.tool_name is None
        assert context.retry_count == 0
        assert context.max_retries == 0
        assert context.engine_state is None


# ===================================================
# Test Condensed ErrorDecision Primitives
# ===================================================

class TestErrorDecision:
    def test_retry_decision_immediate(self):
        decision = ErrorDecision.RETRY()
        assert decision.name == "RETRY"
        assert decision.delay == 0.0
        assert decision.exponential_base == 1.0
    
    def test_retry_decision_with_backoff(self):
        decision = ErrorDecision.RETRY(delay=2.0, exponential_base=2.0)
        assert decision.name == "RETRY"
        assert decision.delay == 2.0
        assert decision.exponential_base == 2.0
    
    def test_skip_decision(self):
        decision = ErrorDecision.SKIP()
        assert decision.name == "SKIP"
    
    def test_abandon_decision(self):
        decision = ErrorDecision.ABANDON()
        assert decision.name == "ABANDON"
    
    def test_resolve_with_decision(self):
        decision = ErrorDecision.RESOLVE_WITH(msg="Context exceeded, please summarize.")
        assert decision.name == "RESOLVE_WITH"
        assert decision.msg == "Context exceeded, please summarize."


# ===================================================
# Test Standard Handlers with Python Native Exceptions
# ===================================================

class TestSilentHandler:
    @pytest.mark.asyncio
    async def test_silent_handler_on_error(self):
        handler = SilentHandler()
        context = ErrorContext(error=ProviderRateLimitError("Rate limited"))
        
        result = await handler.on_error(context)
        
        assert isinstance(result.action, ErrorDecision.ABANDON)


class TestSmartRetryHandler:
    @pytest.mark.asyncio
    async def test_rate_limit_triggers_backoff_retry(self):
        handler = SmartRetryHandler(max_retries=3, base_delay=1.0)
        context = ErrorContext(
            error=ProviderRateLimitError("Rate limited"),
            retry_count=0,
            max_retries=3
        )
        
        result = await handler.on_error(context)
        
        assert isinstance(result.action, ErrorDecision.RETRY)
        assert result.action.delay >= 1.0
        assert result.action.exponential_base > 1.0
    
    @pytest.mark.asyncio
    async def test_auth_error_abandons_immediately(self):
        handler = SmartRetryHandler(max_retries=3)
        context = ErrorContext(error=ProviderAuthenticationError("Invalid key"))
        
        result = await handler.on_error(context)
        
        assert isinstance(result.action, ErrorDecision.ABANDON)
    
    @pytest.mark.asyncio
    async def test_context_limit_error_triggers_retry(self):
        handler = SmartRetryHandler()
        context = ErrorContext(error=RuntimeError("Token limit exceeded"))
        
        result = await handler.on_error(context)
        
        # Simulating a dynamic retry so the engine loop can natively truncate memory
        assert isinstance(result.action, ErrorDecision.RETRY)
    
    @pytest.mark.asyncio
    async def test_max_retries_exceeded_abandons(self):
        handler = SmartRetryHandler(max_retries=2)
        context = ErrorContext(
            error=ProviderRateLimitError("Rate limited"),
            retry_count=2,  # Already at max
            max_retries=2
        )
        
        result = await handler.on_error(context)
        
        assert isinstance(result.action, ErrorDecision.ABANDON)


# ===================================================
# Test GraphRoutingDecision for DAG Engines
# ===================================================

class TestDAGGraphRouting:
    @pytest.mark.asyncio
    async def test_dag_node_permanent_failure_fallback(self):
        handler = DAGSmartRetryHandler(fallback_on_permanent_failure=True)
        
        result = await handler.on_node_permanent_failure(
            "node_1", RuntimeError("Failed permanently")
        )
        
        assert isinstance(result.action, GraphRoutingDecision.IGNORE)
    
    @pytest.mark.asyncio
    async def test_dag_node_permanent_failure_cascade(self):
        handler = DAGSmartRetryHandler(fallback_on_permanent_failure=False)
        
        result = await handler.on_node_permanent_failure(
            "node_1", RuntimeError("Failed permanently")
        )
        
        assert isinstance(result.action, GraphRoutingDecision.CASCADE)


class TestDAGCascadeOnErrorHandler:
    @pytest.mark.asyncio
    async def test_cascade_on_node_failure(self):
        handler = DAGCascadeOnErrorHandler()
        
        result = await handler.on_node_permanent_failure(
            "critical_node", RuntimeError("Critical failure")
        )
        
        assert isinstance(result.action, GraphRoutingDecision.CASCADE)


# ===================================================
# Test Decision Event Wrapper Consistency
# ===================================================

class TestDecisionEventUnion:
    def test_decision_event_with_retry_action(self):
        action = ErrorDecision.RETRY(delay=1.5)
        event = DecisionEvent(action=action)
        
        assert isinstance(event.action, ErrorDecision.RETRY)
        assert event.action.delay == 1.5
    
    def test_decision_event_with_resolve_action(self):
        action = ErrorDecision.RESOLVE_WITH(msg="Recovering...")
        event = DecisionEvent(action=action)
        
        assert isinstance(event.action, ErrorDecision.RESOLVE_WITH)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])