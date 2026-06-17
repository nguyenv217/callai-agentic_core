"""
DAG handler implementations.
"""
from __future__ import annotations 
from typing import TYPE_CHECKING, Callable, Optional
import logging

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..models import AgentResponse, DAGResponse
    from ..engines.dag_engine import NodeState
    
from .base import AgentEventHandler
from ..decisions import DecisionEvent, GraphRoutingAction, GraphRoutingDecision, ErrorContext, ErrorAction, ErrorDecision
from ..exceptions import ProviderRateLimitError, ProviderTimeoutError

class DAGEventHandler(AgentEventHandler):
    """Base handler for DAG execution events."""
    
    async def on_node_queued(self, node_id: str, priority: int):
        logger.info(f"[DAG] Node {node_id} queued with priority {priority}")
    
    async def on_node_start(self, node_id: str, worker_id: int):
        logger.info(f"[DAG] Worker {worker_id} starting node {node_id}")
    
    async def on_node_complete(self, node_id: str, status: NodeState, result: AgentResponse):
        logger.info(f"[DAG] Node {node_id} completed with status {status}")
    
    async def on_node_retry(self, node_id: str, retry_count: int, max_retries: int):
        logger.info(f"[DAG] Node {node_id} failed. Retrying ({retry_count}/{max_retries})...")
    
    async def on_graph_complete(self, diagnostics: DAGResponse):
        logger.info(f"[DAG] Graph execution complete. Diagnostics: {diagnostics.to_dict()}")
    
    async def on_error(self, error_context: ErrorContext) -> DecisionEvent[ErrorAction]:
        """Handle errors during DAG execution."""
        logger.warning(f"[DAG] Error in node: {error_context.error.__class__.__name__} - {str(error_context.error)}")
        return DecisionEvent(action=ErrorDecision.ABANDON())
    
    async def on_node_permanent_failure(self, node_id: str, error: Exception) -> DecisionEvent[GraphRoutingAction]:
        return DecisionEvent(action=GraphRoutingDecision.CASCADE())


class DAGSmartRetryHandler(DAGEventHandler):
    """
    DAG handler with intelligent retry and failure handling.
    """
    def __init__(
        self,
        max_retries_per_node: int = 3,
        base_backoff_delay: float = 1.0,
        max_backoff_delay: float = 60.0,
        fallback_on_permanent_failure: bool = True,
        error_filter: Optional[Callable[[ErrorContext], bool]] = None
    ):
        self.max_retries_per_node = max_retries_per_node
        self.base_backoff_delay = base_backoff_delay
        self.max_backoff_delay = max_backoff_delay
        self.fallback_on_permanent_failure = fallback_on_permanent_failure
        self.error_filter = error_filter
    
    async def on_error(self, error_context: ErrorContext) -> DecisionEvent[ErrorAction]:
        # Let the DAG engine handle all retries. Force AgentRunner's internal loop to abort.
        if "node_id" not in (error_context.engine_state or {}):
            return DecisionEvent(action=ErrorDecision.ABANDON())

        if self.error_filter and not self.error_filter(error_context):
            return DecisionEvent(action=ErrorDecision.SKIP())
        
        error = error_context.error
        retry_count = error_context.retry_count
        max_retries = error_context.max_retries or self.max_retries_per_node
        
        error_msg = str(error).lower()
        is_transient = (
            isinstance(error, (ProviderRateLimitError, ProviderTimeoutError, ConnectionError)) or
            any(kw in error_msg for kw in ["rate limit", "timeout", "network", "connection"])
        )
        
        if is_transient:
            if retry_count < max_retries:
                return DecisionEvent(action=ErrorDecision.RETRY(
                    delay=self.base_backoff_delay,
                    exponential_base=2.0
                ))
            return DecisionEvent(action=ErrorDecision.ABANDON())
            
        elif error_context.tool_name is not None:
            logger.warning(f"[DAG] Tool {error_context.tool_name} failed, skipping")
            return DecisionEvent(action=ErrorDecision.SKIP())
            
        else:
            if retry_count < 1:
                return DecisionEvent(action=ErrorDecision.RETRY(delay=0.0))
            return DecisionEvent(action=ErrorDecision.ABANDON())
    
    async def on_node_permanent_failure(self, node_id: str, error: Exception) -> DecisionEvent[GraphRoutingAction]:
        """Handle permanent node failures."""
        if self.fallback_on_permanent_failure:
            logger.info(f"[DAG] Node {node_id} failed permanently, ignoring to allow downstream execution.")
            return DecisionEvent(action=GraphRoutingDecision.IGNORE())
        return DecisionEvent(action=GraphRoutingDecision.CASCADE())


class DAGCascadeOnErrorHandler(DAGEventHandler):
    """
    DAG handler that cascades errors to downstream nodes.
    """
    async def on_error(self, error_context: ErrorContext) -> DecisionEvent[ErrorAction]:
        logger.error(f"[DAG CASCADE] Error detected: {error_context.error.__class__.__name__}")
        return DecisionEvent(action=ErrorDecision.ABANDON())
    
    async def on_node_permanent_failure(self, node_id: str, error: Exception) -> DecisionEvent[GraphRoutingAction]:
        return DecisionEvent(action=GraphRoutingDecision.CASCADE())