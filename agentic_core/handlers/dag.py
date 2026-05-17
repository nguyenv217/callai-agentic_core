"""DAG handler implementations.

This module exists in core for backward compatibility.
The canonical implementations live in the optional `callai-agentic_core-handlers` package,
but core keeps thin copies so that importing `agentic_core.handlers.dag` works
without installing any extras.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import logging

from .base import AgentEventHandler
from ..decisions import (
    DecisionEvent,
    GraphRoutingAction,
    GraphRoutingDecision,
    ErrorContext,
    ErrorAction,
    ErrorDecision,
)

if TYPE_CHECKING:
    from ..interfaces import AgentResponse, DAGResponse
    from ..engines.dag_engine import NodeState

logger = logging.getLogger(__name__)


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
        logger.warning(
            f"[DAG] Error in node: {error_context.error.__class__.__name__} - {str(error_context.error)}"
        )
        return DecisionEvent(action=ErrorDecision.ABANDON())

    async def on_node_permanent_failure(self, node_id: str, error: Exception) -> DecisionEvent[GraphRoutingAction]:
        return DecisionEvent(action=GraphRoutingDecision.CASCADE())


