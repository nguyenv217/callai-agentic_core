from __future__ import annotations
import time
import json
import logging
from typing import Any, TYPE_CHECKING
from .dag import DAGEventHandler
from ..models import AgentResponse, DAGResponse
from ..decisions import ErrorContext

if TYPE_CHECKING:
    from ..engines.dag_engine import NodeState

logger = logging.getLogger("agentic_core.telemetry")

class StructuredTelemetryHandler(DAGEventHandler):
    """
    A deterministic handler that emits machine-readable JSON trace events.
    Optimized for log aggregation platforms (Datadog, ELK, Prometheus).
    """
    def __init__(self, trace_id: str | None = None):
        self.trace_id = trace_id or str(time.time())
        self._spans = {}
    
    async def on_node_start(self, node_id: str, worker_id: int):
        self._spans[node_id] = time.time()
        logger.info(json.dumps({
            "trace_id": self.trace_id, 
            "event": "node_start",
            "node_id": node_id, 
            "worker_id": worker_id, 
            "timestamp": self._spans[node_id]
        }))
    
    async def on_node_complete(self, node_id: str, status: NodeState, result: AgentResponse):
        start_time = self._spans.pop(node_id, time.time())
        duration = time.time() - start_time
        usage = result.usage if hasattr(result, 'usage') and result.usage else {}
        logger.info(json.dumps({
            "trace_id": self.trace_id, 
            "event": "node_complete",
            "node_id": node_id, 
            "status": status.name, 
            "duration_s": round(duration, 3),
            "usage": usage
        }))
    
    async def on_error(self, error_context: ErrorContext) -> Any:
        node_id = error_context.engine_state.get("node_id", "unknown") if error_context.engine_state else "unknown"
        logger.error(json.dumps({
            "trace_id": self.trace_id, 
            "event": "error", 
            "node_id": node_id,
            "error_type": error_context.error.__class__.__name__, 
            "message": str(error_context.error)
        }))
        return await super().on_error(error_context)
        
    async def on_graph_complete(self, diagnostics: DAGResponse):
        logger.info(json.dumps({
            "trace_id": self.trace_id, 
            "event": "graph_complete",
            "success": diagnostics.error is None
        }))
