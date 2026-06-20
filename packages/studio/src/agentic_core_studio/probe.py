import json
import socket
from agentic_core.handlers.dag import DAGSmartRetryHandler

class StudioProbe(DAGSmartRetryHandler):
    """
    Telemetry probe that attaches to a GraphAgentRunner and broadcasts 
    non-blocking UDP telemetry to the standalone Studio Analyzer.
    
    Usage: 
        graph = GraphAgentRunner(..., handler=StudioProbe())
    """
    def __init__(self, host="127.0.0.1", port=9876, **kwargs):
        super().__init__(**kwargs)
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)

    def _broadcast(self, event_type: str, payload: dict):
        try:
            data = json.dumps({"type": event_type, "payload": payload}).encode('utf-8')
            if len(data) < 60000:  # Stay safely under UDP packet limits
                self.sock.sendto(data, (self.host, self.port))
        except Exception:
            pass  # Silently drop telemetry if analyzer is offline

    async def on_node_queued(self, node_id, priority):
        self._broadcast("node_queued", {"node_id": node_id, "priority": priority})
        return await super().on_node_queued(node_id, priority)

    async def on_node_start(self, node_id, worker_id):
        self._broadcast("node_start", {"node_id": node_id, "worker_id": worker_id})
        return await super().on_node_start(node_id, worker_id)

    async def on_node_complete(self, node_id, status, result):
        snippet = ""
        if result and hasattr(result, 'text') and result.text:
            snippet = result.text[:200] + ("..." if len(result.text) > 200 else "")
        elif isinstance(result, str):
            snippet = result[:200] + ("..." if len(result) > 200 else "")
        self._broadcast("node_complete", {"node_id": node_id, "status": status.name, "snippet": snippet})
        return await super().on_node_complete(node_id, status, result)

    async def on_error(self, error_context):
        decision = await super().on_error(error_context)
        node_id = error_context.engine_state.get("node_id", "Unknown") if error_context.engine_state else "Unknown"
        self._broadcast("error", {"node_id": node_id, "error": str(error_context.error), "action": decision.action.name})
        return decision

    async def on_tool_start(self, tool_name, tool_id, tool_args):
        self._broadcast("tool_start", {"tool_name": tool_name})
        return await super().on_tool_start(tool_name, tool_id, tool_args)

    async def on_graph_complete(self, diagnostics):
        self._broadcast("graph_complete", {})
        return await super().on_graph_complete(diagnostics)
