import json
import asyncio
from agentic_core.handlers.dag import DAGSmartRetryHandler

class StudioProbe(DAGSmartRetryHandler):
    """
    Telemetry probe that attaches to a GraphAgentRunner and broadcasts 
    non-blocking TCP JSONlines telemetry to the standalone Studio Analyzer.
    
    Usage: 
        graph = GraphAgentRunner(..., handler=StudioProbe())
    """
    def __init__(self, host="127.0.0.1", port=9876, **kwargs):
        super().__init__(**kwargs)
        self.host = host
        self.port = port
        self.queue = asyncio.Queue()
        self._task = None

    def _broadcast(self, event_type: str, payload: dict):
        self.queue.put_nowait({"type": event_type, "payload": payload})
        if self._task is None:
            try:
                loop = asyncio.get_running_loop()
                self._task = loop.create_task(self._publisher())
            except RuntimeError:
                pass

    async def _publisher(self):
        writer = None
        try:
            while True:
                msg = await self.queue.get()
                if not writer:
                    try:
                        _, writer = await asyncio.open_connection(self.host, self.port)
                    except Exception:
                        self.queue.task_done()
                        continue

                try:
                    data = json.dumps(msg) + "\n"
                    writer.write(data.encode('utf-8'))
                    await writer.drain()
                except Exception:
                    writer = None
                self.queue.task_done()
        except asyncio.CancelledError:
            if writer:
                writer.close()
                try:
                    await writer.wait_closed()
                except Exception:
                    pass

    async def on_node_queued(self, node_id, priority):
        self._broadcast("node_queued", {"node_id": node_id, "priority": priority})
        return await super().on_node_queued(node_id, priority)

    async def on_node_start(self, node_id, worker_id):
        self._broadcast("node_start", {"node_id": node_id, "worker_id": worker_id})
        return await super().on_node_start(node_id, worker_id)

    async def on_node_complete(self, node_id, status, result):
        payload = {
            "node_id": node_id,
            "status": status.name,
            "text": getattr(result, 'text', '') if result else str(result),
            "reasoning": getattr(result, 'reasoning', '') if result else "",
            "tool_calls": getattr(result, 'tool_calls', []) if result else []
        }
        self._broadcast("node_complete", payload)
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
        try:
            await asyncio.wait_for(self.queue.join(), timeout=2.0)
        except asyncio.TimeoutError:
            pass
        return await super().on_graph_complete(diagnostics)
