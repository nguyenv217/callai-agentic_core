from __future__ import annotations
import asyncio
import traceback
import logging
from typing import Any, Callable, Tuple, Awaitable

from agentic_core.config import RunnerConfig
from agentic_core.engines.engine import AgentRunner
from agentic_core.interfaces import AgentResponse, DAGResponse, DAGNodeResponse
from agentic_core.tools.base import BaseTool
from agentic_core.handlers.dag import DAGEventHandler
from agentic_core.engines.dag_engine import NodeState
from agentic_core.utils import clean_context_for_downstream

logger = logging.getLogger(__name__)

class UpdateSwarmStateTool(BaseTool):
    def __init__(self):
        super().__init__()
        self._name = "update_swarm_state"
        self._schema = {
            "type": "function",
            "function": {
                "name": "update_swarm_state",
                "description": "Updates the shared global state bus for the entire swarm. Use this to pass data, findings, or instructions to downstream agents.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string"},
                        "value": {"type": "string"}
                    },
                    "required": ["key", "value"]
                }
            }
        }

    async def execute(self, args: dict, context: dict) -> str:
        state = context.get("swarm_state")
        if state is None: 
            return "Error: Swarm state bus not found."
        state[args["key"]] = args["value"]
        return f"State updated: {args['key']} = {args['value']}"

class TransferControlTool(BaseTool):
    def __init__(self, available_agents: list[str], queue_callback: Callable[[str, str], Awaitable[None]]):
        super().__init__()
        self._name = "transfer_control"
        self.available_agents = available_agents
        self.queue_callback = queue_callback
        self._schema = {
            "type": "function",
            "function": {
                "name": "transfer_control",
                "description": "Transfers execution control to another agent in the swarm. Use this to explicitly hand off a task, return from a delegation, or request a review.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_agent": {
                            "type": "string",
                            "enum": available_agents + ["__end__"],
                            "description": "The ID of the agent to transfer control to, or '__end__' if the task is completely finished."
                        },
                        "reason": {
                            "type": "string",
                            "description": "Reason for the transfer or termination (will be logged)."
                        }
                    },
                    "required": ["target_agent", "reason"]
                }
            }
        }

    async def execute(self, args: dict, context: dict) -> str:
        target = args.get("target_agent")
        source_node = context.get("current_node_id", "unknown")
        if target == "__end__":
            return "Swarm termination requested. This branch will now end."
        if target not in self.available_agents:
            return f"Error: Agent '{target}' not found. Available: {self.available_agents + ['__end__']}"
        await self.queue_callback(target, source_node)
        return f"Successfully requested transfer of control to {target}."

class StatefulSwarmEngine:
    def __init__(
        self,
        nodes_def: dict[str, Tuple[AgentRunner, RunnerConfig, str, int]],
        edges: list[Tuple[str, str]],
        max_swarm_steps: int = 50,
        max_concurrency: int = 4,
        handler: DAGEventHandler | None = None,
        initial_state: dict[str, Any] | None = None
    ):
        self.nodes_def = nodes_def
        self.edges = edges
        self.max_swarm_steps = max_swarm_steps
        self.max_concurrency = max_concurrency
        self.handler = handler or DAGEventHandler()
        self.state = initial_state or {}
        
        self.out_edges = {n: [] for n in nodes_def}
        self.in_edges = {n: [] for n in nodes_def}
        for u, v in edges:
            if u in self.out_edges and v in self.in_edges:
                self.out_edges[u].append(v)
                self.in_edges[v].append(u)

    async def execute(self, start_nodes: list[str] | None = None) -> DAGResponse:
        queue = asyncio.Queue()
        current_in_degree = {n: len(self.in_edges[n]) for n in self.nodes_def}
        
        if start_nodes:
            for n in start_nodes:
                queue.put_nowait((n, None))
        else:
            for n, deg in current_in_degree.items():
                if deg == 0:
                    queue.put_nowait((n, None))
                    
        if queue.empty() and not start_nodes and self.nodes_def:
            queue.put_nowait((list(self.nodes_def.keys())[0], None))
                    
        step_count = 0
        results = {n: DAGNodeResponse(state=NodeState.PENDING.name, result=None) for n in self.nodes_def}
        node_history = {n: [] for n in self.nodes_def}
        active_tasks = 0
        state_lock = asyncio.Lock()
        
        async def enqueue_agent(target_id: str, source_id: str):
            await self.handler.on_node_queued(target_id, 0)
            queue.put_nowait((target_id, source_id))

        state_tool = UpdateSwarmStateTool()
        transfer_tool = TransferControlTool(list(self.nodes_def.keys()), enqueue_agent)
        
        for n_id, defs in self.nodes_def.items():
            defs[0].tools.register_tool(state_tool)
            defs[0].tools.register_tool(transfer_tool)

        async def _worker(worker_id: int):
            nonlocal step_count, active_tasks
            while True:
                try:
                    node_id, source_id = await queue.get()
                    active_tasks += 1
                    
                    if step_count >= self.max_swarm_steps:
                        logger.warning(f"Swarm step limit ({self.max_swarm_steps}) reached.")
                        results[node_id].state = NodeState.FAILED.name
                        results[node_id].error_details = "Max swarm steps reached (Circuit breaker)."
                        queue.task_done()
                        active_tasks -= 1
                        continue
                        
                    step_count += 1
                    await self.handler.on_node_start(node_id, worker_id)
                    runner, config, prompt, max_retries = self.nodes_def[node_id]
                    results[node_id].state = NodeState.RUNNING.name
                    
                    retries = 0
                    while retries <= max_retries:
                        try:
                            async with state_lock:
                                state_str = "\n".join(f"{k}: {v}" for k, v in self.state.items()) if self.state else "No state yet."
                            
                            parent_results = []
                            if source_id and node_history.get(source_id):
                                parent_results.append(f"[Transferred from {source_id}]: {clean_context_for_downstream(node_history[source_id][-1].text)}")
                            else:
                                for p_id in self.in_edges.get(node_id, []):
                                    if node_history[p_id]:
                                        parent_results.append(f"[Upstream {p_id}]: {clean_context_for_downstream(node_history[p_id][-1].text)}")
                            
                            context_prefix = ("\n\n[UPSTREAM AGENT CONTEXT]\n" + "\n".join(parent_results)) if parent_results else ""
                            full_prompt = prompt + context_prefix + f"\n\n[SHARED SWARM STATE]\n{state_str}\n\nUpdate the state if necessary using `update_swarm_state`. If you need to route to another agent or signal completion, use `transfer_control`."
                            
                            if not config.extra_context:
                                config.extra_context = {}
                            config.extra_context["swarm_state"] = self.state
                            config.extra_context["current_node_id"] = node_id
                            
                            if not config.tools:
                                config.tools = [state_tool.schema, transfer_tool.schema]
                            else:
                                if state_tool.schema not in config.tools:
                                    config.tools.append(state_tool.schema)
                                if transfer_tool.schema not in config.tools:
                                    config.tools.append(transfer_tool.schema)

                            result = await runner.run_turn(full_prompt, config=config, handler=self.handler)
                            
                            results[node_id] = DAGNodeResponse(state=NodeState.SUCCESS.name, result=result)
                            node_history[node_id].append(result)
                            await self.handler.on_node_complete(node_id, NodeState.SUCCESS, result)
                            
                            explicit_transfer = any(tc.get("function", {}).get("name") == "transfer_control" for tc in result.tool_calls)
                            if not explicit_transfer:
                                for nxt in self.out_edges.get(node_id, []):
                                    current_in_degree[nxt] -= 1
                                    if current_in_degree[nxt] <= 0:
                                        await self.handler.on_node_queued(nxt, 0)
                                        queue.put_nowait((nxt, None))
                                        current_in_degree[nxt] = len(self.in_edges[nxt])
                            break
                            
                        except Exception as e:
                            if retries == max_retries:
                                tb = traceback.format_exc()
                                results[node_id] = DAGNodeResponse(state=NodeState.FAILED.name, result=None, error=e, error_details=tb)
                                break
                            retries += 1
                            logger.warning(f"Retrying node {node_id} ({retries}/{max_retries}) due to error: {e}")
                            await asyncio.sleep(1.0)
                    
                    queue.task_done()
                    active_tasks -= 1
                        
                except asyncio.CancelledError:
                    break

        workers = [asyncio.create_task(_worker(i)) for i in range(self.max_concurrency)]
        
        while True:
            if queue.empty() and active_tasks == 0:
                break
            await asyncio.sleep(0.05)
        
        for w in workers:
            w.cancel()
            
        resp = DAGResponse(nodes=results)
        await self.handler.on_graph_complete(resp)
        return resp
