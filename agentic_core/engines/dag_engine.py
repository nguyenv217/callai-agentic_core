from __future__ import annotations
from typing import Tuple, TYPE_CHECKING, Optional
import asyncio
import traceback
from enum import Enum, auto
from dataclasses import dataclass
import logging

from agentic_core.decisions import (
    GraphRoutingDecision, 
    ErrorContext, 
    ErrorDecision,
)
from agentic_core.handlers import DAGEventHandler
from agentic_core.config import ConfigurationError, RunnerConfig
from agentic_core.interfaces import (
    AgentResponse, 
    DAGNodeResponse, 
    DAGResponse, 
    NodeValidationError, 
    NodeExecutionError
)
from agentic_core.utils import clean_context_for_downstream, convert_exception_to_message

if TYPE_CHECKING:
    from agentic_core.engines import AgentRunner

logger = logging.getLogger(__name__)

class NodeState(Enum):
    PENDING = auto()
    READY = auto()
    RUNNING = auto()
    SUCCESS = auto()
    FAILED = auto()
    FAILED_UPSTREAM = auto()
    RETRYING = auto()

@dataclass
class DAGNode:
    node_id: str
    runner: AgentRunner
    config: RunnerConfig
    prompt: str
    priority: int = 0
    state: NodeState = NodeState.PENDING
    in_degree: int = 0
    result: AgentResponse | None = None
    max_retries: int = 0
    current_retries: int = 0
    error: BaseException | None = None
    error_details: str | None = None
    failed_by: str | None = None

class DAGAgentRunner:
    def __init__(
        self, 
        nodes_def: dict[str, Tuple[AgentRunner, RunnerConfig, str, int]] | None, 
        edges: list[Tuple[str, str]], 
        max_concurrency: int = 4,
        handler: DAGEventHandler | None = None,
        checkpoint_state: dict[str, AgentResponse] | None = None,
        default_max_retries: int = 3,
        default_backoff_base: float = 1.0,
        default_backoff_max: float = 60.0
    ):
        """
        Engine for concurrent dispatch of agent swarms with dependencies modeled as a Directed Acyclic Graph (DAG) .

        Args:
            nodes_def: {node_id: (runner, config, prompt, max_retries)}
            edges: [(parent_id, child_id)]
            max_concurrency: Maximum number of concurrent nodes to run at once.
            handler: Optional handler for tracking execution events.
            checkpoint_state: Optional state from a previous failed run to resume from.
            default_max_retries: Default max retries for nodes that don't specify.
            default_backoff_base: Default base delay for exponential backoff.
            default_backoff_max: Default max delay for exponential backoff.
        """
        self.nodes: dict[str, DAGNode] = {}
        self.out_edges: dict[str, list[str]] = {node_id: [] for node_id in nodes_def}
        self.in_edges: dict[str, list[str]] = {node_id: [] for node_id in nodes_def}
        self.in_degree: dict[str, int] = {node_id: 0 for node_id in nodes_def}
        
        self.default_backoff_base = default_backoff_base
        self.default_backoff_max = default_backoff_max

        for node_id, def_vals in nodes_def.items():
            runner, config, prompt = def_vals[:3]
            max_retries = def_vals[3] if len(def_vals) > 3 else default_max_retries
            self.nodes[node_id] = DAGNode(node_id, runner, config, prompt, max_retries=max_retries)

        for parent, child in edges:
            if parent not in self.nodes or child not in self.nodes:
                raise NodeValidationError(f"Edge {parent} -> {child} contains undefined nodes")
            self.out_edges[parent].append(child)
            self.in_edges[child].append(parent)
            self.in_degree[child] += 1
            self.nodes[child].in_degree = self.in_degree[child]

        self.max_concurrency = max_concurrency
        self.handler = handler or DAGEventHandler()
        self.queue = asyncio.PriorityQueue()
        self.active_retries = 0  
        
        if checkpoint_state:
            for node_id, result in checkpoint_state.items():
                if node_id in self.nodes:
                    node = self.nodes[node_id]
                    node.state = NodeState.SUCCESS
                    node.result = result
                    for child_id in self.out_edges[node_id]:
                        self.in_degree[child_id] -= 1
                        self.nodes[child_id].in_degree = self.in_degree[child_id]



    def compile(self):
        from collections import deque
        temp_in_degree = self.in_degree.copy()
        queue = deque([n for n in self.nodes if temp_in_degree[n] == 0])
        visited_nodes = set()
        topo_order = []

        while queue:
            u = queue.popleft()
            visited_nodes.add(u)
            topo_order.append(u)
            for v in self.out_edges[u]:
                temp_in_degree[v] -= 1
                if temp_in_degree[v] == 0:
                    queue.append(v)

        if len(visited_nodes) != len(self.nodes):
            raise NodeValidationError("Cycle detected in DAG.")

        priorities = {}
        for u in reversed(topo_order):
            children = self.out_edges[u]
            priorities[u] = 1 if not children else 1 + max(priorities[v] for v in children)

        for node_id, priority in priorities.items():
            self.nodes[node_id].priority = priority

    async def _schedule_retry(self, node_id: str, priority: int, delay: float):
        try:
            await asyncio.sleep(delay)
            await self.queue.put((-priority, node_id))
        finally:
            self.active_retries -= 1

    # In agentic_core/engines/dag_engine.py
    async def _create_error_context(self, e: BaseException, node_id: str, retry_count: int = 0) -> ErrorContext:
        return ErrorContext(
            error=e,
            retry_count=retry_count,
            max_retries=self.nodes[node_id].max_retries,  
            engine_state={"node_id": node_id}
        )

    async def _handle_error_decision(self, error_context: ErrorContext, node_id: str) -> tuple[bool, Optional[AgentResponse]]:
        decision_event = await self.handler.on_error(error_context)
        action = decision_event.action
        
        if isinstance(action, ErrorDecision.RETRY):
            return False, None
            
        elif isinstance(action, ErrorDecision.SKIP):
            return True, AgentResponse(text=f"Node {node_id} skipped due to error: {str(error_context.error)}")
            
        elif isinstance(action, ErrorDecision.RESOLVE_WITH):
            return True, AgentResponse(text=action.msg)
            
        else: # ABANDON
            return True, None

    async def _worker(self, worker_id: int):
        while True:
            try:
                prio, node_id = await self.queue.get()
                node = self.nodes[node_id]

                if node.state in (NodeState.SUCCESS, NodeState.FAILED, NodeState.FAILED_UPSTREAM):
                    self.queue.task_done()
                    continue

                await self.handler.on_node_start(node_id, worker_id)
                node.state = NodeState.RUNNING

                try:
                    parent_results = [
                        f"Node {p_id} result: {clean_context_for_downstream(self.nodes[p_id].result.text)}"
                        for p_id in self.in_edges[node_id]
                    ]
                    context_prefix = "\n\nParent Context:\n" + "\n".join(parent_results) if parent_results else ""
                    
                    result = await node.runner.run_turn(
                        user_input=node.prompt + context_prefix,
                        handler=self.handler,
                        config=node.config
                    )

                    if result.error:
                        raise NodeExecutionError(node_id, str(result.error), result.error)

                    node.result = result
                    node.state = NodeState.SUCCESS
                    await self.handler.on_node_complete(node_id, NodeState.SUCCESS, result)

                    for child_id in self.out_edges[node_id]:
                        child = self.nodes[child_id]
                        if child.state == NodeState.FAILED_UPSTREAM: continue
                        child.in_degree -= 1
                        if child.in_degree == 0:
                            child.state = NodeState.READY
                            await self.queue.put((-child.priority, child_id))
                            await self.handler.on_node_queued(child_id, child.priority)

                except Exception as e:
                    tb_str = traceback.format_exc()
                    error_context = await self._create_error_context(e, node_id, node.current_retries)
                    should_abort, fallback_response = await self._handle_error_decision(error_context, node_id)

                    if not should_abort and node.current_retries < node.max_retries:
                        node.current_retries += 1
                        node.state = NodeState.RETRYING
                        await self.handler.on_node_retry(node_id, node.current_retries, node.max_retries)
                        
                        # Inspect the decision again to grab backoff config
                        decision = (await self.handler.on_error(error_context)).action
                        delay = decision.delay * (decision.exponential_base ** node.current_retries) if isinstance(decision, ErrorDecision.RETRY) else self.default_backoff_base
                        
                        self.active_retries += 1
                        asyncio.create_task(self._schedule_retry(node_id, node.priority, delay))
                        
                    elif should_abort and fallback_response:
                        node.result = fallback_response
                        node.state = NodeState.SUCCESS
                        node.error = e
                        node.error_details = tb_str

                        for child_id in self.out_edges[node_id]:
                            child = self.nodes[child_id]
                            if child.state == NodeState.FAILED_UPSTREAM: continue
                            child.in_degree -= 1
                            if child.in_degree == 0:
                                child.state = NodeState.READY
                                await self.queue.put((-child.priority, child_id))
                        await self.handler.on_node_complete(node_id, NodeState.SUCCESS, fallback_response)
                    else:
                        node.state = NodeState.FAILED
                        node.error = e
                        node.error_details = tb_str

                        decision_event = await self.handler.on_node_permanent_failure(node_id, e)
                        if isinstance(decision_event.action, GraphRoutingDecision.IGNORE):
                            node.result = AgentResponse(text=f"IGNORED: Node {node_id} failed permanently: {convert_exception_to_message(e)}.")
                            node.state = NodeState.SUCCESS 
                            await self.handler.on_node_complete(node_id, NodeState.SUCCESS, node.result)
                            for child_id in self.out_edges[node_id]:
                                child = self.nodes[child_id]
                                if child.state == NodeState.FAILED_UPSTREAM: continue
                                child.in_degree -= 1
                                if child.in_degree == 0:
                                    child.state = NodeState.READY
                                    await self.queue.put((-child.priority, child_id))
                        elif isinstance(decision_event.action, GraphRoutingDecision.FALLBACK):
                            # Future implementation: Inject dynamic fallback runner here
                            pass 
                        else: # CASCADE
                            await self._cascade_failure(node_id)
                            await self.handler.on_node_complete(node_id, NodeState.FAILED, str(e))
                finally:
                    self.queue.task_done()
            except asyncio.CancelledError:
                break

    async def _cascade_failure(self, failed_node_id: str):
        stack = list(self.out_edges[failed_node_id])
        visited = set()
        while stack:
            node_id = stack.pop()
            if node_id in visited: continue
            visited.add(node_id)
            node = self.nodes[node_id]

            if node.state not in (NodeState.SUCCESS, NodeState.FAILED):
                node.state = NodeState.FAILED_UPSTREAM
                node.failed_by = failed_node_id
                node.result = f"Upstream failure caused by node: {failed_node_id}"
                await self.handler.on_node_complete(node_id, NodeState.FAILED_UPSTREAM, node.result)
                stack.extend(self.out_edges[node_id])

    async def execute(self) -> DAGResponse:
        try:
            self.compile()
        except RuntimeError as e:
            return DAGResponse(error=RuntimeError(f"DAG compilation failed: {str(e)}"))
            
        for node_id, node in self.nodes.items():
            if node.state == NodeState.SUCCESS: continue 
            if node.in_degree == 0:
                node.state = NodeState.READY
                await self.queue.put((-node.priority, node_id))

        workers = [asyncio.create_task(self._worker(i)) for i in range(self.max_concurrency)]

        try:
            while True:  # avoid queue.join() misintepreting empty queue while retrying is underway 
                await self.queue.join()
                if self.active_retries == 0:
                    break
                await asyncio.sleep(0.01)
        finally:
            for w in workers: w.cancel()
            await asyncio.gather(*workers, return_exceptions=True)

        nodes_resp = {
            node_id: DAGNodeResponse(state=n.state.name, result=n.result, error=n.error, error_details=n.error_details, failed_by=n.failed_by)
            for node_id, n in self.nodes.items()
        }

        response = DAGResponse(nodes=nodes_resp)
        await self.handler.on_graph_complete(response)
        return response