from __future__ import annotations
from typing import Tuple, TYPE_CHECKING
import asyncio
import traceback
from enum import Enum, auto
from dataclasses import dataclass
import logging

from agentic_core.decisions import NodeFailureDecision
from agentic_core.observers import DAGEventObserver
from agentic_core.config import ConfigurationError, RunnerConfig
from agentic_core.interfaces import AgentResponse, DAGNodeResponse, DAGResponse, ProviderRateLimitError, ProviderTimeoutError, NodeExecutionError
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
        observer: DAGEventObserver | None = None,
        checkpoint_state: dict[str, AgentResponse] | None = None
    ):
        """
        Engine for concurrent dispatch of agent swarms with dependencies modeled as a Directed Acyclic Graph (DAG) .

        Args:
            nodes_def: {node_id: (runner, config, prompt, max_retries)}
            edges: [(parent_id, child_id)]
            max_concurrency: Maximum number of concurrent nodes to run at once.
            observer: Optional observer for tracking execution events.
        """
        self.nodes: dict[str, DAGNode] = {}
        self.out_edges: dict[str, list[str]] = {node_id: [] for node_id in nodes_def}
        self.in_edges: dict[str, list[str]] = {node_id: [] for node_id in nodes_def}
        self.in_degree: dict[str, int] = {node_id: 0 for node_id in nodes_def}

        for node_id, def_vals in nodes_def.items():
            runner, config, prompt = def_vals[:3]
            max_retries = def_vals[3] if len(def_vals) > 3 else 0
            self.nodes[node_id] = DAGNode(node_id, runner, config, prompt, max_retries=max_retries)

        for parent, child in edges:
            if parent not in self.nodes or child not in self.nodes:
                raise ConfigurationError(f"Edge {parent} -> {child} contains undefined nodes")
            self.out_edges[parent].append(child)
            self.in_edges[child].append(parent)
            self.in_degree[child] += 1
            self.nodes[child].in_degree = self.in_degree[child]

        self.max_concurrency = max_concurrency
        self.observer = observer or DAGEventObserver()
        self.queue = asyncio.PriorityQueue()
        
        if checkpoint_state:
            for node_id, result in checkpoint_state.items():
                if node_id in self.nodes:
                    node = self.nodes[node_id]
                    node.state = NodeState.SUCCESS
                    node.result = result
                    # Decrement in_degree for all downstream nodes to unlock them
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
            unvisited_nodes = [node_id for node_id in self.nodes if node_id not in visited_nodes]
            cycle_message = (
                f"Cycle detected in DAG. Unprocessed nodes: {unvisited_nodes}. "
                f"Total nodes: {len(self.nodes)}, Visited nodes: {len(visited_nodes)}."
            )
            raise RuntimeError(cycle_message)

        priorities = {}
        for u in reversed(topo_order):
            children = self.out_edges[u]
            if not children:
                priorities[u] = 1
            else:
                priorities[u] = 1 + max(priorities[v] for v in children)

        for node_id, priority in priorities.items():
            self.nodes[node_id].priority = priority

    async def _schedule_retry(self, node_id: str, priority: int, delay: float):
        await asyncio.sleep(delay)
        await self.queue.put((-priority, node_id))

    async def _worker(self, worker_id: int):
        while True:
            try:
                prio, node_id = await self.queue.get()
                node = self.nodes[node_id]

                if node.state in (NodeState.SUCCESS, NodeState.FAILED, NodeState.FAILED_UPSTREAM):
                    self.queue.task_done()
                    continue

                await self.observer.on_node_start(node_id, worker_id)
                node.state = NodeState.RUNNING

                try:
                    parent_results = [
                        f"Node {p_id} result: {clean_context_for_downstream(self.nodes[p_id].result.text)}"
                        for p_id in self.in_edges[node_id]
                    ]
                    context_prefix = "\n\nParent Context:\n" + "\n".join(parent_results) if parent_results else ""
                    full_prompt = node.prompt + context_prefix

                    result = await node.runner.run_turn(
                        user_input=full_prompt,
                        observer=self.observer,
                        config=node.config
                    )

                    if result.error:
                        raise NodeExecutionError(node_id, str(result.error), result.error)

                    node.result = result
                    node.state = NodeState.SUCCESS
                    await self.observer.on_node_complete(node_id, NodeState.SUCCESS, result)

                    for child_id in self.out_edges[node_id]:
                        child = self.nodes[child_id]
                        if child.state == NodeState.FAILED_UPSTREAM:
                            continue
                        child.in_degree -= 1
                        if child.in_degree == 0:
                            child.state = NodeState.READY
                            await self.queue.put((-child.priority, child_id))
                            await self.observer.on_node_queued(child_id, child.priority)

                except Exception as e:
                    tb_str = traceback.format_exc()
                    error_msg = str(e)
                    
                    retryable_keywords = ["timeout", "rate limit", "api error", "overloaded"]
                    retryable_exec_types = [ProviderRateLimitError, ProviderTimeoutError]
                    is_retryable = any(kw in error_msg.lower() for kw in retryable_keywords) or any(isinstance(e, exc_type) for exc_type in retryable_exec_types)

                    if is_retryable and node.current_retries < node.max_retries:
                        node.current_retries += 1
                        node.state = NodeState.RETRYING
                        await self.observer.on_node_retry(node_id, node.current_retries, node.max_retries)

                        backoff = 2 ** node.current_retries
                        asyncio.create_task(self._schedule_retry(node_id, node.priority, backoff))
                        await self.queue.put((-node.priority, node_id))
                    else:
                        logger.error(f"Node {node_id} failed permanently:\n{tb_str}")
                        node.state = NodeState.FAILED
                        node.error = e
                        node.error_details = tb_str

                        decision_event = await self.observer.on_node_permanent_failure(node_id, e)
                        match decision_event.action:
                            case NodeFailureDecision.CASCADE():
                                await self._cascade_failure(node_id)
                                await self.observer.on_node_complete(node_id, NodeState.FAILED, error_msg)
                                raise e

                            case NodeFailureDecision.IGNORE():
                                node.result = AgentResponse(text=f"IGNORED: Node {node_id} failed permanently: {convert_exception_to_message(e)}.")
                                node.state = NodeState.SUCCESS 
                        
                            case NodeFailureDecision.FALLBACK(msg):
                                node.result = AgentResponse(text=msg)
                                node.state = NodeState.SUCCESS 
                finally:
                    self.queue.task_done()
            except asyncio.CancelledError:
                break

    async def _cascade_failure(self, failed_node_id: str):
        stack = list(self.out_edges[failed_node_id])
        visited = set()
        while stack:
            node_id = stack.pop()
            if node_id in visited:
                continue
            visited.add(node_id)
            node = self.nodes[node_id]

            if node.state not in (NodeState.SUCCESS, NodeState.FAILED):
                node.state = NodeState.FAILED_UPSTREAM
                node.failed_by = failed_node_id
                fail_msg = f"Upstream failure caused by node: {failed_node_id}"
                node.result = fail_msg
                await self.observer.on_node_complete(node_id, NodeState.FAILED_UPSTREAM, fail_msg)
                stack.extend(self.out_edges[node_id])

    async def execute(self) -> DAGResponse:
        try:
            self.compile()
        except RuntimeError as e:
            return DAGResponse(error=RuntimeError(f"DAG compilation failed: {str(e)}"))
            
        for node_id, node in self.nodes.items():
            if node.state == NodeState.SUCCESS:
                continue # Skip nodes pre-completed by checkpoint
                
            if node.in_degree == 0:
                node.state = NodeState.READY
                await self.queue.put((-node.priority, node_id))
                await self.observer.on_node_queued(node_id, node.priority)

        workers = [asyncio.create_task(self._worker(i)) for i in range(self.max_concurrency)]

        try:
            await self.queue.join()
        finally:
            for w in workers:
                w.cancel()
            await asyncio.gather(*workers, return_exceptions=True)

        nodes_resp = {}
        for node_id, node in self.nodes.items():
            nodes_resp[node_id] = DAGNodeResponse(
                state=node.state.name,
                result=node.result,
                error=node.error,
                error_details=node.error_details,
                failed_by=node.failed_by
            )

        response = DAGResponse(nodes=nodes_resp)
        await self.observer.on_graph_complete(response)
        return response