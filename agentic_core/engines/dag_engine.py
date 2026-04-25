
import asyncio
from enum import Enum, auto
from typing import Any, Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field
import logging

from .engine import AgentRunner, RunnerConfig
from ..memory.manager import MemoryManager
from ..observers import AgentEventObserver

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
    result: Any = None
    max_retries: int = 0
    current_retries: int = 0

class DAGEventObserver(AgentEventObserver):
    def on_node_queued(self, node_id: str, priority: int):
        logger.info(f"[DAG] Node {node_id} queued with priority {priority}")

    def on_node_start(self, node_id: str, worker_id: int):
        logger.info(f"[DAG] Worker {worker_id} starting node {node_id}")

    def on_node_complete(self, node_id: str, status: NodeState, result: Any):
        logger.info(f"[DAG] Node {node_id} completed with status {status}")

    def on_node_retry(self, node_id: str, retry_count: int, max_retries: int):
        logger.info(f"[DAG] Node {node_id} failed. Retrying ({retry_count}/{max_retries})...")

    def on_graph_complete(self, diagnostics: Dict[str, Any]):
        logger.info(f"[DAG] Graph execution complete. Diagnostics: {diagnostics}")

class DAGAgentRunner:
    def __init__(
        self, 
        nodes_def: Dict[str, Tuple[AgentRunner, RunnerConfig, str, Optional[int]]], 
        edges: List[Tuple[str, str]], 
        max_concurrency: int = 4,
        observer: Optional[DAGEventObserver] = None
    ):
        """
        nodes_def: {node_id: (runner, config, prompt, max_retries)}
        edges: [(parent_id, child_id)]
        """
        self.nodes: Dict[str, DAGNode] = {}
        self.out_edges: Dict[str, List[str]] = {node_id: [] for node_id in nodes_def}
        self.in_edges: Dict[str, List[str]] = {node_id: [] for node_id in nodes_def}
        self.in_degree: Dict[str, int] = {node_id: 0 for node_id in nodes_def}

        for node_id, def_vals in nodes_def.items():
            runner, config, prompt = def_vals[:3]
            max_retries = def_vals[3] if len(def_vals) > 3 else 0
            self.nodes[node_id] = DAGNode(node_id, runner, config, prompt, max_retries=max_retries)

        for parent, child in edges:
            if parent not in self.nodes or child not in self.nodes:
                raise ValueError(f"Edge {parent} -> {child} contains undefined nodes")
            self.out_edges[parent].append(child)
            self.in_edges[child].append(parent)
            self.in_degree[child] += 1
            self.nodes[child].in_degree = self.in_degree[child]

        self.max_concurrency = max_concurrency
        self.observer = observer or DAGEventObserver()
        self.queue = asyncio.PriorityQueue()

    def compile(self):
        """Cycle detection and Critical Path priority scoring."""
        temp_in_degree = self.in_degree.copy()
        queue = [n for n in self.nodes if temp_in_degree[n] == 0]
        visited_count = 0

        while queue:
            u = queue.pop(0)
            visited_count += 1
            for v in self.out_edges[u]:
                temp_in_degree[v] -= 1
                if temp_in_degree[v] == 0:
                    queue.append(v)

        if visited_count != len(self.nodes):
            raise RuntimeError("ConfigurationError: Cycle detected in DAG")

        priorities = {node_id: 0 for node_id in self.nodes}
        topo_order = []
        temp_in_degree = self.in_degree.copy()
        queue = [n for n in self.nodes if temp_in_degree[n] == 0]
        while queue:
            u = queue.pop(0)
            topo_order.append(u)
            for v in self.out_edges[u]:
                temp_in_degree[v] -= 1
                if temp_in_degree[v] == 0:
                    queue.append(v)

        for u in reversed(topo_order):
            if not self.out_edges[u]:
                priorities[u] = 1
            else:
                priorities[u] = 1 + max(priorities[v] for v in self.out_edges[u])

        for node_id, priority in priorities.items():
            self.nodes[node_id].priority = priority

    async def _worker(self, worker_id: int):
        while True:
            try:
                prio, node_id = await self.queue.get()
                node = self.nodes[node_id]

                self.observer.on_node_start(node_id, worker_id)
                node.state = NodeState.RUNNING

                try:
                    parent_results = [
                        f"Node {p_id} result: {self.nodes[p_id].result}"
                        for p_id in self.in_edges[node_id]
                    ]

                    context_prefix = "\\n\\nParent Context:\\n" + "\\n".join(parent_results) if parent_results else ""
                    full_prompt = node.prompt + context_prefix

                    result = await node.runner.run_turn(
                        user_input=full_prompt,
                        observer=self.observer,
                        config=node.config
                    )

                    if "error" in result:
                        raise Exception(result["error"])

                    node.result = result
                    node.state = NodeState.SUCCESS
                    self.observer.on_node_complete(node_id, NodeState.SUCCESS, result)

                    for child_id in self.out_edges[node_id]:
                        child = self.nodes[child_id]
                        if child.state == NodeState.FAILED_UPSTREAM:
                            continue
                        child.in_degree -= 1
                        if child.in_degree == 0:
                            child.state = NodeState.READY
                            await self.queue.put((-child.priority, child_id))
                            self.observer.on_node_queued(child_id, child.priority)

                except Exception as e:
                    error_msg = str(e)
                    # Determine if error is retryable (simplistic keyword check for v1)
                    retryable_keywords = ["timeout", "rate limit", "api error", "overloaded"]
                    is_retryable = any(kw in error_msg.lower() for kw in retryable_keywords)

                    if is_retryable and node.current_retries < node.max_retries:
                        node.current_retries += 1
                        node.state = NodeState.RETRYING
                        self.observer.on_node_retry(node_id, node.current_retries, node.max_retries)

                        # Exponential backoff: 2^retry_count seconds
                        backoff = 2 ** node.current_retries
                        await asyncio.sleep(backoff)

                        # Re-queue the node
                        await self.queue.put((-node.priority, node_id))
                    else:
                        logger.error(f"Node {node_id} failed permanently: {e}")
                        node.state = NodeState.FAILED
                        node.result = error_msg
                        self.observer.on_node_complete(node_id, NodeState.FAILED, node.result)
                        await self._cascade_failure(node_id)

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
                self.observer.on_node_complete(node_id, NodeState.FAILED_UPSTREAM, "Upstream failure")
                stack.extend(self.out_edges[node_id])

    async def execute(self):
        self.compile()
        for node_id, node in self.nodes.items():
            if node.in_degree == 0:
                node.state = NodeState.READY
                await self.queue.put((-node.priority, node_id))
                self.observer.on_node_queued(node_id, node.priority)

        workers = [asyncio.create_task(self._worker(i)) for i in range(self.max_concurrency)]
        try:
            await self.queue.join()
        finally:
            for w in workers:
                w.cancel()
            await asyncio.gather(*workers, return_exceptions=True)

        diagnostics = {node_id: node.state.name for node_id, node in self.nodes.items()}
        self.observer.on_graph_complete(diagnostics)
        return diagnostics
