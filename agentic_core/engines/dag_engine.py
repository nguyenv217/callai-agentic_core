from __future__ import annotations
from typing import Tuple, TYPE_CHECKING, Optional, Callable, Any
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
from agentic_core.models import AgentResponse, DAGNodeResponse, DAGResponse
from agentic_core.exceptions import NodeValidationError, NodeExecutionError
from agentic_core.utils import clean_context_for_downstream, convert_exception_to_message

if TYPE_CHECKING:
    from agentic_core.engines import AgentRunner
    from agentic_core.interfaces import IPersistenceProvider

logger = logging.getLogger(__name__)

class NodeState(Enum):
    PENDING = auto()
    READY = auto()
    RUNNING = auto()
    SUCCESS = auto()
    FAILED = auto()
    FAILED_UPSTREAM = auto()
    RETRYING = auto()
    SKIPPED = auto()
    SUSPENDED = auto()

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
    active_parents: list[str] = None
    skipped_parents: list[str] = None
    context_assembler: Callable[[dict[str, AgentResponse], dict[str, Any]], str] | None = None
    history: list[AgentResponse] = None

    def __post_init__(self):
        if self.active_parents is None: self.active_parents = []
        if self.skipped_parents is None: self.skipped_parents = []
        if self.history is None: self.history = []

@dataclass
class DAGTask:
    runner: AgentRunner
    prompt: str
    config: RunnerConfig | None = None
    max_retries: int | None = None
    context_assembler: Callable[[dict[str, AgentResponse], dict[str, Any]], str] | None = None

class GraphAgentRunner:
    def __init__(
        self, 
        nodes_def: dict[str, Tuple[AgentRunner, RunnerConfig, str, int] | DAGTask] | None, 
        edges: list[Tuple[str, str] | Tuple[str, str, Callable[[AgentResponse, dict[str, Any]], bool]]], 
        max_concurrency: int = 4,
        handler: DAGEventHandler | None = None,
        checkpoint_state: dict[str, AgentResponse] | None = None,
        default_max_retries: int = 3,
        default_backoff_base: float = 1.0,
        default_backoff_max: float = 60.0,
        shared_state: dict[str, Any] | None = None,
        persistence_provider: IPersistenceProvider | None = None,
        session_id: str | None = None,
        task_broker: Any | None = None,
        state_reducer: Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], None] | None = None
    ):
        self.nodes: dict[str, DAGNode] = {}
        self.out_edges: dict[str, list[str]] = {node_id: [] for node_id in nodes_def}
        self.in_edges: dict[str, list[str]] = {node_id: [] for node_id in nodes_def}
        self.in_degree: dict[str, int] = {node_id: 0 for node_id in nodes_def}
        self.edge_conditions: dict[Tuple[str, str], Callable[[AgentResponse, dict[str, Any]], bool]] = {}
        self.shared_state = shared_state if shared_state is not None else {}
        
        self.default_backoff_base = default_backoff_base
        self.default_backoff_max = default_backoff_max

        for node_id, def_vals in nodes_def.items():
            if isinstance(def_vals, tuple):
                runner, config, prompt = def_vals[:3]
                max_retries = def_vals[3] if len(def_vals) > 3 else default_max_retries
                context_assembler = def_vals[4] if len(def_vals) > 4 else None
            else:
                runner = def_vals.runner
                config = def_vals.config or RunnerConfig()
                prompt = def_vals.prompt
                max_retries = def_vals.max_retries if def_vals.max_retries is not None else default_max_retries
                context_assembler = def_vals.context_assembler

            self.nodes[node_id] = DAGNode(
                node_id=node_id, 
                runner=runner, 
                config=config, 
                prompt=prompt, 
                max_retries=max_retries,
                context_assembler=context_assembler
            )

        for edge in edges:
            if len(edge) == 2:
                parent, child = edge
                condition = None
            elif len(edge) == 3:
                parent, child, condition = edge
            else:
                raise NodeValidationError(f"Invalid edge format: {edge}")

            if parent not in self.nodes or child not in self.nodes:
                raise NodeValidationError(f"Edge {parent} -> {child} contains undefined nodes")
            
            self.out_edges[parent].append(child)
            self.in_edges[child].append(parent)
            self.in_degree[child] += 1
            self.nodes[child].in_degree = self.in_degree[child]
            self.edge_conditions[(parent, child)] = condition

        self.max_concurrency = max_concurrency
        self.handler = handler or DAGEventHandler()
        self.queue = task_broker if task_broker is not None else asyncio.PriorityQueue()
        self.active_retries = 0  
        self.persistence_provider = persistence_provider
        self.session_id = session_id
        self.state_reducer = state_reducer
        self._state_lock = asyncio.Lock()
        
        if checkpoint_state:
            self._apply_checkpoint(checkpoint_state)

    def _apply_checkpoint(self, checkpoint_state: dict[str, AgentResponse]):
        for node_id, result in checkpoint_state.items():
            if node_id in self.nodes:
                node = self.nodes[node_id]
                node.state = NodeState.SUCCESS
                node.result = result
                for child_id in self.out_edges[node_id]:
                    cond = self.edge_conditions.get((node_id, child_id))
                    passed = True
                    if cond:
                        try:
                            passed = cond(result, self.shared_state)
                        except Exception:
                            passed = False
                    
                    child = self.nodes[child_id]
                    if passed:
                        child.active_parents.append(node_id)
                    else:
                        child.skipped_parents.append(node_id)
                        
                    self.in_degree[child_id] -= 1
                    self.nodes[child_id].in_degree = self.in_degree[child_id]

    async def _save_checkpoint(self, completed_node_id: str | None = None):
        if not (self.persistence_provider and self.session_id):
            return

        if completed_node_id and hasattr(self.persistence_provider, "save_node_result"):
            node = self.nodes[completed_node_id]
            try:
                await self.persistence_provider.save_node_result(
                    self.session_id, 
                    completed_node_id, 
                    node.result.to_dict() if node.result else {}
                )
            except Exception as e:
                logger.error(f"Failed to save node checkpoint: {e}")

        state = {
            "results": {
                n_id: n.result.to_dict() 
                for n_id, n in self.nodes.items() 
                if n.state == NodeState.SUCCESS and n.result
            },
            "shared_state": self.shared_state
        }
        try:
            await self.persistence_provider.save_checkpoint(self.session_id, state)
        except Exception as e:
            logger.error(f"Failed to save DAG checkpoint: {e}")

    def _apply_state_delta(self, shared_target: dict, original_snapshot: dict, local_state: dict):
        for k, v in local_state.items():
            if k not in original_snapshot:
                shared_target[k] = v
            elif original_snapshot[k] != v:
                if isinstance(v, dict) and isinstance(shared_target.get(k), dict) and isinstance(original_snapshot[k], dict):
                    self._apply_state_delta(shared_target[k], original_snapshot[k], v)
                elif isinstance(v, list) and isinstance(shared_target.get(k), list) and isinstance(original_snapshot[k], list):
                    orig_len = len(original_snapshot[k])
                    if len(v) > orig_len and v[:orig_len] == original_snapshot[k]:
                        shared_target[k].extend(v[orig_len:])
                    else:
                        shared_target[k] = v
                else:
                    shared_target[k] = v

    def to_mermaid(self) -> str:
        """Exports the compiled graph topology as a Mermaid.js flowchart."""
        try:
            self.compile()
        except Exception:
            pass
        
        lines = ["graph TD"]
        for node_id in self.nodes:
            lines.append(f"    {node_id}[{node_id}]")
            
        for parent, children in getattr(self, 'forward_edges', {}).items():
            for child in children:
                lines.append(f"    {parent} -->|Forward| {child}")
                
        for parent, children in getattr(self, 'back_edges', {}).items():
            for child in children:
                lines.append(f"    {parent} -.->|Loop| {child}")
                
        return "\n".join(lines)

    def compile(self):
        visited = set()
        rec_stack = set()
        self.forward_edges = {u: [] for u in self.nodes}
        self.back_edges = {u: [] for u in self.nodes}

        def dfs(u):
            visited.add(u)
            rec_stack.add(u)
            for v in self.out_edges[u]:
                if v not in visited:
                    self.forward_edges[u].append(v)
                    dfs(v)
                elif v in rec_stack:
                    self.back_edges[u].append(v)
                else:
                    self.forward_edges[u].append(v)
            rec_stack.remove(u)

        for node in self.nodes:
            if node not in visited:
                dfs(node)

        self.forward_in_degree = {n: 0 for n in self.nodes}
        for u in self.nodes:
            for v in self.forward_edges[u]:
                self.forward_in_degree[v] += 1
        
        for node_id, node in self.nodes.items():
            node.in_degree = self.forward_in_degree[node_id]

        from collections import deque
        temp_in_degree = self.forward_in_degree.copy()
        queue = deque([n for n in self.nodes if temp_in_degree[n] == 0])
        topo_order = []

        while queue:
            u = queue.popleft()
            topo_order.append(u)
            for v in self.forward_edges[u]:
                temp_in_degree[v] -= 1
                if temp_in_degree[v] == 0:
                    queue.append(v)

        priorities = {n: 1 for n in self.nodes}
        for u in reversed(topo_order):
            children = self.forward_edges[u]
            if children:
                priorities[u] = 1 + max(priorities[v] for v in children)

        for node_id, priority in priorities.items():
            self.nodes[node_id].priority = priority

    async def _schedule_retry(self, node_id: str, priority: int, delay: float):
        try:
            await asyncio.sleep(delay)
            await self.queue.put((-priority, node_id))
        finally:
            self.active_retries -= 1

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

    async def _resolve_forward_child(self, child_id: str, parent_id: str, passed_condition: bool):
        child = self.nodes[child_id]
        if child.state == NodeState.FAILED_UPSTREAM:
            return

        if passed_condition:
            if parent_id not in child.active_parents:
                child.active_parents.append(parent_id)
        else:
            if parent_id not in child.skipped_parents:
                child.skipped_parents.append(parent_id)

        total_received = len(child.active_parents) + len(child.skipped_parents)
        if total_received == self.forward_in_degree[child_id]:
            if not child.active_parents:
                child.state = NodeState.SKIPPED
                child.result = AgentResponse(text=f"Node {child_id} skipped because all parent branches were pruned.")
                await self.handler.on_node_complete(child_id, NodeState.SKIPPED, child.result)
                for grand_child_id in self.forward_edges[child_id]:
                    await self._resolve_forward_child(grand_child_id, child_id, passed_condition=False)
            else:
                child.state = NodeState.READY
                await self.queue.put((-child.priority, child_id))
                await self.handler.on_node_queued(child_id, child.priority)

    async def _trigger_back_edge(self, target_id: str, source_id: str):
        loop_body = set()
        stack = [target_id]
        while stack:
            curr = stack.pop()
            if curr not in loop_body:
                loop_body.add(curr)
                stack.extend(self.forward_edges[curr])
        
        async with self._state_lock:
            for node_id in loop_body:
                node = self.nodes[node_id]
                if node.result is not None:
                    node.history.append(node.result)
                node.state = NodeState.PENDING
                node.result = None
                node.error = None
                node.error_details = None
                node.failed_by = None
                node.current_retries = 0
                node.active_parents = [p for p in node.active_parents if p not in loop_body]
                node.skipped_parents = [p for p in node.skipped_parents if p not in loop_body]
        
        target_node = self.nodes[target_id]
        if source_id not in target_node.active_parents:
            target_node.active_parents.append(source_id)
            
        target_node.state = NodeState.READY
        await self.queue.put((-target_node.priority, target_id))
        await self.handler.on_node_queued(target_id, target_node.priority)

    async def _worker(self, worker_id: int):
        while True:
            try:
                prio, node_id = await self.queue.get()
                node = self.nodes[node_id]

                if node.state in (NodeState.SUCCESS, NodeState.FAILED, NodeState.FAILED_UPSTREAM, NodeState.SKIPPED):
                    self.queue.task_done()
                    continue

                await self.handler.on_node_start(node_id, worker_id)
                node.state = NodeState.RUNNING

                try:
                    import copy
                    turn_config = copy.copy(node.config)
                    if turn_config.extra_context is None:
                        turn_config.extra_context = {}
                    else:
                        turn_config.extra_context = turn_config.extra_context.copy()
                    
                    async with self._state_lock:
                        local_state = copy.deepcopy(self.shared_state)
                        original_state_snapshot = copy.deepcopy(local_state)
                    turn_config.extra_context["dag_state"] = local_state

                    # Retrieve executed parent results using the compiled graph in-edges
                    parent_responses = {}
                    for p_id in self.in_edges.get(node_id, []):
                        p_node = self.nodes[p_id]
                        if p_node.result is not None:
                            parent_responses[p_id] = p_node.result
                        elif getattr(p_node, 'history', None):
                            parent_responses[p_id] = p_node.history[-1]

                    if node.context_assembler:
                        try:
                            context_prefix = node.context_assembler(parent_responses, self.shared_state)
                        except Exception as ce:
                            logger.error(f"Context assembler failed for node {node_id}: {ce}")
                            context_prefix = ""
                    else:
                        parent_results = [
                            f"Node {p_id} result: {clean_context_for_downstream(resp.text)}"
                            for p_id, resp in parent_responses.items()
                        ]
                        context_prefix = "\n\nParent Context:\n" + "\n".join(parent_results) if parent_results else ""

                    result = await node.runner.run_turn(
                        user_input=node.prompt + context_prefix,
                        handler=self.handler,
                        config=turn_config
                    )

                    if result.error:
                        raise NodeExecutionError(node_id, str(result.error), result.error)

                    async with self._state_lock:
                        if self.state_reducer:
                            self.state_reducer(self.shared_state, original_state_snapshot, local_state)
                        else:
                            self._apply_state_delta(self.shared_state, original_state_snapshot, local_state)

                    node.result = result
                    node.state = NodeState.SUCCESS
                    await self.handler.on_node_complete(node_id, NodeState.SUCCESS, result)
                    await self._save_checkpoint(node_id)

                    for child_id in self.forward_edges[node_id]:
                        cond = self.edge_conditions.get((node_id, child_id))
                        passed = True
                        if cond:
                            try:
                                passed = cond(result, local_state)
                            except Exception as ce:
                                logger.warning(f"Edge condition {node_id}->{child_id} failed with exception: {ce}")
                                passed = False
                        await self._resolve_forward_child(child_id, node_id, passed)

                    for child_id in self.back_edges[node_id]:
                        cond = self.edge_conditions.get((node_id, child_id))
                        passed = True
                        if cond:
                            try:
                                passed = cond(result, local_state)
                            except Exception as ce:
                                logger.warning(f"Back-edge condition {node_id}->{child_id} failed with exception: {ce}")
                                passed = False
                        if passed:
                            await self._trigger_back_edge(child_id, node_id)

                except Exception as e:
                    tb_str = traceback.format_exc()
                    error_context = await self._create_error_context(e, node_id, node.current_retries)
                    should_abort, fallback_response = await self._handle_error_decision(error_context, node_id)

                    if not should_abort and node.current_retries < node.max_retries:
                        node.current_retries += 1
                        node.state = NodeState.RETRYING
                        await self.handler.on_node_retry(node_id, node.current_retries, node.max_retries)
                        
                        decision = (await self.handler.on_error(error_context)).action
                        delay = decision.delay * (decision.exponential_base ** node.current_retries) if isinstance(decision, ErrorDecision.RETRY) else self.default_backoff_base
                        
                        self.active_retries += 1
                        asyncio.create_task(self._schedule_retry(node_id, node.priority, delay))
                        
                    elif should_abort and fallback_response:
                        node.result = fallback_response
                        node.state = NodeState.SUCCESS
                        node.error = e
                        node.error_details = tb_str

                        await self.handler.on_node_complete(node_id, NodeState.SUCCESS, fallback_response)
                        await self._save_checkpoint(node_id)
                        for child_id in self.forward_edges[node_id]:
                            cond = self.edge_conditions.get((node_id, child_id))
                            passed = True
                            if cond:
                                try:
                                    passed = cond(fallback_response, self.shared_state)
                                except Exception as ce:
                                    logger.warning(f"Edge condition {node_id}->{child_id} failed with exception: {ce}")
                                    passed = False
                            await self._resolve_forward_child(child_id, node_id, passed)
                            
                        for child_id in self.back_edges[node_id]:
                            cond = self.edge_conditions.get((node_id, child_id))
                            passed = True
                            if cond:
                                try:
                                    passed = cond(fallback_response, self.shared_state)
                                except Exception as ce:
                                    logger.warning(f"Back-edge condition {node_id}->{child_id} failed with exception: {ce}")
                                    passed = False
                            if passed:
                                await self._trigger_back_edge(child_id, node_id)
                    else:
                        node.state = NodeState.FAILED
                        node.error = e
                        node.error_details = tb_str

                        decision_event = await self.handler.on_node_permanent_failure(node_id, e)
                        if isinstance(decision_event.action, GraphRoutingDecision.IGNORE):
                            node.result = AgentResponse(text=f"IGNORED: Node {node_id} failed permanently: {convert_exception_to_message(e)}.")
                            node.state = NodeState.SUCCESS 
                            await self.handler.on_node_complete(node_id, NodeState.SUCCESS, node.result)
                            await self._save_checkpoint(node_id)
                            for child_id in self.forward_edges[node_id]:
                                cond = self.edge_conditions.get((node_id, child_id))
                                passed = True
                                if cond:
                                    try:
                                        passed = cond(node.result, self.shared_state)
                                    except Exception as ce:
                                        logger.warning(f"Edge condition {node_id}->{child_id} failed with exception: {ce}")
                                        passed = False
                                await self._resolve_forward_child(child_id, node_id, passed)
                                
                            for child_id in self.back_edges[node_id]:
                                cond = self.edge_conditions.get((node_id, child_id))
                                passed = True
                                if cond:
                                    try:
                                        passed = cond(node.result, self.shared_state)
                                    except Exception as ce:
                                        logger.warning(f"Back-edge condition {node_id}->{child_id} failed with exception: {ce}")
                                        passed = False
                                if passed:
                                    await self._trigger_back_edge(child_id, node_id)
                        elif isinstance(decision_event.action, GraphRoutingDecision.FALLBACK):
                            pass 
                        else:
                            await self._cascade_failure(node_id)
                            await self.handler.on_node_complete(node_id, NodeState.FAILED, str(e))
                finally:
                    self.queue.task_done()
            except asyncio.CancelledError:
                break

    async def _cascade_failure(self, failed_node_id: str):
        stack = list(self.forward_edges[failed_node_id])
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
                stack.extend(self.forward_edges[node_id])

    async def execute(self) -> DAGResponse:
        if self.persistence_provider and self.session_id:
            try:
                persisted = await self.persistence_provider.load_checkpoint(self.session_id)
                if persisted:
                    from agentic_core.models import AgentResponse
                    ckpt = {}
                    for k, v in persisted.get("results", {}).items():
                        if isinstance(v, dict):
                            ckpt[k] = AgentResponse(**{key: val for key, val in v.items() if key in AgentResponse.__dataclass_fields__})
                        else:
                            ckpt[k] = v
                    if ckpt:
                        self._apply_checkpoint(ckpt)
                    if "shared_state" in persisted:
                        self.shared_state.update(persisted["shared_state"])
            except Exception as e:
                logger.error(f"Failed to load DAG checkpoint: {e}")

        try:
            self.compile()
        except RuntimeError as e:
            return DAGResponse(error=RuntimeError(f"DAG compilation failed: {str(e)}"))
            
        for node_id, node in self.nodes.items():
            if node.state == NodeState.SUCCESS: continue 
            if self.forward_in_degree[node_id] == 0:
                node.state = NodeState.READY
                await self.queue.put((-node.priority, node_id))

        workers = [asyncio.create_task(self._worker(i)) for i in range(self.max_concurrency)]

        try:
            while True:  
                await self.queue.join()
                if self.active_retries == 0:
                    break
                await asyncio.sleep(0.05)
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

DAGAgentRunner = GraphAgentRunner
