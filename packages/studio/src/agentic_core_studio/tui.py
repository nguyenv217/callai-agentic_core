import asyncio
from typing import Any
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from agentic_core.handlers.dag import DAGSmartRetryHandler
from agentic_core.engines.dag_engine import NodeState
from agentic_core.models import AgentResponse, DAGResponse
from agentic_core.decisions import ErrorContext, DecisionEvent, ErrorAction, GraphRoutingAction

class GraphStudioHandler(DAGSmartRetryHandler):
    """A TUI 'multimeter' for tracing cyclic graph execution in real-time."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nodes_info = {}
        self.logs = []
        self.live = None

    def _log(self, msg: str):
        self.logs.append(msg)
        if len(self.logs) > 15:
            self.logs.pop(0)

    def generate_layout(self):
        layout = Layout()
        layout.split_column(
            Layout(name="table", ratio=2),
            Layout(name="logs", ratio=1)
        )
        
        table = Table(title="Graph Execution Studio", expand=True)
        table.add_column("Node ID", style="cyan")
        table.add_column("State", style="magenta")
        table.add_column("Retries", justify="right")
        table.add_column("Output Snippet", style="green")

        for node_id, info in self.nodes_info.items():
            state_val = info['state']
            state_str = state_val.name if isinstance(state_val, NodeState) else str(state_val)
            color = "white"
            if state_str == "SUCCESS": color = "green"
            elif state_str == "RUNNING": color = "yellow"
            elif state_str in ("FAILED", "FAILED_UPSTREAM"): color = "red"
            elif state_str == "SKIPPED": color = "grey50"
            elif state_str == "SUSPENDED": color = "blue"
            elif state_str == "RETRYING": color = "magenta"
            
            table.add_row(
                node_id, 
                f"[{color}]{state_str}[/{color}]", 
                str(info.get('retries', 0)), 
                info.get('snippet', '')[:80].replace('\n', ' ')
            )

        log_text = "\n".join(self.logs)
        layout["table"].update(Panel(table))
        layout["logs"].update(Panel(log_text, title="Telemetry & Hooks"))
        return layout

    async def on_node_queued(self, node_id: str, priority: int):
        if self.live is None:
            self.live = Live(self.generate_layout(), refresh_per_second=10)
            self.live.start()
        self.nodes_info.setdefault(node_id, {'state': NodeState.PENDING, 'retries': 0, 'snippet': ''})
        self._log(f"[blue]Queued[/blue] {node_id} (Priority: {priority})")
        self.live.update(self.generate_layout())
        await super().on_node_queued(node_id, priority)

    async def on_node_start(self, node_id: str, worker_id: int):
        self.nodes_info[node_id]['state'] = NodeState.RUNNING
        self._log(f"[yellow]Started[/yellow] {node_id} on worker {worker_id}")
        self.live.update(self.generate_layout())
        await super().on_node_start(node_id, worker_id)

    async def on_node_complete(self, node_id: str, status: NodeState, result: AgentResponse):
        self.nodes_info[node_id]['state'] = status
        if result and hasattr(result, 'text'):
            self.nodes_info[node_id]['snippet'] = result.text
        elif isinstance(result, str):
            self.nodes_info[node_id]['snippet'] = result
        self._log(f"[green]Completed[/green] {node_id} -> {status.name}")
        self.live.update(self.generate_layout())
        await super().on_node_complete(node_id, status, result)

    async def on_node_retry(self, node_id: str, retry_count: int, max_retries: int):
        self.nodes_info[node_id]['state'] = NodeState.RETRYING
        self.nodes_info[node_id]['retries'] = retry_count
        self._log(f"[magenta]Retrying[/magenta] {node_id} ({retry_count}/{max_retries})")
        self.live.update(self.generate_layout())
        await super().on_node_retry(node_id, retry_count, max_retries)

    async def on_error(self, error_context: ErrorContext) -> DecisionEvent[ErrorAction]:
        decision = await super().on_error(error_context)
        node_id = error_context.engine_state.get("node_id", "Unknown") if error_context.engine_state else "Unknown"
        self._log(f"[red]Error[/red] in {node_id}: {error_context.error.__class__.__name__} -> {decision.action.name}")
        if self.live: self.live.update(self.generate_layout())
        return decision

    async def on_node_permanent_failure(self, node_id: str, error: Exception) -> DecisionEvent[GraphRoutingAction]:
        decision = await super().on_node_permanent_failure(node_id, error)
        self._log(f"[bold red]Permanent Failure[/bold red] {node_id}: {error} -> {decision.action.name}")
        if self.live: self.live.update(self.generate_layout())
        return decision

    async def on_graph_complete(self, diagnostics: DAGResponse):
        self._log(f"[bold green]Graph Execution Complete[/bold green]")
        if self.live:
            self.live.update(self.generate_layout())
            self.live.stop()
        await super().on_graph_complete(diagnostics)
