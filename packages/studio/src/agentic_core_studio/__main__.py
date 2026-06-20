import sys
import json
import asyncio
import datetime
from collections import deque
from rich.live import Live
from rich.layout import Layout
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

class UDPTelemetryServer(asyncio.DatagramProtocol):
    def __init__(self, analyzer):
        self.analyzer = analyzer

    def datagram_received(self, data, addr):
        try:
            message = json.loads(data.decode('utf-8'))
            self.analyzer.handle_event(message)
        except Exception:
            pass

class StudioAnalyzer:
    def __init__(self):
        self.nodes = {}
        self.logs = deque(maxlen=30)
        self.status = "LISTENING (Port 9876)"
        self.metrics = {"tools_called": 0, "errors": 0}
        self.start_time = datetime.datetime.now()

    def handle_event(self, message):
        evt_type = message.get("type")
        payload = message.get("payload", {})

        if evt_type == "node_queued":
            self.nodes[payload["node_id"]] = {"state": "QUEUED", "snippet": ""}
            self.logs.appendleft(f"[cyan]Queued[/cyan] {payload['node_id']}")
        elif evt_type == "node_start":
            self.nodes[payload["node_id"]]["state"] = "RUNNING"
            self.logs.appendleft(f"[yellow]Started[/yellow] {payload['node_id']}")
        elif evt_type == "node_complete":
            self.nodes[payload["node_id"]]["state"] = payload["status"]
            self.nodes[payload["node_id"]]["snippet"] = payload.get("snippet", "").replace('\n', ' ')
            self.logs.appendleft(f"[green]Completed[/green] {payload['node_id']} -> {payload['status']}")
        elif evt_type == "error":
            self.metrics["errors"] += 1
            self.logs.appendleft(f"[red]Error[/red] in {payload.get('node_id')}: {payload.get('error')} -> {payload.get('action')}")
            if payload.get("node_id") in self.nodes:
                self.nodes[payload["node_id"]]["state"] = "ERROR"
        elif evt_type == "tool_start":
            self.metrics["tools_called"] += 1
            self.logs.appendleft(f"[magenta]Tool Execution[/magenta] {payload.get('tool_name')}")
        elif evt_type == "graph_complete":
            self.status = "COMPLETED"
            self.logs.appendleft("[bold green]Graph Execution Completed[/bold green]")

    def generate_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main")
        )
        layout["main"].split_row(
            Layout(name="left_panel", ratio=2),
            Layout(name="right_panel", ratio=1)
        )
        layout["right_panel"].split_column(
            Layout(name="metrics", size=8),
            Layout(name="logs")
        )

        # Header
        uptime = datetime.datetime.now() - self.start_time
        header_text = Text(f"Agentic Core Studio | Status: {self.status} | Uptime: {str(uptime).split('.')[0]}", style="bold white on blue", justify="center")
        layout["header"].update(Panel(header_text))

        # Nodes Table
        table = Table(title="Live Execution Topology", expand=True)
        table.add_column("Node ID", style="cyan")
        table.add_column("State", style="bold")
        table.add_column("Output Snippet", style="dim")
        
        for node_id, info in self.nodes.items():
            state = info["state"]
            color = "white"
            if state == "SUCCESS": color = "green"
            elif state == "RUNNING": color = "yellow"
            elif state == "QUEUED": color = "cyan"
            elif state in ("FAILED", "ERROR", "FAILED_UPSTREAM"): color = "red"
            elif state == "SKIPPED": color = "grey50"
            
            table.add_row(node_id, f"[{color}]{state}[/{color}]", info["snippet"][:60])
        
        layout["left_panel"].update(Panel(table, border_style="blue"))

        # Metrics
        metrics_table = Table.grid(padding=1)
        metrics_table.add_column(style="bold cyan")
        metrics_table.add_column(justify="right")
        metrics_table.add_row("Active Nodes:", str(sum(1 for n in self.nodes.values() if n["state"] == "RUNNING")))
        metrics_table.add_row("Tools Executed:", str(self.metrics["tools_called"]))
        metrics_table.add_row("Exceptions Caught:", str(self.metrics["errors"]))
        layout["metrics"].update(Panel(metrics_table, title="Global Metrics", border_style="cyan"))

        # Logs
        log_text = "\n".join(self.logs)
        layout["logs"].update(Panel(log_text, title="Live Event Telemetry", border_style="magenta"))

        return layout

async def run_studio():
    analyzer = StudioAnalyzer()
    loop = asyncio.get_running_loop()
    transport, protocol = await loop.create_datagram_endpoint(
        lambda: UDPTelemetryServer(analyzer),
        local_addr=('127.0.0.1', 9876)
    )
    
    try:
        with Live(analyzer.generate_layout(), refresh_per_second=10, screen=True) as live:
            while True:
                await asyncio.sleep(0.1)
                live.update(analyzer.generate_layout())
    except KeyboardInterrupt:
        pass
    finally:
        transport.close()

def main():
    try:
        asyncio.run(run_studio())
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
