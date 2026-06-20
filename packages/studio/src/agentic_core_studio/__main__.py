import sys
import json
import asyncio
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, DataTable, RichLog, Markdown
from textual.containers import Horizontal, Vertical

class TCPServerProtocol(asyncio.Protocol):
    def __init__(self, app):
        self.app = app
        self.buffer = ""

    def data_received(self, data):
        self.buffer += data.decode('utf-8')
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            if line.strip():
                try:
                    msg = json.loads(line)
                    self.app.call_from_thread(self.app.handle_event, msg)
                except Exception:
                    pass

class StudioAnalyzer(App):
    CSS = """
    #left_pane { width: 35%; border-right: solid $primary; }
    #right_pane { width: 65%; }
    #details_panel { height: 70%; border-bottom: solid $primary; overflow: auto; padding: 1 2; }
    #logs_panel { height: 30%; }
    DataTable { height: 100%; }
    """
    BINDINGS = [("q", "quit", "Quit")]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            with Vertical(id="left_pane"):
                yield DataTable(id="nodes_table")
            with Vertical(id="right_pane"):
                yield Markdown("# Waiting for Telemetry...", id="details_panel")
                yield RichLog(id="logs_panel", wrap=True, highlight=True)
        yield Footer()

    async def on_mount(self):
        self.nodes_data = {}
        self.table = self.query_one("#nodes_table", DataTable)
        self.table.add_columns("Node ID", "State")
        self.table.cursor_type = "row"
        self.log_widget = self.query_one("#logs_panel", RichLog)
        self.details = self.query_one("#details_panel", Markdown)
        
        loop = asyncio.get_running_loop()
        try:
            self.server = await loop.create_server(
                lambda: TCPServerProtocol(self),
                '127.0.0.1', 9876
            )
            self.log_widget.write("[bold green]Agentic Studio Analyzer Listening on TCP 127.0.0.1:9876[/bold green]")
        except Exception as e:
            self.log_widget.write(f"[bold red]Failed to bind TCP server: {e}[/bold red]")

    def handle_event(self, message):
        evt_type = message.get("type")
        payload = message.get("payload", {})
        node_id = payload.get("node_id")

        if node_id and node_id not in self.nodes_data:
            self.nodes_data[node_id] = {"state": "UNKNOWN", "text": "", "reasoning": "", "error": "", "tools": []}
            self.table.add_row(node_id, "UNKNOWN", key=node_id)

        if evt_type == "node_queued":
            self.nodes_data[node_id]["state"] = "QUEUED"
            self.log_widget.write(f"[cyan]Queued[/cyan] {node_id}")
        elif evt_type == "node_start":
            self.nodes_data[node_id]["state"] = "RUNNING"
            self.log_widget.write(f"[yellow]Started[/yellow] {node_id}")
        elif evt_type == "node_complete":
            self.nodes_data[node_id]["state"] = payload.get("status", "SUCCESS")
            self.nodes_data[node_id]["text"] = payload.get("text", "")
            self.nodes_data[node_id]["reasoning"] = payload.get("reasoning", "")
            self.nodes_data[node_id]["tools"] = payload.get("tool_calls", [])
            self.log_widget.write(f"[green]Completed[/green] {node_id} -> {payload.get('status')}")
        elif evt_type == "error":
            self.nodes_data[node_id]["state"] = "ERROR"
            self.nodes_data[node_id]["error"] = payload.get("error", "")
            self.log_widget.write(f"[red]Error[/red] in {node_id}: {payload.get('error')}")
        elif evt_type == "tool_start":
            self.log_widget.write(f"[magenta]Tool Execution[/magenta] {payload.get('tool_name')}")
        elif evt_type == "graph_complete":
            self.log_widget.write("[bold green]Graph Execution Completed[/bold green]")

        if node_id:
            self.table.update_cell(node_id, "State", self.nodes_data[node_id]["state"])
            if self.table.cursor_row is not None and self.table.cursor_row >= 0:
                row_key = self.table.coordinate_to_cell_key(self.table.cursor_coordinate).row_key
                if row_key is not None and row_key.value == node_id:
                    self.update_details(node_id)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted):
        if event.row_key is None:
            return
        node_id = event.row_key.value
        self.update_details(node_id)

    def update_details(self, node_id):
        data = self.nodes_data.get(node_id, {})
        md = f"# Node: {node_id}\n**State:** {data['state']}\n\n"
        if data.get('error'):
            md += f"## Error Trace\n```python\n{data['error']}\n```\n"
        if data.get('tools'):
            md += f"## Tool Calls\n"
            for tc in data['tools']:
                func = tc.get("function", {})
                md += f"- **{func.get('name')}**: `{func.get('arguments')}`\n"
            md += "\n"
        if data.get('reasoning'):
            md += f"## Reasoning\n{data['reasoning']}\n\n"
        if data.get('text'):
            md += f"## Final Output\n{data['text']}\n"
        
        self.details.update(md)

def main():
    app = StudioAnalyzer()
    app.run()

if __name__ == "__main__":
    main()
