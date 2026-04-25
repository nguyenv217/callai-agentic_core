"""
Slightly more complicated example using `textual` for TUI coding agents.

Requirements: 
    `agentic_core`: this package
    `textual`: for the layouts. Install via `pip install textual`
"""
import asyncio
import os
import psutil
from pathlib import Path

try:
    from textual.app import App, ComposeResult
    from textual.containers import Container, Vertical, VerticalScroll
    from textual.widgets import Header, Footer, Input, Static, Label, ProgressBar, Markdown
    from textual.reactive import reactive
    from textual.binding import Binding
except ImportError:
    raise ImportError("Please install 'textual' to try out this example!")

from agentic_core.agents import create_openai_agent, chat
from agentic_core.engines.engine import RunnerConfig

class SystemInfoWidget(Static):
    """Widget to display system info."""
    def on_mount(self):
        self.update_info()
        self.set_interval(2, self.update_info)

    def update_info(self):
        cwd = os.getcwd()
        cpu_usage = psutil.cpu_percent()
        ram = psutil.virtual_memory()
        
        try:
            files = [f.name for f in Path(cwd).iterdir() if f.is_file()][:5]
            struct = "\n".join([f"  • {f}" for f in files])
        except Exception:
            struct = "  • Unable to read"

        cpu_freq = psutil.cpu_freq().current if psutil.cpu_freq() else "N/A"

        content = (
            f"[cyan]CWD:[/cyan] {cwd}\n"
            f"[cyan]CWD Structure:[/cyan]\n{struct}\n"
            f"[cyan]CPU:[/cyan] {cpu_freq} GHz, {psutil.cpu_count()} Cores, {cpu_usage}% Used\n"
            f"[cyan]RAM:[/cyan] {ram.total // (1024**3)} GB, {ram.percent}% Used"
        )
        self.update(content)

class AgentStatusWidget(Static):
    """Widget to display agent status."""
    progress = reactive(0.0)
    
    def render(self):
        return (
            f"[white]Task Progress:[/white] [bold]{int(self.progress * 100)}%[/bold]\n"
            f"• Model: Codex-X-v2\n"
            f"• Sub-agents: 5 Active\n"
            f"• Current Workflow:\n"
        )

class CodexAgentTUI(App):
    CSS = """
    Screen {
        background: #0f111a;
    }
    #main-container {
        layout: horizontal;
    }
    #left-column {
        width: 70%;
        margin: 1;
    }
    #session-header {
        background: #1a1c25;
        color: #569cd6;
        text-style: bold;
        padding: 1;
        margin-bottom: 1;
        border: solid #333;
    }
    #console-area {
        border: solid #333;
        background: #1a1c25;
        overflow-x: hidden;
    }
    #chat-log {
        height: 1fr;
        padding: 1;
        overflow-x: hidden; 
    }
    #sidebar {
        width: 30%;
        margin: 1;
    }
    .section-box {
        border: solid #333;
        margin-bottom: 1;
        padding: 1;
        background: #1a1c25;
    }
    .section-title {
        color: #569cd6;
        text-style: bold;
        margin-bottom: 1;
    }
    #input-area {
        dock: bottom;
        margin: 1;
    }
    .user-message {
        color: #aaa;
        margin-bottom: 1;
        text-style: italic;
    }
    .agent-message {
        margin-bottom: 2;
        border-left: solid #569cd6;
        padding-left: 1;
        width: 100%;
        height: auto; /* Ensure it expands to fit content */
    }
    ProgressBar {
        width: 100%;
        margin-bottom: 1;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(id="main-container"):
            with Vertical(id="left-column"):
                yield Static("[bold cyan]CODEX AGENT[/bold cyan] Session: main-proj-X | Active | Cloud Sync | git: 'feature/api-v2'", id="session-header")
                with Vertical(id="console-area"):
                    yield VerticalScroll(id="chat-log")
                    yield Input(placeholder="Enter command (e.g., codex explain calculate_risk)...", id="input-area")
            
            with Vertical(id="sidebar"):
                with Vertical(classes="section-box"):
                    yield Label("Agent Status", classes="section-title")
                    yield AgentStatusWidget(id="agent-status")
                    yield ProgressBar(id="task-progress", total=100, show_eta=False)
                    yield Label("• Tools Loaded: 14\n• MCP Servers: 'proj_data_server', 'api_helper'")
                
                with Vertical(classes="section-box"):
                    yield Label("System Info", classes="section-title")
                    yield SystemInfoWidget()
        yield Footer()

    def append_text(self, text: str, style_class: str = ""):
        chat_log = self.query_one("#chat-log", VerticalScroll)
        chat_log.mount(Static(text, classes=style_class))
        chat_log.scroll_end()

    def append_markdown(self, text: str):
        chat_log = self.query_one("#chat-log", VerticalScroll)
        # Use height="auto" to ensure Markdown doesn't truncate
        markdown_widget = Markdown(text)
        container = Vertical(markdown_widget, classes="agent-message")
        chat_log.mount(container)
        chat_log.scroll_end()

    async def on_mount(self):
        self.runner = create_openai_agent(
            api_key=os.getenv("AGENT_API_KEY"),
            model=os.getenv("AGENT_MODEL"),
            base_url=os.getenv("AGENT_BASE_URL")
        )
        self.config = RunnerConfig(
            system_prompt="You are a highly advanced CLI coding agent. Format your responses cleanly with sections for Analysis, Flowcharts (using ASCII boxes), and Refined Code blocks.",
            max_iterations=10
        )
        # Removed the header append_text since it's now in compose()

    async def on_input_submitted(self, event: Input.Submitted):
        user_input = event.value.strip()
        if not user_input:
            return
        
        input_widget = self.query_one("#input-area", Input)
        input_widget.value = ""
        
        self.append_text(f">>> {user_input}", "user-message")
        
        progress_bar = self.query_one("#task-progress", ProgressBar)
        status_widget = self.query_one("#agent-status", AgentStatusWidget)
        
        for p in range(0, 101, 20):
            progress_bar.progress = p
            status_widget.progress = p / 100.0
            await asyncio.sleep(0.05)

        try:
            response = await chat(
                message=user_input,
                runner=self.runner,
                config=self.config,
                verbose=False
            )
            self.append_markdown(f"**Agent**\n\n{response}")
        except Exception as e:
            self.append_text(f"[red]Error: {str(e)}[/red]")
        
        progress_bar.progress = 0
        status_widget.progress = 0.0

if __name__ == "__main__":
    app = CodexAgentTUI()
    app.run()
