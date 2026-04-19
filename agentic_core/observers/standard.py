"""
Standard observer implementations.
"""
from .base import AgentEventObserver


class DefaultObserver(AgentEventObserver):
    """A no-op observer that prints nothing. Use this if you don't care about events."""
    
    def on_turn_start(self) -> None: pass
    def on_iteration_start(self, iteration: int, max_iterations: int) -> None: pass
    def on_llm_progress(self, info: str) -> None: pass
    def on_tool_start(self, tool_name: str, tool_id: str) -> None: pass
    def on_tool_complete(self, tool_name: str, tool_id: str, success: bool, result: str) -> None: pass
    def on_turn_complete(self, response: dict) -> None: pass
    def on_error(self, error: str) -> None: pass


class PrintObserver(AgentEventObserver):
    """An observer that prints everything - great for debugging."""
    
    def on_turn_start(self) -> None:
        print("📍 [TURN START]")
    
    def on_iteration_start(self, iteration: int, max_iterations: int) -> None:
        print(f"🔄 [ITERATION {iteration}/{max_iterations}]")
    
    def on_llm_progress(self, info: str) -> None:
        print(f"💬 [LLM]: {info[:200]}")
    
    def on_tool_start(self, tool_name: str, tool_id: str) -> None:
        print(f"🔧 [TOOL START]: {tool_name}")
    
    def on_tool_complete(self, tool_name: str, tool_id: str, success: bool, result: str) -> None:
        status = "✅" if success else "❌"
        print(f"{status} [TOOL COMPLETE]: {tool_name} -> {str(result)[:100]}")
    
    def on_turn_complete(self, response: dict) -> None:
        print(f"🏁 [TURN COMPLETE]")
    
    def on_error(self, error: str) -> None:
        print(f"❗ [ERROR]: {error}")