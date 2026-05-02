"""
Standard observer implementations.
"""
from .base import AgentEventObserver
from ..decisions import ToolStartAction, ToolStartDecision, DecisionEvent
from ..tools import ToolExecutionController

class SilentObserver(AgentEventObserver):
    """A no-op observer that prints nothing. Use this if you don't care about events."""
    
    def on_turn_start(self) -> None: pass
    def on_iteration_start(self, iteration: int, max_iterations: int) -> None: pass
    def on_llm_progress(self, info: str) -> None: pass
    def on_tool_start(self, tool_name, tool_id, tool_args): return DecisionEvent(ToolStartDecision.CONTINUE())
    def on_tool_complete(self, tool_name: str, tool_id: str, success: bool, result: str) -> None: pass
    def on_turn_complete(self, response: dict) -> None: pass
    def on_error(self, error: str) -> None: pass


class PrintObserver(AgentEventObserver, ToolExecutionController):
    """An observer that prints everything - great for debugging."""
    
    def on_turn_start(self) -> None:
        print("📍 [TURN START]")
    
    def on_iteration_start(self, iteration: int, max_iterations: int) -> None:
        print(f"🔄 [ITERATION {iteration}/{max_iterations}]")
    
    def on_llm_progress(self, info: str) -> None:
        print(f"💬 [LLM]: {info[:200]}")

    def on_tool_call_session_start(self, reasoning_text, tool_calls, iteration, max_iterations):
        print(f"💡 [ITERATION {iteration}/{max_iterations}]: {reasoning_text[:500]}")
    
    def on_tool_start(self, tool_name: str, tool_id: str, tool_args: str | dict | None) -> DecisionEvent[ToolStartAction]:
        print(f"🔧 [TOOL START]: {tool_name}")
        return DecisionEvent(ToolStartDecision.CONTINUE())
    
    def on_tool_complete(self, tool_name: str, tool_id: str, success: bool, result: str) -> None:
        status = "✅" if success else "❌"
        print(f"{status} [TOOL COMPLETE]: {tool_name} -> {str(result)[:500]}")
    
    def on_turn_complete(self, response: dict) -> None:
        print(f"🏁 [TURN COMPLETE]")
    
    def on_error(self, error: str) -> None:
        print(f"❗ [ERROR]: {error}")

    def on_prompt_respond(self, prompt: str) -> str:
        return input(prompt)
    
    def on_prompt_confirmation(self, prompt, on_yes, on_no):
        if (input(prompt).strip().lower() in ['y', 'yes']): on_yes()
        elif on_no: on_no()

        