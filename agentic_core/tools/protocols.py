from __future__ import annotations
from typing import Awaitable, Callable, Protocol, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..decisions import DecisionEvent, ToolOnPromptDecision

class ToolExecutionController(Protocol):
    """Protocol for tool execution control."""
    on_chat_notified: Callable[[str], Awaitable[None]] | None = None
    on_prompt_respond: Callable[[Any], Awaitable[str]] | None = None
    on_prompt_confirmation: Callable[[Any], Awaitable[DecisionEvent[ToolOnPromptDecision]]] | None = None