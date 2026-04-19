
from typing import Protocol, Iterator, Any

class LLMResponse(Protocol):
    success: bool
    text: str | None
    tool_calls: list[dict] | None
    usage: dict | None
    error: str | None

class ILLMClient(Protocol):
    def ask(
        self,
        messages: list[dict],
        tools: list[dict],
        **kwargs,
    ) -> Iterator[LLMResponse]:
        """Yield progress or final response – completely agnostic to routing."""
        ...
