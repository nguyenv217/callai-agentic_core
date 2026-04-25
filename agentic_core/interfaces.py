from dataclasses import dataclass
from typing import Generic, TypeVar, Protocol

class DecisionAction(Protocol):
    @property
    def required_message(self) -> bool: ...
    @property
    def name(self) -> str: ...

ActionT = TypeVar("ActionT", bound=DecisionAction)

@dataclass
class DecisionEvent(Generic[ActionT]):
    """Event for observing the decision made by an agent."""    
    action: ActionT
    message: str | None = None

    def __post_init__(self):
        if self.action.required_message and self.message is None:
            error_prefix = f"{type(self.action).__name__}.{self.action.name}"
            raise ValueError(f"{error_prefix} cannot be used without a message")
