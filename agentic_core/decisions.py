from __future__ import annotations

from typing import Generic, TypeVar, Union, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from agentic_core.engines import AgentRunner

# ===================================================
# Decision Events
# ===================================================

ActionT = TypeVar("ActionT")

@dataclass
class DecisionEvent(Generic[ActionT]):
    """Event for observing the decision made by an agent.
    Attributes:
        action: a context-specific `ActionT` event, may contain additional context as attribute.
    """    
    action: ActionT

# ===================================================
# On Tool start
# ===================================================     

class ToolStartDecision:
    """
    Decision before each tool execution.
    
    Options:
        CONTINUE: proceed with execution
        SKIP: skip this tool only
        ABANDON: halt all execution and return final response immediately 
        SKIP_WITH_MSG: skip this tool, BUT leave a message for the agent as the tool result
        BREAK_WITH_MSG: skip every tool in this turn, BUT leave a message for the agent as the tool result
    """
    @dataclass(frozen=True)
    class CONTINUE:
        name: str = "CONTINUE"  
    
    @dataclass(frozen=True)
    class SKIP:
        name: str = "SKIP"      

    @dataclass(frozen=True)
    class ABANDON:
        name: str = "ABANDON"   
    
    @dataclass(frozen=True)
    class SKIP_WITH_MSG:
        msg: str
        name: str = "SKIP_WITH_MSG"
    
    @dataclass(frozen=True)
    class BREAK_WITH_MSG:
        msg: str
        name: str = "BREAK_WITH_MSG"

    @dataclass(frozen=True)
    class SUSPEND:
        name: str = "SUSPEND"

ToolStartAction = Union[
    ToolStartDecision.CONTINUE,
    ToolStartDecision.SKIP, 
    ToolStartDecision.SKIP_WITH_MSG, 
    ToolStartDecision.ABANDON, 
    ToolStartDecision.BREAK_WITH_MSG,
    ToolStartDecision.SUSPEND
]

# ===================================================
# On Error Handling
# ===================================================

@dataclass
class ErrorContext:
    error: BaseException
    tool_name: str | None = None
    retry_count: int = 0
    max_retries: int = 0
    # Additional state can be bundled to avoid a 10-argument constructor
    engine_state: dict | None = None


class ErrorDecision:

    @dataclass(frozen=True)
    class RETRY:
        """Handles both immediate and backoff retries."""
        delay: float = 0.0                # 0.0 means immediate retry
        exponential_base: float = 1.0     # 1.0 means flat delay, >1.0 means backoff
        name: str = "RETRY"
    
    @dataclass(frozen=True)
    class SKIP:
        """Skip current operation and continue."""
        name: str = "SKIP"
    
    @dataclass(frozen=True)
    class ABANDON:
        """Stop execution and bubble up the failure."""
        name: str = "ABANDON"
    
    @dataclass(frozen=True)
    class RESOLVE_WITH:
        """
        Unifies FALLBACK, DECAY, ESCALATE, and CUSTOM.
        Injects a specific message or result back into the agent's context 
        to gracefully recover or pivot.
        """
        msg: str                          
        name: str = "RESOLVE_WITH"

ErrorAction = Union[
    ErrorDecision.RETRY,
    ErrorDecision.SKIP,
    ErrorDecision.ABANDON,
    ErrorDecision.RESOLVE_WITH
]

# ===================================================
# On Last Iteration
# ===================================================   

class LastIterationDecision:
    """
    Decision after the last tool execution.
    
    Options:
        CONTINUE: proceed with the last iteration (agent may continue calling tools until iteration budget is depleted)
        LEAVE_MSG: leave a final message for the agent and continue with the last iteration
        ABANDON: return immediately
        EXTEND: extends the max iteration budget by `max_iterations_count` (if not supplied/is None, defaults to current config's `max_iterations`) 
    """
    @dataclass(frozen=True)
    class CONTINUE:
        name: str = "CONTINUE"
    
    @dataclass(frozen=True)
    class ABANDON:
        name: str = "ABANDON"
    
    @dataclass(frozen=True)
    class LEAVE_MSG:
        msg: str
        name: str = "LEAVE_MSG"
    
    @dataclass(frozen=True)
    class EXTEND:
        extra_iterations_count: int | None = None
        name: str = "EXTEND"

LastIterationAction = Union[
    LastIterationDecision.CONTINUE,
    LastIterationDecision.LEAVE_MSG,
    LastIterationDecision.ABANDON,
    LastIterationDecision.EXTEND
]

# ===================================================
# During Tool Execution
# ===================================================   

class ToolOnPromptDecision:
    """
    Decision during tool execution when confirmation is required.
    
    Options:
        CONFIRM: proceed with execution
        REJECT: reject the execution
        REJECT_WITH_MSG: reject with a message
    """
    @dataclass(frozen=True)
    class CONFIRM:
        name: str = "CONFIRM"
    
    @dataclass(frozen=True)
    class REJECT:
        name: str = "REJECT"
    
    @dataclass(frozen=True)
    class REJECT_WITH_MSG:
        msg: str
        name: str = "REJECT_WITH_MSG"

ToolOnPromptAction = Union[
    ToolOnPromptDecision.CONFIRM,
    ToolOnPromptDecision.REJECT,
    ToolOnPromptDecision.REJECT_WITH_MSG
]

# ===================================================
# During DAG engine node failure (after retrying)
# ===================================================      

class GraphRoutingDecision:
    """
    Decision for graph topology when a node fails permanently (all retries exhausted).
    """
    @dataclass(frozen=True)
    class CASCADE:
        """Standard behavior: Fail all downstream nodes that depend on this one."""
        name: str = "CASCADE"
    
    @dataclass(frozen=True)
    class IGNORE:
        """Continue execution: Allow downstream nodes to run (they must handle missing inputs)."""
        name: str = "IGNORE"
    
    @dataclass(frozen=True)
    class FALLBACK:
        """Dynamic replacement: Inject a new AgentRunner/Node to take its place."""
        fallback_runner: AgentRunner # Pass the actual runner/logic here
        name: str = "FALLBACK"

GraphRoutingAction = Union[
    GraphRoutingDecision.CASCADE,
    GraphRoutingDecision.IGNORE,
    GraphRoutingDecision.FALLBACK
]
