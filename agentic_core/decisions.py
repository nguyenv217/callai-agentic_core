from typing import Generic, TypeVar, Union
from dataclasses import dataclass

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

ToolStartAction = Union[
   ToolStartDecision.CONTINUE,
   ToolStartDecision.SKIP, 
   ToolStartDecision.SKIP_WITH_MSG, 
   ToolStartDecision.ABANDON, 
   ToolStartDecision.BREAK_WITH_MSG
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

class NodeFailureDecision:
    """
    Decision to make when a node fails permanently (after retries)
    Options:
        CASCADE: Fail all downstream nodes
        IGNORE: Ignore the failure and continue execution
        FALLBACK: Use a fallback `AgentRunner` instance passed along context to replace the failed node 
    """
    @dataclass(frozen=True)
    class CASCADE:
        name: str = "CASCADE"
    
    @dataclass(frozen=True)
    class IGNORE:
        name: str = "IGNORE"
    
    @dataclass(frozen=True)
    class FALLBACK:
        msg: str
        name: str = "FALLBACK"

NodeFailureAction = Union[
    NodeFailureDecision.CASCADE,
    NodeFailureDecision.IGNORE,
    NodeFailureDecision.FALLBACK
]
