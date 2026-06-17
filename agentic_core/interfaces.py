from typing import Any, Literal, TypedDict, NotRequired

# ===================================================
# Common response interfaces 
# ===================================================

class ToolResponse(TypedDict):
    id: str
    type: Literal["function"]

    class Function(TypedDict):
        name: str
        arguments: str | dict[str, Any]
    
    function: Function

class Message(TypedDict):
    role: Literal["user", "assistant", "tool"]
    content: str | list[Any]
    
    # The following arguments are presented when role = "tool"
    tool_name: NotRequired[str | None]
    tool_call_id: NotRequired[str | None]
    
    # The following arguments are presented when role = "assistant"
    tool_calls: NotRequired[list[ToolResponse]]
    reasoning: NotRequired[str]
    usage: NotRequired[dict[str, Any] | None]

class ToolSchema(TypedDict):
    type: Literal["function"]

    class FunctionSchema(TypedDict):
        name: str
        description: str

        class Parameters(TypedDict):
            type: Literal["object"]
            properties: dict[str, dict[str, Any]]

        parameters: Parameters
        required: list[str]

    function: FunctionSchema

class MCPServerDef(TypedDict, total=False):
    command: str
    args: list[str]
    env: dict[str, str]
    log_file: str
    timeout_s: float
    url: str

class MCPConfigDict(TypedDict, total=False):
    mcpServers: dict[str, MCPServerDef]
