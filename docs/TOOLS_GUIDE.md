# Tool Implementation Guide

Tools in `agentic_core` are implemented by subclassing `BaseTool`. The framework maps the python implementation to an OpenAI-compatible JSON schema and manages the execution context.

## 1. Class Implementation

To implement a tool, define the `name` and `schema` class attributes (or `_name` and `_schema` instance-wide attribute in constructor), and override the `execute` method.

```python
from agentic_core.tools.base import BaseTool

class SystemQueryTool(BaseTool):
    def __init__(self):
        super().__init__()
        self._name = "system_query"
        self._schema = {
            "type": "function",
            "function": {
                "name": "system_query",
                "description": "Retrieve system metrics.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "metric": {"type": "string", "enum": ["cpu", "ram"]}
                    },
                    "required": ["metric"]
                }
            }
        }

    async def execute(self, args: dict, context: dict) -> str:
        # 'args' contains the validated JSON parameters from the LLM
        metric = args.get("metric")
        return f"{metric} usage is at 45%"
```

### Return Types
The `execute` method must return a `str`. If the tool interacts with APIs returning JSON, the implementation must serialize it to a string (`json.dumps()`). If the string exceeds the configured `max_chars` limit in `RunnerConfig`, the framework automatically truncates the output.

## 2. Context Injection

The `context` dictionary passed to the `execute` method provides access to runtime orchestration objects.

### Custom Variables
Variables can be injected globally during `AgentBuilder` initialization or per-turn via `RunnerConfig`.
```python
# Injected via RunnerConfig(extra_context={"user_id": 123})
user_id = context.get("user_id")
```

### Graph State Bus
If the tool is executing within a `GraphAgentRunner`, the global state bus is accessible via `context["dag_state"]`. The tool can mutate this dictionary safely as the execution engine guarantees state isolation and deterministic merging. Subsequent nodes in the graph can read these mutations.
```python
async def execute(self, args: dict, context: dict) -> str:
    if "dag_state" in context:
        context["dag_state"]["last_accessed_metric"] = args.get("metric")
    return "Success"
```

## 3. Tool Registration

Tools are bound to the execution environment via the `AgentBuilder`.

```python
from agentic_core.agents import AgentBuilder

agent = AgentBuilder() \
    .with_tools([SystemQueryTool()]) \
    .build()
```

For dynamic grouping, use `ToolManager` toolsets.

```python
from agentic_core.config import ToolsetConfig
from agentic_core.tools import ToolManager

tools = ToolManager(
    toolsets={
        "diagnostics": ToolsetConfig(
            tools=["system_query"],
            prompt="You have access to diagnostic tools. Use them to debug the server."
        )
    }
)
tools.register_tool(SystemQueryTool())
```