# Tools Framework Documentation

This document explains how the **tooling framework** in `agentic_core.tools` works, how to implement custom tools, register them, and interact with the Model Context Protocol (MCP) dynamic tools. It is complementary to the **MCP configuration guide** (`docs/MCP config.md`).

---

## 1. Core Concepts

| Concept | Description |
|---------|-------------|
| **BaseTool** | Abstract base class that all executable tools must inherit from. It defines the required `name`, `schema`, and `execute` members. |
| **ToolManager** | Central registry that holds tool instances, builds the JSON‑schema list for the LLM, and coordinates execution. It also handles lazy loading of MCP tools. |
| **MCPToolAdapter** | A thin wrapper that turns an external MCP JSON‑RPC tool definition into a `BaseTool`‑compatible object. |
| **Discovery Tools** | Two built‑in tools – `list_mcp_catalog` and `load_mcp_tool` – that let an agent explore and load MCP tools at runtime. |

---

## 2. Implementing a Custom Tool

1. **Create a subclass of `BaseTool`** in `agentic_core/tools` (or any importable module).
2. **Define the `name` property** – the identifier the LLM will call.
3. **Provide a JSON schema** via the `schema` property. The schema follows the OpenAI function‑calling format:
   ```python
   @property
   def schema(self) -> dict:
       return {
           "type": "function",
           "function": {
               "name": "my_tool",
               "description": "Brief description for the LLM.",
               "parameters": {
                   "type": "object",
                   "properties": {
                       "path": {"type": "string", "description": "File path to read."}
                   },
                   "required": ["path"]
               }
           }
       }
   ```
4. **Implement `execute(self, args: dict, context: dict) -> str`** – the business logic. `args` contains validated parameters from the LLM, and `context` holds any extra data supplied by the runner (e.g., user session, environment variables). This is super useful for stateful tools.
5. **Optionally override `__init__`** to accept dependencies (database connections, API clients, etc.).

### Example
```python
from agentic_core.tools.base import BaseTool

class EchoTool(BaseTool):
    def __init__(self, prefix: str = ""):
        super().__init__()
        # Setting _name and _schema automatically generates properties `name` and `schema`
        self._name = "echo"
        self._schema = {
            "type": "function",
            "function": {
                "name": "echo",
                "description": "Returns the supplied text unchanged.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to echo."}
                    },
                    "required": ["text"]
                }
            }
        }
        self._prefix = prefix

    def execute(self, args: dict, context: dict) -> str:
        return f"{self._prefix}{args['text']}"
```
---

## 3. Registering Tools with `ToolManager`

### 3.1 Direct Registration (Static Tools)
```python
from agentic_core.tools.manager import ToolManager

from my_tools import EchoTool

manager = ToolManager()
manager.register_tool(EchoTool(prefix="[Echo] "))
```

### 3.2 Registering MCP Tools (Dynamic)
MCP tools are discovered lazily. The manager automatically registers the two discovery tools (`list_mcp_catalog` and `load_mcp_tool`). To load a specific MCP tool:
```python
# LLM calls the discovery tool first
result = manager._plugins["list_mcp_catalog"].execute({}, {})
# Then the LLM decides which tools to load and calls:
load_result = manager._plugins["load_mcp_tool"].execute({"tool_names": ["github_create_issue"]}, {})
```
* Internally `load_mcp_tool` moves the selected `MCPToolAdapter` from the standby registry into the active plugin map.
* After loading, the tool behaves exactly like a static `BaseTool`.

---

## 4. Lifecycle Management

Because `ToolManager` often manages connections to external Model Context Protocol (MCP) servers (which run as separate subprocesses), proper lifecycle management is critical. Failing to shut down these servers can lead to "zombie processes" that continue to consume CPU and memory, or cause port conflicts when you restart your application.

### 4.1 The Recommended Approach: Async Context Manager

The most robust way to use `ToolManager` is as an async context manager. This ensures that `shutdown_mcp()` is called automatically when the block is exited, regardless of whether the code finished successfully or raised an exception.

```python
from agentic_core.tools.manager import ToolManager

async def main():
    # The 'async with' statement handles both setup and teardown
    async with ToolManager(mcp_config_path="mcp.json") as manager:
        try:
            # Your agent logic here (or could just pass this `manager` instance to a `AgentRunner`)
            result = await manager.execute("some_tool", { "arg": "val"})
            print(result)
        except Exception as e:
            print(f"An error occurred: {e}")
            # The context manager will still ensure cleanup happens here

    # At this point, all MCP subprocesses are gracefully terminated
```

**What happens under the hood?**

*   `__aenter__`: Simply returns the manager instance for use.
*   `__aexit__`: Triggers `shutdown_mcp()`, which iterates through all active MCP sessions, closes the JSON-RPC connections, and terminates the associated subprocesses. It also unregisters the `atexit` fallback to avoid redundant cleanup calls.

### 4.2 Manual Lifecycle Management

In some architectures (e.g., if the `ToolManager` is a long-lived singleton in a web server or a GUI application), you might not be able to wrap your entire app in a local `async with` block. In these cases, you must manually trigger the shutdown during your application's teardown phase.

```python
manager = ToolManager(mcp_config_path="mcp.json")

# ... application runs ...

# During app shutdown (e.g., FastAPI on_event("shutdown") or a SIGTERM handler)
await manager.shutdown_mcp()
```

### 4.3 The `atexit` Fallback (Last Resort)

As a safety net, `ToolManager` registers a synchronous `cleanup()` method with Python's `atexit` module during initialization. If the program terminates unexpectedly without `shutdown_mcp()` being called, `cleanup()` attempts to stop the servers.

> **Warning:** `atexit` is less deterministic. Because it runs during the final stages of process exit, the asyncio event loop may already be closed or in a state where it cannot spawn new tasks. This can lead to "loop closed" warnings in your logs or slightly delayed process termination. This is why the context manager is strongly preferred.

### 4.4 Lifecycle Strategy Comparison

| Strategy | Deterministic? | Exception Safe? | Effort | Recommendation |
| :--- | :---: | :---: | :---: | :--- |
| Async Context Manager | ✅ Yes | ✅ Yes | Low | Gold Standard |
| Manual `shutdown_mcp()` | ✅ Yes | ⚠️ Manual | Medium | Use for singletons / long-lived apps |
| `atexit` Fallback | ❌ No | ⚠️ Partial | Zero | Emergency safety net only (enough for most simple apps) |

---

## 5. Execution Flow

1. **Runner creates a `ToolManager`** (optionally with `toolsets`, `mcp_config_path`, etc.).
2. **Discovery tools are injected** if `enable_mcp_discovery` is `True`.
3. **When the LLM decides to call a tool**, the runner looks up `manager._plugins[tool_name]` and invokes `execute(args, context)`.
4. **Result string** is returned to the LLM and can be fed back into the conversation.
5. **Background cleanup** – `ToolManager.cleanup` is registered with `atexit` to shut down any MCP client threads.

---

## 6. Advanced Features

### 6.1 Toolsets & Prompts
* `ToolManager` accepts a `toolsets` mapping where each key is a logical group (e.g., `"file_ops"`) and the value is a list of tool names.
* Optional `prompt` strings can be attached to a toolset; the runner can prepend them to the LLM prompt to give context about the available capabilities.

### 6.2 Extra Context
* `extra_context` passed to the manager is merged into the `context` argument of every tool's `execute` call. Use it to share session state, authentication tokens, or temporary variables.

### 6.3 Security – Path Validation
* `BaseTool._is_allowed_path` provides a safe‑check for file‑system tools. It ensures the supplied path resolves inside a given base directory and rejects absolute paths or null bytes.

---

## 7. Real usage Example: static, stateful tool, and registration with custom toolsets
```python
from agentic_core.tools.manager import ToolManager
from agentic_core.tools.base import BaseTool

# ==== Tools ====
# Simple plain tool
class UpperCaseTool(BaseTool):
    def __init__(self):
        self._name = "uppercase"
        self._schema = {
            "type": "function",
            "function": {
                "name": "uppercase",
                "description": "Converts text to upper case.",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"]
                }
            }
        }

    def execute(self, args, context)
        return args["text"].upper()

# More complex tool
class DocumentEditorTool(BaseTool):                                                                                                                         
    total_global_edits = 0                                                                                                                                  
                                                                                                                                                            
    def __init__(self):                                                                                                                                     
        self._name = "document_editor"                                                                                                                      
        self.content = []                                                                                                                                   
        self.cursor_line = 0                                                                                                                                
                                                                                                                                                            
        self._schema = {                                                                                                                                    
            "type": "function",                                                                                                                             
            "function": {                                                                                                                                   
                "name": "document_editor",                                                                                                                  
                "description": "Appends text to a document or reads it back.",                                                                              
                "parameters": {                                                                                                                             
                    "type": "object",                                                                                                                       
                    "properties": {                                                                                                                         
                        "action": {"type": "string", "enum": ["write", "read", "clear"]},                                                                   
                        "text": {"type": "string", "description": "Text to write (required for 'write' action)"}                                            
                    },                                                                                                                                      
                    "required": ["action"]                                                                                                                  
                }                                                                                                                                           
            }                                                                                                                                               
        }                                                                                                                                                   
                                                                                                                                                            
    def execute(self, args, context):                     
        # access global context like this                                                                                                  
        user_id = context.get("user_id", "anonymous")                                                                                                       
        session_id = context.get("session_id", "unknown_session")                                                                                           
                                                                                                                                                            
        action = args.get("action")                                                                                                                         
                                                                                                                                                            
        if action == "write":                                                                                                                               
            text = args.get("text", "")                                                                                                                     
            # use stateful attributes
            self.content.append(text)                                                                                                                       
            self.cursor_line += 1                                                                                                                           
                                                                                                                                                            
            # class variables can also!
            DocumentEditorTool.total_global_edits += 1                                                                                                      
                                                                                                                                                            
            return f"[User: {user_id}] Added text to line {self.cursor_line}. (Global edits: {DocumentEditorTool.total_global_edits})"                      
                                                                                                                                                            
        elif action == "read":                                                                                                                              
            full_text = "\n".join(self.content) if self.content else "Document is empty."                                                                   
            return f"Session {session_id} Document:\n{full_text}"                                                                                           
                                                                                                                                                            
        elif action == "clear":                                                                                                                             
            self.content = []                                                                                                                               
            self.cursor_line = 0                                                                                                                            
            return "Document cleared."                                                                                                                      
                                                                                                                                                            
        return "Invalid action."

# Initialise manager with discovery enabled and even a system prompt
manager = ToolManager(
    toolsets={
        "simple_plain_toolset": ["uppercase"] # simple toolset
        # you can specify a prompt to be dynamically injected for this toolset. This is perfect for toolset auto-routing systems.
        "complicated_toolset": { 
            "tools": ["uppercase", "document_editor"],
            "prompt": "System to Agent. You can use `document_editor` to read into the secret of universe. Don't use `uppercase` unless you want to go BOOM!"
        }
    },
    mcp_config_path="examples/mcp_config.json",
    enable_mcp_discovery=True
)

# Register static tool
manager.register_tool(UpperCaseTool())
# For stateful tool, it is better to initalize the tool instance first then inject into necessary places to maintain internal states
doc_edit_tool = DocumentEditorTool()
manager.register_tool(doc_edit_tool)

# If you forgot to register `toolset`, use `manager.add_toolset(name, tools, prompt)`
manager.add_toolset("other_plain_toolset", ["some_random_tool1", "random_tool2"])
```
---

## 7. Reference API Summary

| Class / Function | Key Methods / Attributes |
|------------------|--------------------------|
| `BaseTool` | `name`, `schema`, `execute(args, context)`, `_is_allowed_path` |
| `ToolManager` | `register_tool(tool_instance, load_mcp=False)`, `cleanup()`, `ensure_mcp_initialized()`, internal `_register_discovery_tools()` |
| `ListMCPTools` (inherits `BaseTool`) | `execute(args, context)` – returns a human‑readable catalog |
| `LoadMCPTool` (inherits `BaseTool`) | `execute(args, context)` – loads selected MCP tools |
| `MCPToolAdapter` | Wraps external MCP definitions; implements `execute` by forwarding JSON‑RPC calls |

---

## 8. Testing & Debugging
* Unit tests located in `tests/test_mcp.py` and `tests/test_core.py` cover registration, discovery, and loading.
* Use the `debug.log` file to inspect MCP initialization messages (`logger.info`/`debug`).
* When developing a new tool, run `pytest -k <tool_name>` to ensure schema compliance.

---
