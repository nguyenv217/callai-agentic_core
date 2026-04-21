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
* The tool becomes immediately available to the LLM.
* Its schema is added to `manager.tools_schema` – the list sent to the LLM on each turn.

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

## 4. Execution Flow

1. **Runner creates a `ToolManager`** (optionally with `toolsets`, `mcp_config_path`, etc.).
2. **Discovery tools are injected** if `enable_mcp_discovery` is `True`.
3. **When the LLM decides to call a tool**, the runner looks up `manager._plugins[tool_name]` and invokes `execute(args, context)`.
4. **Result string** is returned to the LLM and can be fed back into the conversation.
5. **Background cleanup** – `ToolManager.cleanup` is registered with `atexit` to shut down any MCP client threads.

---

## 5. Advanced Features

### 5.1 Toolsets & Prompts
* `ToolManager` accepts a `toolsets` mapping where each key is a logical group (e.g., `"file_ops"`) and the value is a list of tool names.
* Optional `prompt` strings can be attached to a toolset; the runner can prepend them to the LLM prompt to give context about the available capabilities.

### 5.2 Extra Context
* `extra_context` passed to the manager is merged into the `context` argument of every tool's `execute` call. Use it to share session state, authentication tokens, or temporary variables.

### 5.3 Security – Path Validation
* `BaseTool._is_allowed_path` provides a safe‑check for file‑system tools. It ensures the supplied path resolves inside a given base directory and rejects absolute paths or null bytes.

---

## 6. Real usage Example: static, stateful tool, and registration with custom toolsets
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
# For stateful tool, it is better to initalize the tool instance first then inject to necessary places to maintain internal state attributes
doc_edit_tool = DocumentEditorTool()
manager.register_tool(doc_edit_tool)

# If you forgot to register `toolset`, use `tool_manager.add_toolset(name, tools, prompt)`
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