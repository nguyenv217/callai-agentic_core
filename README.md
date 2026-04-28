# Agentic Core

- **If you want an autonomous agent immediately**: Got an API key? You can start an agent in one line. No classes, no protocols, no boilerplate.
- **If you want to build production-ready multi-agent system with clear, structured interfaces**: That's even better. This engine gives you:
  - native MCP integration with dynamic load and discovery options, 
  - customize client wrapper option for smart routing strategies, 
  - smart-default or your own domain-specific memory truncation strategies, 
  - clear decision actions at each step of the agent's plan, 
  - and a simple but extendable base tool class that can be used to implement arbitrarily complex workflows.

**Example Applications**: Tool-calling Assistant Bots, DAG-based Agent Swarms (via `DAGAgentRunner` and agent-spawning tools), Legal Consultant Bot with RAG-based tools (via `RAG tool suite` installed with extra [rag] (see [docs/RAG_TOOLS.md](docs/RAG_TOOLS.md)), and many more.


## Table of Contents

1. [Quick Start](#quick-start)
2. [The Simple `chat()` Function](#the-simple-chat-function)
3. [Tooling system: Bring-your-own-tools & MCP integration (GitHub, Filesystem, etc.)](#using-tools-implementing-basetool-or-integrate-mcp-servers)
4. [Agent creation (Advanced) & Memory Management](#creating-your-own-agent-advanced--custom-memory-management-strategy)
5. [Configuration Options](#configuration-options)
6. [Project Structure](#project-structure)
7. [Troubleshooting](#troubleshooting)
8. [Notes on security](#security--production-readiness)
---

## QUICK START


### Installation
1. Install the package with pip:
```bash
pip install callai-agentic_core
```

2. Clone from this repo:
```bash
git clone https://github.com/nguyenv217/callai-agentic_core 
cd callai-agentic_core
pip install ".[all]"
```

*Note: To install only specific providers, use `pip install ".[openai]"` or `pip install ".[anthropic]"`.*

### If you have an OpenAI key:

```python
import asyncio
from agentic_core.agents import chat

async def main():
    result = await chat(
        message="What's the weather in Tokyo?",
        provider="openai",
        api_key="sk-..."  # Your OpenAI key
    )
    print(result.text)

asyncio.run(main())
```

### If you are using an OpenAI-compatible API (e.g. HuggingFace, vLLM):

```python
import asyncio
from agentic_core.agents import chat

async def main():
    result = await chat(
        message="Hello!",
        provider="openai",
        api_key="your_hf_token",
        base_url="https://router.huggingface.co/v1",
        model="meta-llama/Llama-3.1-8B-Instruct"
    )
    print(result.text)

asyncio.run(main())
```

### If you have an Anthropic key:

```python
result = await chat(
    message="What's the weather in Tokyo?",
    provider="anthropic", 
    api_key="sk-ant-..."  # Your Anthropic key
)
print(result.text)
```

### If you have Ollama (local, no key needed):

```python
result = await chat(
    message="What's the weather in Tokyo?",
    provider="ollama"  # No API key needed!
)
print(result.text)
```

**That's it!** Want to see what's happening? Add `verbose=True`.

---


## The Simple `chat()` Function

The easiest way to use `agentic_core`:

```python
from agentic_core.agents import chat
from agentic_core.config import RunnerConfig

# Simple usage
result = await chat(
    message="What's the weather in Tokyo?",
    provider="openai",
    api_key="sk-...",
    verbose=True
)


# Persistent session usage (preserves memory and MCP connections)
result = await chat(
    message="Remember my name is Alice",
    provider="openai",
    api_key="sk-...",
    session_id="user_session_123"
)
print(result.text)


# Advanced usage with RunnerConfig
result = await chat(
    message="Analyze this repo",
    provider="openai",
    api_key="sk-...",
    config=RunnerConfig(
        max_iterations=10,
        system_prompt="You are a senior software engineer.",
        mcp_enable_discovery=True
    ),
    mcp_config_path="mcp.json"
)
```

**Returns:** An `AgentResponse` object containing `text`, `reasoning`, `tool_calls`, `usage`, and `error` fields.

---

## Using Tools (Implementing `BaseTool` or integrate MCP servers)

### Custom Tools via `BaseTool`

The `BaseTool` base class give you robust but structured method to create and use tools in your project. It is designed to extend functionality with a consistent framework (e.g, designing agent swarms with `spawn_agents`, `create_task`, etc.).
If you want to write your own Python tools, inherit from `BaseTool`, register it with `ToolManager`, before specifying it in your agent workflow via `RunnerConfig`:

```python
from agentic_core.tools.base import BaseTool
from agentic_core.tools.manager import ToolManager

class UpperCaseTool(BaseTool):
    name = "uppercase"
    schema = {
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

    def execute(self, args: dict, context: dict) -> str:
        return args["text"].upper()

# Register and use
tools = ToolManager()
uppercase_tool = UpperCaseTool()
tools.register_tool(uppercase_tool)

# === More advanced usage: `toolset` ===
tools = ToolManager(
    toolsets = {
        "my_custom_toolset": {
            "tools": ["uppercase"],
            "prompt": "This prompt is dynamically injected when toolset=my_custom_toolset"
        },
        "my_other_toolset": ["some_other_tool", "other_tool2", ...] # Or just a list if no custom prompt is needed
        },
    )
    
# Then pass to `RunnerConfig`
config = RunnerConfig(
    system_prompt = "My base sytem prompt."
    tools = [uppercase_tool.schema, ...] # only inject these tools, this parameter takes priority over `toolset`
    toolset = "my_custom_toolset" # or a toolset for a specific known set of specific tools with a custom prompt
)
```

**RECOMMENDED**: For a detailed guide on building and registering tools, see [docs/TOOLS_GUIDE.md](docs/TOOLS_GUIDE.md).

---

### MCP Tools (Dynamic Integration)

> Note: Many MCP servers require Node.js installed on your system to run npx commands. Do check with the your specified servers' documentation.

MCP (Model Context Protocol) lets your agent use external tools via standardized protocols. 
`agentic_core` natively supports them and even gives you easy means to dynamically discover and load these tools in order to save on your token usage! Here's how to use it:

### Step 1: Create `mcp.json`

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"], 
      "env": { "GITHUB_PERSONAL_ACCESS_TOKEN": "your-token" } 
    },
    "filesystem": {
      "command": "npx", 
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/your/project"]
    }
  }
}
```

Alternatively, `"env":{"TOKEN_ENV_NAME":"${YOUR_TOKEN}"}` works as well assuming environment variable `YOUR_TOKEN` exists.  
Refer to `examples/mcp_config.json` for a few no-config, plug-and-play example servers.  

### Step 2: Use it in chat()
 
 ```python
 result = await chat(
     message="Create a new issue in my repo about the bug",
     provider="openai",
     api_key="sk-...",
     mcp_config_path="mcp.json"  # <-- This enables MCP!
 )
 ```
 
 ### Step 3: Programmatic MCP Config (No JSON file needed)
 
 If you prefer not to use a JSON file, you can add MCP servers directly to your `ToolManager` instance.
 
 ```python
 from agentic_core import ToolManager
 
 tools = ToolManager()
 tools.add_mcp_server(
     server_name="everything",
     command="npx",
     args=["-y", "@modelcontextprotocol/server-everything"]
 )
 
 # Now use this ToolManager with your agent
 from agentic_core import AgentRunner
 from agentic_core.agents import OpenAILLM
 
 agent = AgentRunner(
     llm_client=OpenAILLM(api_key="sk-..."),
     tools=tools
 )
 ```

**How it works:**
1. The agent sees available MCP tools via `list_mcp_catalog`
2. It loads the tools it needs via `load_mcp_tool`
3. It uses them to answer your question

**Advanced**   
By default, `agentic-core` injects Meta-Tools into the agent's context to enable autonomous tool management:

* `list_mcp_catalog`: The agent can browse available servers and their tool definitions.
* `load_mcp_tool`: The agent can "install" a tool into its active session on-the-fly.

This "Lazy Loading" approach keeps your prompt context small and saves tokens by only loading the specific tools the agent decides it needs for the task.

If you already know which tools the agent will need, you can bypass the discovery turns entirely using `mcp_preload_tools` and setting `enable_mcp_discovery=False` in your`RunnerConfig`. This is highly recommended for production environments to reduce latency and costs.

```python
from agentic_core.config import RunnerConfig

config = RunnerConfig(
    mcp_active_servers=["sqlite"],       # Initialize these servers immediately
    mcp_preload_tools=["sqlite_query"]   # Move these tools to 'Active' before Turn 1. NOTE that you must prefix the tool name with server name 
)

# The agent starts with 'sqlite_query' already지 set in its toolkit
agent.chat("How many users in the database?", config=config)
```

Please see this more detailed [explanation](docs/MCP_CONFIG_GUIDE.md) in how to configuring `RunnerConfig` to fully realize the MCP protocol in your project.

---

## Creating Your Own Agent (Advanced) & Custom Memory Management Strategy

If you need more control, you can create agents manually:

```python
from agentic_core.engines import AgentRunner 
from agentic_core.memory import MemoryManager
from agentic_core.tools import ToolManager
from agentic_core.llm_providers import OpenAILLM
from agentic_core.observers import PrintObserver

# 1. Create components
llm = OpenAILLM(api_key="sk-...", model="gpt-4o")
memory = MemoryManager()
memory.set_system_prompt("You are a helpful coding assistant.")
tools = ToolManager()

# 2. Create agent
agent = AgentRunner(llm_client=llm, tools=tools, memory=memory)

# 3. Run!
result = await agent.run_turn(
    user_input="Hello!",
    observer=PrintObserver()
)
print(result.text)
```

### Memory Management & Context Truncation

To prevent token overflow in long conversations, `MemoryManager` supports configurable truncation. By default, it uses a `DefaultTruncationStrategy` that uses intelligently prunes tool outputs and long text before deleting entire messages.

```python
from agentic_core.memory.manager import MemoryManager
from agentic_core.memory.strategies import DefaultTruncationStrategy

# Custom strategy: lower thresholds for aggressive pruning
strategy = DefaultTruncationStrategy(tool_threshold=1000, text_threshold=500)

memory = MemoryManager(
    max_chars=4000, 
    strategy=strategy
)
```
You can also implement your own truncation logic by inheriting from the `TruncationStrategy` interface.

### Available LLM Adapters

```python
from agentic_core.llm_providers import (
    OpenAILLM,      # OpenAI-compatible endpoints
    AnthropicLLM,   # Anthropic Claude
    OllamaLLM,      # Local Ollama
)
```

### Available Observers

```python
from agentic_core.observers import (
    SilentObserver,  # Silent, does nothing
    PrintObserver,    # Prints everything (great for debugging)
)
```

---

## Configuration Options

### chat() parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `message` | str | (required) | Your message to the agent |
| `base_url` | str | provider default | Custom API endpoint (e.g. for Local LLMs or Proxies) |
| `provider` | str | "openai" | "openai", "anthropic", or "ollama" |
| `api_key` | str | None | Required for openai/anthropic |
| `model` | str | provider default | Model name |
| `system_prompt` | str | "You are a helpful assistant." | Agent persona |
| `mcp_config_path` | str | None | Path to MCP config |
| `verbose` | bool | False | Print all events |
| `temperature` | float | provider default | LLM creativity |
| `max_tokens` | int | provider default | Max response size |
| `config` | `RunnerConfig` | None | Custom execution loop configuration |
### ToolManager with MCP

```python
from agentic_core.tools import ToolManager

tools = ToolManager(
    mcp_config_path="mcp.json",  # Enable MCP: str | Path
    toolsets={
        "local": ["read_file", "write_file"],
        "web": ["web_search"], 
        ...
    },
    enable_mcp_discovery=True, # Preload MCP discovery tools (2 tools)
    extra_env={"DATABASE": "..."} # Extra environment variables for MCP server intialization (pass only what you need)
)
```

---

## Structure
 
 ```
 agentic_core/
 ├── __init__.py              
 ├── engine.py                # AgentRunner (The execution loop)
 ├── agents/                  # Agent construction and orchestration
 │   └── builder.py           # Agent builder logic
 ├── llm_providers/           # LLM Adapters (OpenAI, Anthropic, Ollama)
 │   ├── base.py              # Base LLM interface
 │   ├── openai.py            # OpenAI/Compatible provider
 │   ├── anthropic.py         # Anthropic Claude provider
 │   └── ollama.py            # Local Ollama provider
 ├── memory/                  # Context and state management
 │   ├── manager.py           # MemoryManager
 │   └── strategies.py        # Truncation strategies
 ├── observers/               # Event logging and observation
 │   ├── base.py              # Base Observer
 │   └── standard.py          # Default/Print observers
 ├── tools/                   # Tooling system
 │   ├── base.py              # `BaseTool`, schemas, etc.
 │   ├── mcp.py               # MCP server management
 │   ├── 🔍 rag/             # Custom RAG-based tool suite, supporting your custom backends, default options: ChromaDB, Sqlite. see 'docs/RAG_TOOLS.md'
 │   └── manager.py           # `ToolManager` class
 └── interfaces/              # Type definitions and Protocols
     ├── llm.py
     └── events.py
 ```

---

## Troubleshooting

### "openai not found"

```bash
pip install openai
# or
pip install anthropic
# or  
pip install ollama
```

### "API key invalid"

Check your key is correct and has credits.

### "MCP not working"

1. Make sure `mcp_config_path` points to a valid JSON file
2. Install the MCP servers: `npx -y @modelcontextprotocol/server-github`
3. Check the logs with `verbose=True`

### "Model not found" (Ollama)

Make sure Ollama is running:
```bash
ollama serve
ollama pull llama3.1
```

---

## Quick Reference

```python
# Simplest possible usage
from agentic_core.agents import chat
await chat("Hello", provider="ollama")

# With OpenAI
await chat("Hello", provider="openai", api_key="sk-...")

# With MCP tools
await chat("Check my GitHub", provider="openai", api_key="sk-...", mcp_config_path="mcp.json")

# Verbose mode (see everything)
await chat("Hello", provider="ollama", verbose=True)
```

**That's it!** Start building agents in seconds. 🚀


## Security & Production Readiness

This tool is designed to be as lightweight and robust as possible. While `agentic-core` safely handles execution, developers must secure the deployment environment. Below are some important considerations:

### 1. Remote Code Execution via MCP Config:
The `mcp_config.json` dictates exactly which terminal commands your system will run (via the `command` and `args` fields). **Never allow end-users to upload, modify, or provide their own `mcp_config.json`.** This file must remain strictly server-side.

### 2. Prompt Injection to Tool Execution: 
If an agent is given a tool that reads external data (like fetching a webpage or reading a user-submitted file), a malicious payload in that data can instruct the LLM to execute other available tools. The agent could be hijacked into executing destructive actions (e.g., via a GitHub or filesystem MCP) without human oversight. SO, do use `AgentEventObserver.on_tool_start()` method for granular control over agent's actions. 

### 3. Denial of Service via Payload Serialization:
Agents interacting with APIs that return massive, deeply nested JSON payloads may experience performance degradation during the engine's double-serialization checks. To mitigate this, limit the scope of the data your tools are allowed to fetch.

**MITIGATIONS:**  
* The tool strictly requires MCP configuration via `RunnerConfig` to be valid and well-formed to avoid malicious runtime tool injection.  
* Utilize `ToolExecutionController.on_prompt_respond()` (blocks, prompt the user for feedback) and `ToolExecutionController.on_prompt_confirmation()` (blocks, prompt the user to confirm (y/n) with event hooks) for control **during tool execution**.
* `AgentEventObserver` implementing `on_tool_start()` provides means to enforce human validation before assembling tool coroutine pool and executing, with different levels of control (use `ToolStartDecision`).
* Write your system prompt carefully and choose your MCP servers wisely.
* Finally, this tool is all about robustness - it's meant to be a lightweight engine for agentic applications. Hence, it is at the developer responsibility to enforce security measures against the above risks.
