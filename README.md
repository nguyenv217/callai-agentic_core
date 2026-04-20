# Agentic Core
**TL;DR**: Got an API key? You can start an agent in one line. No classes, no protocols, no boilerplate.

## Table of Contents

1. [Quick Start](#the-30-second-quick-start)
2. [The Simple `chat()` Function](#the-simple-chat-function)
3. [MCP integration (GitHub, Filesystem, etc.)](#using-mcp-tools-github-filesystem-etc)
4. [Agent creation (Advanced)](#creating-your-own-agent-advanced)
5. [Configuration Options](#configuration-options)
6. [Project Structure](#project-structure)
7. [Troubleshooting](#troubleshooting)
8. [Notes on security](#security--production-readiness-️)
---

## QUICK START


### Installation
```bash
git clone https://github.com/nguyenv217/callai-agentic_core # clone this repo

pip install .
```
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
    print(result)

asyncio.run(main())
```

### If you have an Anthropic key:

```python
result = await chat(
    message="What's the weather in Tokyo?",
    provider="anthropic", 
    api_key="sk-ant-..."  # Your Anthropic key
)
print(result)
```

### If you have Ollama (local, no key needed):

```python
result = await chat(
    message="What's the weather in Tokyo?",
    provider="ollama"  # No API key needed!
)
print(result)
```

**That's it!** Want to see what's happening? Add `verbose=True`.

---


## The Simple `chat()` Function

The easiest way to use Agentic Core:

```python
from agentic_core.agents import chat

result = await chat(
    model="gpt-4o",           # Optional, provider has good defaults
    base_url="https://...",      # Optional, for OpenAI-compatible providers
    system_prompt="You are...",  # Optional, defaults to helpful assistant
    mcp_config_path="mcp.json",  # Optional, for MCP tools
    verbose=False,            # Set True to see all events
    temperature=0.7,          # Optional, LLM parameters
)
```

**Returns:** The agent's text response as a string.

---

## Using MCP Tools (GitHub, Filesystem, etc.)

> Note: MCP tools require Node.js installed on your system to run npx commands

MCP (Model Context Protocol) lets your agent use external tools. Here's how:

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
     mcp_config_path="path/to/mcp.json"  # <-- This enables MCP!
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
     tool_manager=tools
 )
 ```

**How it works:**
1. The agent sees available MCP tools via `list_mcp_catalog`
2. It loads the tools it needs via `load_mcp_tool`
3. It uses them to answer your question

---

## Creating Your Own Agent (Advanced)

If you need more control, you can create agents manually:

```python
import asyncio
from agentic_core import AgentRunner, MemoryManager, ToolManager
from agentic_core.agents import OpenAILLM, PrintObserver

# 1. Create components
llm = OpenAILLM(api_key="sk-...", model="gpt-4o")
memory = MemoryManager()
memory.set_system_prompt("You are a helpful coding assistant.")
tools = ToolManager()

# 2. Create agent
agent = AgentRunner(llm_client=llm, tool_manager=tools, memory=memory)

# 3. Run!
result = await agent.run_turn(
    user_input="Hello!",
    observer=PrintObserver()  # Prints all events
)
print(result)
```

### Available LLM Adapters

```python
from agentic_core.agents import (
    OpenAILLM,      # OpenAI-compatible endpoints
    AnthropicLLM,   # Anthropic Claude
    OllamaLLM,      # Local Ollama
)
```

### Available Observers

```python
from agentic_core.agents import (
    DefaultObserver,  # Silent, does nothing
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
| `model` | str | provider default | Model to use |
| `system_prompt` | str | "You are a helpful assistant." | Agent persona |
| `mcp_config_path` | str | None | Path to MCP config |
| `verbose` | bool | False | Print all events |
| `temperature` | float | provider default | LLM creativity |
| `max_tokens` | int | provider default | Max response size |

### ToolManager with MCP

```python
from agentic_core import ToolManager

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

## Project Structure
 
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
 │   └── manager.py           # MemoryManager
 ├── tools/                   # Tooling and MCP integration
 │   ├── base.py              # BaseTool (for custom tools)
 │   ├── manager.py           # ToolManager
 │   └── mcp.py               # MCP Protocol support
 ├── observers/               # Event logging and observation
 │   ├── base.py              # Base Observer
 │   └── standard.py          # Default/Print observers
 └── interfaces/              # Type definitions and Protocols
     ├── llm.py
     ├── memory.py
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

While `agentic-core` safely handles execution, developers must secure the deployment environment:

### Remote Code Execution via MCP Config (CRITICAL)
The `mcp_config.json` dictates exactly which terminal commands your system will run (via the `command` and `args` fields). **Never allow end-users to upload, modify, or provide their own `mcp_config.json`.** This file must remain strictly server-side.

### Prompt Injection to Tool Execution: 
If an agent is given an MCP tool that reads external data (like fetching a webpage or reading a user-submitted file), a malicious payload in that data can instruct the LLM to execute other available tools. Because the engine loop automatically runs `await self.tools.execute(...)`, the agent could be hijacked into executing destructive actions (e.g., via a GitHub or filesystem MCP) without human oversight.

### Denial of Service via Payload Serialization
Agents interacting with APIs that return massive, deeply nested JSON payloads may experience performance degradation during the engine's double-serialization checks. To mitigate this, limit the scope of the data your tools are allowed to fetch.

**MITIGATIONS:**  
* Each `BaseTool` class inherits an `is_allowed_path()` method that reliably prevents path traversal. Implementation of tools dealing with local filesystem may use this to improve security. 
* Utilize `ToolExecutionController.on_prompt_respond()` and `ToolExecutionController.on_prompt_confirmation()` to design safe, human-in-the-loop agentic applications.
* `AgentEventObserver` implementing `on_tool_start()` provides means to enforce human validation before assembling tool coroutine pool and executing, with different levels of control (use `ToolStartDecision`).
