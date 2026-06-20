# agentic_core

`agentic_core` is a deterministic, asynchronous execution engine and orchestration framework for Large Language Model (LLM) agents. It provides native support for the Model Context Protocol (MCP), arbitrary cyclic graph execution topologies, and extensible state persistence.

## 1. Installation

```bash
pip install callai-agentic_core
```

Install with specific provider dependencies:
```bash
pip install "callai-agentic_core[openai]"
pip install "callai-agentic_core[anthropic]"
pip install "callai-agentic_core[ollama]"
pip install "callai-agentic_core[all]" # Installs all providers and MCP support
```

## 2. Core API: `AgentBuilder`

The framework utilizes a fluent builder pattern for agent instantiation, avoiding constructor parameter bloat.

```python
import asyncio
from agentic_core.agents import AgentBuilder
from agentic_core.handlers import PrintHandler

async def main():
    # 1. Instantiate the agent via AgentBuilder
    agent = AgentBuilder() \
        .with_provider_openai(api_key="sk-...", model="gpt-4o") \
        .with_system_prompt("You are a strict data extraction tool.") \
        .with_handler(PrintHandler()) \
        .build()

    # 2. Execute a turn
    result = await agent.run_turn(
        user_input="Extract entities from: John Doe works at Acme Corp."
    )
    
    # result is an AgentResponse object
    print(result.text)
    print(result.usage)

asyncio.run(main())
```

### Provider Support
- `with_provider_openai(api_key: str, model: str, base_url: str = None)`: Supports OpenAI and OpenAI-compatible endpoints (e.g., vLLM, LMStudio).
- `with_provider_anthropic(api_key: str, model: str)`: Supports Anthropic Claude models.
- `with_provider_ollama(model: str, base_url: str)`: Supports local Ollama instances.

## 3. Tool Management and MCP Integration

Tools are registered via the `AgentBuilder` or directly into the initialized `AgentRunner.tools` (a `ToolManager` instance).

```python
# Static Tool Registration
agent = AgentBuilder() \
    .with_provider_openai(api_key="sk-...") \
    .with_tools([MyCustomTool()]) \
    .build()

# Dynamic MCP Configuration
agent = AgentBuilder() \
    .with_provider_openai(api_key="sk-...") \
    .with_mcp(
        config_path="mcp_config.json",
        enable_discovery=True,
        extra_env={"GITHUB_TOKEN": "ghp_..."}
    ) \
    .build()
```

Reference `docs/TOOLS_GUIDE.md` for `BaseTool` implementation and `docs/MCP_CONFIG_GUIDE.md` for MCP runtime behaviors.

## 4. Cyclic Graph Orchestration (`GraphAgentRunner`)

For multi-agent workflows, the framework uses a generalized Cyclic Graph orchestrator (`GraphAgentRunner`). Unlike rigid DAGs, it natively supports loopbacks, recursive consensus structures, conditional routing, state reducers, and a deeply isolated state bus.

```python
from agentic_core.engines.dag_engine import GraphAgentRunner, DAGTask
from agentic_core.models import AgentResponse

# 1. Define nodes using DAGTask
nodes_def = {
    "node_a": DAGTask(runner=agent, prompt="Task A", max_retries=3),
    "node_b": DAGTask(runner=agent, prompt="Task B")
}

# 2. Define edges (Dependencies and routing logic)
# Forward edges and back-edges (loops) are automatically resolved via Petri Net semantics
def evaluate_condition(resp: AgentResponse, state: dict) -> bool:
    return "SUCCESS" in resp.text

edges = [
    ("node_a", "node_b", evaluate_condition),
    ("node_b", "node_a") # Cyclic loopback natively supported!
]

# 3. Execute
graph = GraphAgentRunner(nodes_def, edges, shared_state={"key": "value"})
response = await graph.execute()
```

Reference `docs/DAG_ENGINE_GUIDE.md` for loopback evaluation, context assemblers, and state injection mechanics.

## 5. Streaming Execution

The `AgentRunner.stream_turn` method yields `StreamEvent` objects for real-time observability.

```python
from agentic_core.models import StreamEventType

async for event in agent.stream_turn("Query string"):
    if event.type == StreamEventType.TEXT:
        print(event.content, end="")
    elif event.type == StreamEventType.TOOL_CALL:
        print(event.content['function']['name'])
```

## 6. Project Structure

```text
callai-agentic_core/
├── packages/                # Extensible Modules
│   ├── rag/                 # Vector stores and embedding providers
│   ├── shell/               # Local and Docker-isolated shell execution
│   └── skills/              # LLM-based trace extraction and skill synthesis
└── agentic_core/
    ├── engines/             # Execution loops (AgentRunner, DAGAgentRunner)
    ├── agents/              # Builder patterns
    ├── llm_providers/       # Provider adapters
    ├── memory/              # Context truncation and management
    ├── handlers/            # Event emission and telemetry
    └── tools/               # ToolManager and MCP client lifecycles
```