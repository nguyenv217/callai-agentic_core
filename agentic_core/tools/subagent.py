import asyncio
import logging
from typing import Any, Tuple, Optional
from dataclasses import dataclass

from agentic_core.engines import DAGEventObserver

from agentic_core.tools import ToolManager
from agentic_core.llm_providers import ILLMClient
from agentic_core.tools.base import BaseTool
from agentic_core.engines.dag_engine import DAGAgentRunner
from agentic_core.engines.engine import AgentRunner, RunnerConfig
from agentic_core.memory.manager import MemoryManager

logger = logging.getLogger(__name__)

@dataclass
class SubAgentPlan:
    nodes: dict[str, dict[str, Any]]
    edges: list[Tuple[str, str]]
        
class SpawnSubAgentsTool(BaseTool):
    """
    Tool that provides agent with subagent spawning capability.
    It uses a SubAgentCoordinator to manage the lifecycle of the sub-agent swarm.
    """
    name = "spawn_subagents"

    schema = {
        "type": "function",
        "function": {
            "name": "spawn_subagents",
            "description": "Decompose a complex goal into a set of dependent sub-tasks. "
                            "Use this when a task requires multiple steps that can be done in parallel "
                            "or have strict sequential dependencies. "
                            "Define nodes (tasks) and edges (dependencies).",
            "parameters": {
                "type": "object",
                "properties": {
                    "plan": {
                        "type": "object",
                        "description": "The task execution graph.",
                        "properties": {
                            "nodes": {
                                "type": "object",
                                "description": "Map of node_id to task config. Example: {'task1': {'prompt': 'Do X', 'tools': ['web_search']}}",
                                "additionalProperties": {
                                    "type": "object",
                                    "properties": {
                                        "prompt": {"type": "string", "description": "Instruction for the sub-agent."},
                                        "tools": {"type": "array", "items": {"type": "string"}, "description": "List of specific tools to grant this sub-agent."},
                                        "max_retries": {"type": "integer", "description": "Number of retries allowed for this task."}
                                    },
                                    "required": ["prompt"]
                                }
                            },
                            "edges": {
                                "type": "array",
                                "items": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "minItems": 2,
                                    "maxItems": 2,
                                    "description": "[from_node, to_node]"
                                },
                                "description": "list of dependencies. Example: [['task1', 'task2']] means task1 must finish before task2 starts."
                            }
                        },
                        "required": ["nodes", "edges"]
                    }
                },
                "required": ["plan"]
            }
        }
    }
    
    
    async def execute(self, args: dict, context: dict) -> str:
        # Resolve dependencies from context
        llm_client = context.get("llm_client")
        tools_manager = context.get("tools_manager")
        parent_memory = context.get("memory_manager")
        observer: DAGEventObserver = context.get("subagent_observer")

        if not llm_client or not tools_manager:
            return "Error: Sub-agent spawning requires 'llm_client' and 'tools_manager' in the context."

        plan_data = args.get("plan", {})

        nodes_config = plan_data.get("nodes", {})
        edges_raw = plan_data.get("edges", [])

        if not nodes_config:
            return "Error: The plan must contain at least one node."

        # Convert edges to tuples
        edges = [tuple(edge) for edge in edges_raw if isinstance(edge, list) and len(edge) == 2]

        nodes_def = {}
        for node_id, cfg in nodes_config.items():
            # Isolation: each sub-agent gets its own memory
            node_memory = MemoryManager()

            # Inherit system prompt if available
            if parent_memory and parent_memory.system_prompt:
                node_memory.set_system_prompt(parent_memory.system_prompt['content'])

            runner = AgentRunner(self.llm_client, self.tools_manager, node_memory)

            # Sub-agents can be granted specific tools.
            # If not specified, they get a default RunnerConfig (usually only MCP tools).
            config = RunnerConfig()
            requested_tools = cfg.get("tools", [])
            if requested_tools:
                config.tools = requested_tools

            prompt = cfg.get("prompt", "")
            max_retries = cfg.get("max_retries", 0)
            config.max_iterations = max_retries + 1 # approximate mapping

            nodes_def[node_id] = (runner, config, prompt, max_retries)

        try:
            dag_runner = DAGAgentRunner(nodes_def, edges, observer=observer)
            result = await dag_runner.execute()

            if result.error:
                return f"DAG Execution Error: {result.error}"

            # Summarize results for the parent agent
            summary = []
            for node_id, node in dag_runner.nodes.items():
                res = node.result
                res_text = res.text if hasattr(res, 'text') else str(res)
                status = node.state.name
                summary.append(f"[{status}] Task {node_id}: {res_text}")

            return "Sub-agent execution results:\n" + "\n".join(summary)

        except Exception as e:
            logger.exception("Failed to execute sub-agent plan")
            return f"Unexpected error during sub-agent orchestration: {str(e)}"
