from __future__ import annotations
import logging
from typing import Any, Tuple, TYPE_CHECKING
from dataclasses import dataclass

from agentic_core.interfaces import IterationLimitReachedError
from agentic_core.utils import convert_exception_to_message

if TYPE_CHECKING:
    from agentic_core.engines import DAGEventObserver
    from agentic_core.tools import ToolManager
    from agentic_core.llm_providers import ILLMClient

from agentic_core.engines.dag_engine import DAGAgentRunner
from agentic_core.engines.engine import AgentRunner, RunnerConfig
from agentic_core.memory.manager import MemoryManager
from agentic_core.tools.base import BaseTool

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
            "description": (
                        "Decompose a complex goal into a set of dependent sub-tasks. "
                        "Use this when a task requires multiple steps that can be done in parallel "
                        "or have strict sequential dependencies. "
                        "CRITICAL KNOWLEDGE FLOW: Context and outputs from a node are ONLY passed to its "
                        "direct downstream dependents. Parallel branches are completely isolated and share NO knowledge. "
                        "Design your nodes and edges accordingly."
                    ),
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
                                        "tools": {"type": "array", "items": {"type": "string"}, "description": "List of specific tools to grant this sub-agent. Tools must be loaded into your context already. By default, sub-agents do not get any tools and cannot load new tools."},
                                        "max_retries": {"type": "integer", "description": "Number of times to retry the entire task if it encounters an API or network error."},
                                        "max_iterations": {"type": "integer", "description": "Maximum number of tool-calling iterations the sub-agent is allowed to take. Default is 10."}
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
        llm_client: ILLMClient = context.get("llm_client")
        tools_manager: ToolManager = context.get("tools_manager")
        observer: DAGEventObserver = context.get("subagent_observer")

        if not llm_client or not tools_manager:
            return "Error: Sub-agent spawning requires 'llm_client' and 'tools_manager' in the context."

        plan_data = args.get("plan", {})

        nodes_config = plan_data.get("nodes", {})
        edges_raw = plan_data.get("edges", [])

        if not nodes_config or not isinstance(nodes_config, dict):
            return "Validation Error: 'nodes' must be a non-empty dictionary mapping node IDs to their configurations."

        if not isinstance(edges_raw, list):
            return "Validation Error: 'edges' must be a list of dependency pairs."
        
        edges = []
        for edge in edges_raw:
            if not isinstance(edge, (list, tuple)) or len(edge) != 2:
                return f"Validation Error: Invalid edge format {edge}. Each edge must be exactly a pair of node IDs like ['from_node', 'to_node']."
            
            u, v = edge
            if u not in nodes_config:
                return f"Validation Error: Edge references unknown source node '{u}'. Node must exist in 'nodes'."
            if v not in nodes_config:
                return f"Validation Error: Edge references unknown target node '{v}'. Node must exist in 'nodes'."

        # Convert edges to tuples
        edges = [tuple(edge) for edge in edges_raw if isinstance(edge, list) and len(edge) == 2]

        nodes_def = {}
        for node_id, cfg in nodes_config.items():
            # Isolation: each sub-agent gets its own memory
            node_memory = MemoryManager()

            runner = AgentRunner(llm_client, tools_manager, node_memory)

            # Sub-agents can be granted specific tools.
            config = RunnerConfig(mcp_use_loaded_tools=False, mcp_enable_discovery=False)
            requested_tools = cfg.get("tools")
            if requested_tools:
                config.tools = [
                    schema for schema in tools_manager.tool_schemas 
                    if schema["function"]["name"] in requested_tools
                ]

                if len(config.tools) != len(requested_tools):
                    return f"Validation Error: One or more requested tools for node {node_id} are not available. This is likely because you are trying to pass tools you do not have yourself."

            prompt = cfg.get("prompt", "")
            max_retries = cfg.get("max_retries", 0)
            config.max_iterations = cfg.get("max_iterations", 10)

            nodes_def[node_id] = (runner, config, prompt, max_retries)

        try:
            dag_runner = DAGAgentRunner(nodes_def, edges, observer=observer)
            result = await dag_runner.execute()

            if result.error:
                return f"DAG Execution Error: {convert_exception_to_message(result.error)}"

            # Summarize results for the parent agent
            summary = []
            for node_id, node in dag_runner.nodes.items():
                if node.state.name == "SUCCESS" and node.result:
                    res_text = node.result.text if hasattr(node.result, 'text') else str(node.result)
                else:
                    res_text = f"Execution halted: {node.error}"
                    if isinstance(node.error, IterationLimitReachedError):
                        res_text += " (Hint: You can retry spawning this subagent by providing a higher 'max_iterations' in the node config)"
                status = node.state.name
                summary.append(f"[{status}] Task {node_id}: {res_text}")

            return "Sub-agent execution results:\n" + "\n".join(summary)

        except Exception as e:
            logger.exception("Failed to execute sub-agent plan")
            return f"Unexpected error during sub-agent orchestration: {str(e)}"
