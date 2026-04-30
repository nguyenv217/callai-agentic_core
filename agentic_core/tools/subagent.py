import asyncio
import logging
from typing import Any, Tuple, Optional
from dataclasses import dataclass

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

class SubAgentCoordinator:
    """
    Handles the orchestration of sub-agents. 
    Decouples the tool interface from the engine implementation.
    """
    def __init__(self, llm_client: ILLMClient, tools_manager: ToolManager):
        self.llm_client = llm_client
        self.tools_manager = tools_manager

    async def run_plan(self, plan_data: dict[str, Any], parent_memory: Optional[MemoryManager] = None) -> str:
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
            config = RunnerConfig()
            prompt = cfg.get("prompt", "")
            max_retries = cfg.get("max_retries", 0)
            
            nodes_def[node_id] = (runner, config, prompt, max_retries)

        try:
            dag_runner = DAGAgentRunner(nodes_def, edges)
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

class SpawnSubAgentsTool(BaseTool):
    """
    Tool that provides agent with subagent spawning capability.
    It uses a SubAgentCoordinator to manage the lifecycle of the sub-agent swarm.
    """
    name = "spawn_subagents"
    
    schema = {
        "type": "function",
        "funtion": {
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
                                "description": "Map of node_id to task config. Example: {'task1': {'prompt': 'Do X'}}",
                                "additionalProperties": {
                                    "type": "object",
                                    "properties": {
                                        "prompt": {"type": "string", "description": "Instruction for the sub-agent."},
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

    def __init__(self, llm_client: ILLMClient, tools_manager: ToolManager):
        self.coordinator = SubAgentCoordinator(llm_client, tools_manager)

    async def execute(self, args: dict, context: dict) -> str:
        # The coordinator can be passed in context for better testability, 
        # otherwise instantiate it using context providers.
        coordinator = context.get("subagent_coordinator")
        
        if not coordinator:
            if not self.coordinator:
                return "Error: Sub-agent spawning requires 'llm_client' and 'tools_manager' in the context."
            coordinator = self.coordinator

        plan_data = args.get("plan", {})
        parent_memory = context.get("memory_manager")
        
        return await coordinator.run_plan(plan_data, parent_memory)
