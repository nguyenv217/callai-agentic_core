from typing import Union, List, Dict, Any
from dataclasses import dataclass
import json

from .interfaces.llm import ILLMClient
from .tools.manager import ToolManager
from .memory.manager import MemoryManager
from .interfaces.events import AgentEventObserver
from .observers.base import ToolStartDecision

import logging
logger = logging.getLogger(__name__)

@dataclass
class RunnerConfig:
    max_iterations: int = 20
    toolset: str = "none"
    system_prompt: str | None = None
    tools: list['schema'] | None = None # type: ignore
    clear_loaded_tool: bool = True
    mcp_active_servers: list[str] | None = None  # e.g. ["github", "memory"]
    mcp_preload_tools: list[str] | None = None   # e.g. ["github_create_issue"]

class AgentRunner:
    def __init__(
        self,
        llm_client: ILLMClient,
        tool_manager: ToolManager,
        memory: MemoryManager,
    ):
        self.llm = llm_client
        self.tool_manager = tool_manager
        self.memory = memory
        self.session_completion_tokens = 0

    async def run_turn(self, user_input: Union[str, List[Dict]], observer: AgentEventObserver, config: RunnerConfig | None = None) -> Dict:
        config = config or RunnerConfig()

        if config.clear_loaded_tool:
            self.tool_manager.clear_loaded_tools()

        max_iterations = config.max_iterations

        if config.system_prompt:
            self.memory.set_system_prompt(config.system_prompt)
        
        # Normalize and add user input
        messages = [{"role": "user", "content": user_input}] if isinstance(user_input, str) else user_input
        for message in messages:
            self.memory.add_message(message)

        observer.on_turn_start()
        
        # Preparation phase to setup configureed MCP servers and tools
        await self.tool_manager.prepare_turn(config)
            
        active_tools = config.tools or self.tool_manager.get_tools(config.toolset)
        iteration = 1

        final_response = {"text": "", "tool_calls": [], "usage": {}}

        try:
            # The Agentic Loop
            while iteration <= max_iterations:
                observer.on_iteration_start(iteration, max_iterations)
                
                conversation = self.memory.get_history()
                logger.info(f"Tools turn {iteration}: {[t['function']['name'] for t in active_tools]}")
                
                response_iterator = self.llm.ask(conversation, active_tools) # Is async iterator

                turn_response = {"text": "", "tool_calls": []}
                
                async for response in response_iterator:
                    if response.success:
                        if response.text: turn_response["text"] += response.text
                        if response.tool_calls: turn_response["tool_calls"].extend(response.tool_calls)
                        if response.usage:
                            self.session_completion_tokens += response.usage.get('completion_tokens', 0)
                    else:
                        observer.on_error(response.error or "Unknown LLM error")
                        return {"error": response.error}

                logger.info(f"Turn response: {turn_response}")

                # If the LLM returned final text (no tools requested), we are done.
                if not turn_response["tool_calls"]:
                    self.memory.add_message({"role": "assistant", "content": turn_response["text"]})
                    final_response["text"] = turn_response["text"]
                    break

                # Handle Tool Calls
                self.memory.add_message({
                    "role": "assistant", 
                    "content": turn_response.get("text", ""), 
                    "tool_calls": turn_response["tool_calls"]
                })

                tasks = []
                tc_meta = []

                for tc in turn_response["tool_calls"]:
                    tool_name = tc['function']["name"]
                    tool_args = tc['function'].get("arguments", {})
                    tool_id = tc.get("id", "")

                    decision: ToolStartDecision = observer.on_tool_start(tool_name, tool_id, tool_args)
                    if decision == ToolStartDecision.SKIP:
                        continue
                    elif decision == ToolStartDecision.ABANDON:
                        break

                    try:
                        parsed_args = json.loads(tool_args) if isinstance(tool_args, str) else tool_args
                    except json.JSONDecodeError as e:
                        error_msg = f"Error: Invalid JSON arguments provided. Please fix the syntax and try again. Details: {str(e)}"
                        observer.on_tool_complete(tool_name, tool_id, False, error_msg)
                        self.memory.add_tool_result(name=tool_name, tool_call_id=tool_id, content=error_msg)
                        continue

                    tasks.append(self.tool_manager.execute(tool_name, parsed_args, controller=observer))
                    tc_meta.append((tc["id"], tool_name))

                if len(tasks) > 0:
                    import asyncio
                    tool_results = await asyncio.gather(*tasks, return_exceptions=True)

                    for i, tool_result in enumerate(tool_results):
                        tc_id, tool_name = tc_meta[i]
                        success = not isinstance(tool_result, Exception)
                        observer.on_tool_complete(tool_name, tc_id, success, tool_result)
                        
                        self.memory.add_tool_result(
                            tool_call_id=tc_id,
                            name=tool_name,
                            content=str(tool_result)
                        )

                    self.memory.enforce_context_limits()
                    
                iteration += 1

            if iteration > max_iterations:
                observer.on_error(f"Agent failed to provide a final answer after {max_iterations} iterations.")

        except Exception as e:
            logger.exception(f"Error executing agent turn")
            observer.on_error(f"Engine execution error: {str(e)}")
        finally:
            observer.on_turn_complete(final_response)

        return {"success": True, **final_response} if "error" not in final_response else final_response