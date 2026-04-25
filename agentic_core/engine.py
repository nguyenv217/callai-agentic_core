from typing import AsyncIterator, Iterator, Any
import json

from .llm_providers import ILLMClient, LLMResponse
from .tools import ToolManager
from .memory.manager import MemoryManager
from .observers import AgentEventObserver, DecisionEvent, LastIterationDecision, ToolStartDecision
from .config import ConfigurationError, RunnerConfig

import logging
logger = logging.getLogger(__name__)

class AgentRunner:
    """
    A class that manages the execution of an agent, coordinating between an LLM client,
    tools, memory, and configuration to perform tasks.

    Attributes:
        llm (ILLMClient): The LLM client used for generating responses.
        tools (ToolManager): Manages the tools available to the agent.
        memory (MemoryManager): Handles the agent's memory operations.
        last_usage_meta (Any | None): Stores metadata from the last usage.
        config (RunnerConfig): Configuration settings for the agent runner.
    """

    def __init__(
        self,
        llm_client: ILLMClient,
        tools: ToolManager,
        memory: MemoryManager,
        config: RunnerConfig | None = None,
        observer: AgentEventObserver | None = None
    ):
        """
        Initializes the AgentRunner with the provided LLM client, tools, memory, and configuration.

        Args:
            llm_client (ILLMClient): The LLM client used for generating responses.
            tools (ToolManager): Manages the tools available to the agent.
            memory (MemoryManager): Handles the agent's memory operations.
            config (RunnerConfig): Configuration settings for the agent runner. Can be overwritten at runtime.
            observer (AgentEventObserver): Observer for agent events. Can be overwritten at runtime.
        """
        self.llm = llm_client
        self.tools = tools 
        self.memory = memory
        self.last_usage_meta: None | Any = None
        self.config = config or RunnerConfig()
        self.observer = observer

    @staticmethod
    def _to_async_gen(sync_gen: Iterator[LLMResponse]) -> AsyncIterator[LLMResponse]: # QUICK FIX BEFORE STANDARDIZATION
        """Wrap a synchronous iterator as an async iterator."""                                                    
        async def _wrapper():               
            import asyncio                                                   
            for item in sync_gen:                                                                                  
                # give the event loop a chance to run
                await asyncio.sleep(0)                                                                             
                yield item                                                                                         
        return _wrapper()
    
    def _add_error_tool_result(self, tool_name: str, tool_id: str, msg: str, observer: AgentEventObserver): # helper for less verbosity
        observer.on_tool_complete(tool_name, tool_id, False, msg)
        self.memory.add_tool_result(name=tool_name, tool_call_id=tool_id, content=msg)

    async def run_turn(self, user_input: str | list[dict], observer: AgentEventObserver | None = None, config: RunnerConfig | None = None) -> dict:
        if not observer:
            if not self.observer:
                raise ConfigurationError("`observer`: `AgentEventObserver` must be provided either during initialization or at runtime.")
            else:
                observer = self.observer

        config = config or self.config

        if config.mcp_clear_loaded_tools:
            self.tools.clear_loaded_tools()

        # === Sytem prompt ===
        if config.system_prompt:
            # If a toolset also defines a prompt, prepend it to the system prompt.
            toolset_prompt = self.tools.get_toolset_prompt(config.toolset) if config.toolset else None
            if toolset_prompt:
                combined_prompt = f"{toolset_prompt}\n\n{config.system_prompt}"
                self.memory.set_system_prompt(combined_prompt)
            else:
                self.memory.set_system_prompt(config.system_prompt)
        else:
            # No explicit system_prompt; fall back to toolset prompt if available.
            toolset_prompt = self.tools.get_toolset_prompt(config.toolset) if config.toolset else None
            if toolset_prompt:
                self.memory.set_system_prompt(toolset_prompt if not self.memory.system_prompt_exists() else  toolset_prompt + "\n\n" + self.memory.system_prompt['content'])

        # === User input ===
        messages = [{"role": "user", "content": user_input}] if isinstance(user_input, str) else user_input
        for message in messages:
            self.memory.add_message(message)

        observer.on_turn_start()
        
        # Preparation phase to setup configureed MCP servers and tools. MCP settings go here
        await self.tools.prepare_turn(config)
        
        # Tools preprocssing. Verbose but assume no one is modifying `tool_manager.toolsets` directly. 
        active_tools = config.tools or self.tools.get_tools_from_toolset(config.toolset)
        active_tools.extend(self.tools.get_mcp_loaded_tools()) 
        
        if config.mcp_enable_discovery:
            active_tools.extend(self.tools.get_discovery_tools()) 

        max_iterations = config.max_iterations
        iteration = 1

        final_response = {"text": "", "reasoning": "", "tool_calls": [], "usage": {}}

        try:
            # The Agentic Loop
            while iteration <= max_iterations:
                observer.on_iteration_start(iteration, max_iterations)
                
                conversation = self.memory.get_history()
                logger.info(f"Tools turn {iteration}: {[t['function']['name'] for t in active_tools]}")
                
                response_iterator = self.llm.ask(conversation, active_tools) 
                
                if not isinstance(response_iterator, AsyncIterator): # hotfix before standardization
                    response_iterator = AgentRunner._to_async_gen(response_iterator)

                turn_response = {"text": "", "reasoning": "", "tool_calls": []}
                
                async for response in response_iterator:
                    if response.success:
                        if response.text: turn_response["text"] += response.text
                        if response.reasoning: turn_response["reasoning"] += response.reasoning
                        if response.tool_calls: turn_response["tool_calls"].extend(response.tool_calls)
                        if response.usage:
                            self.last_usage_meta = response.usage
                    else:
                        observer.on_error(response.error or "Unknown LLM error")
                        return {"error": response.error}

                logger.info(f"Turn response: {turn_response}")

                # If the LLM returned final text (no tools requested), we are done.
                if not turn_response["tool_calls"]:
                    self.memory.add_message({"role": "assistant", "content": turn_response["text"]})
                    final_response["text"] = turn_response["text"]
                    final_response["reasoning"] = turn_response["reasoning"]
                    final_response['usage'] = self.last_usage_meta
                    break

                # ==== If we reach here, means it's a tool calling session ====
                # Handle Tool Calls
                self.memory.add_message({
                    "role": "assistant", 
                    "content": turn_response.get("text", ""), 
                    "tool_calls": turn_response["tool_calls"]
                })

                reasoning_text = turn_response.get("reasoning") or turn_response.get("text")
                observer.on_tool_call_session_start(
                    reasoning_text=reasoning_text,
                    tool_calls=turn_response["tool_calls"],
                    iteration=iteration,
                    max_iterations=max_iterations
                )

                tasks = []
                tc_meta = []

                for tc in turn_response["tool_calls"]:
                    tool_name = tc['function']["name"]
                    tool_args = tc['function'].get("arguments", {})
                    tool_id = tc.get("id", "")

                    decision_event: DecisionEvent[ToolStartDecision] = observer.on_tool_start(tool_name, tool_id, tool_args)
                    if decision_event.action == ToolStartDecision.SKIP:
                        continue
                    elif decision_event.action == ToolStartDecision.SKIP_WITH_MSG:
                        self._add_error_tool_result(tool_name, tool_id, decision_event.message, observer)
                        continue
                    elif decision_event.action == ToolStartDecision.ABANDON:
                        return {"success": True, **final_response}
                    elif decision_event.action == ToolStartDecision.BREAK_WITH_MSG:
                        self._add_error_tool_result(tool_name, tool_id, decision_event.message, observer)
                        break
                    else:
                        try:
                            parsed_args = json.loads(tool_args) if isinstance(tool_args, str) else tool_args
                        except json.JSONDecodeError as e:
                            error_msg = f"Error: Invalid JSON arguments provided. Please fix the syntax and try again. Details: {str(e)}"
                            observer.on_tool_complete(tool_name, tool_id, False, error_msg)
                            self.memory.add_tool_result(name=tool_name, tool_call_id=tool_id, content=error_msg)
                            continue
                        
                        # === Execution ===
                        tasks.append(
                            self.tools.execute(
                                tool_name, parsed_args, controller=observer, 
                                max_chars=config.max_chars,
                                extra_context=config.extra_context,
                            )
                        )
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
                if iteration == max_iterations:
                    decision_event: DecisionEvent[LastIterationDecision] = observer.on_final_iteration()
                    if decision_event.action == LastIterationDecision.CONTINUE:
                        continue
                    elif decision_event.action == LastIterationDecision.LEAVE_MESSAGE:
                        self.memory.add_message({
                            "role":'user',
                            "content": decision_event.message
                        })
                    elif decision_event.action == LastIterationDecision.ABANDON:
                        break

            if iteration > max_iterations:
                observer.on_error(f"Agent failed to provide a final answer after {max_iterations} iterations.")

        except Exception as e:
            logger.exception(f"Error executing agent turn")
            observer.on_error(f"Engine execution error: {str(e)}")
        finally:
            observer.on_turn_complete(final_response)

        return {"success": True, **final_response} if "error" not in final_response else final_response