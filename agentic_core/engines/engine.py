from typing import Any, AsyncGenerator, Callable
import json
import asyncio

from agentic_core.utils import HeuristicFailedToParse, heuristic_json_parse

from ..llm_providers import ILLMClient 
from ..tools import ToolManager
from ..memory.manager import MemoryManager
from ..observers import AgentEventObserver
from ..decisions import DecisionEvent, LastIterationAction, LastIterationDecision, ToolStartDecision
from ..config import ConfigurationError, RunnerConfig
from ..interfaces import (
    AgentResponse, 
    IterationLimitReachedError, 
    ProviderAuthenticationError, ProviderRateLimitError, ProviderTimeoutError, 
    StreamEvent, StreamEventType
)

import logging
logger = logging.getLogger(__name__)

import os
try:
    agentic_max = os.getenv("AGENTIC_ITERATION_MAXIMUM")
    if agentic_max:
        AGENTIC_ITERATION_MAXIMUM = int(agentic_max)
    else:
        AGENTIC_ITERATION_MAXIMUM = 50
except ValueError:
    logger.warning("Invalid value for AGENTIC_ITERATION_MAXIMUM")
    AGENTIC_ITERATION_MAXIMUM = 50

class AgentRunner:
    """
    A class that manages the execution of an agent, coordinating between an LLM client,
    tools, memory, and configuration to perform tasks.
    """

    def __init__(
        self,
        llm_client: ILLMClient,
        tools: ToolManager,
        memory: MemoryManager,
        config: RunnerConfig | None = None,
        observer: AgentEventObserver | None = None,
        tool_args_parser: Callable[[str], dict[str, Any]] | None = None
    ):
        """
        Initializes the AgentRunner with the provided LLM client, tools, memory, and configuration.

        Args:
            llm_client (ILLMClient): The LLM client used for generating responses.
            tools (ToolManager): Manages the tools available to the agent.
            memory (MemoryManager): Handles the agent's memory operations.
            config (RunnerConfig): Configuration settings for the agent runner. Can be overwritten at runtime.
            observer (AgentEventObserver): Observer for agent events. Can be overwritten at runtime.
            tool_args_parser (Callable[[str], dict[str,Any]): 
                Custom runtime tool argument parser for LLM `tool_call` whenever the argument is a string and need manual parsing. 
                Defaults to `heuristic_json_parse()` which attemps to extract and parse the string heuristically with regex and ast.
        """
        self.llm = llm_client
        self.tools = tools 
        self.memory = memory
        self.last_usage_meta: None | Any = None
        self.config = config or RunnerConfig()
        self.observer = observer
        self._toolset_prompt_loaded = False
        self.tool_args_parser = tool_args_parser or heuristic_json_parse

    # ===================
    # Context management 
    # ===================

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.tools.shutdown_mcp()

    # ================
    #  Helpers 
    # ================

    def _add_error_tool_result(self, tool_name: str, tool_id: str, msg: str, observer: AgentEventObserver):
        observer.on_tool_complete(tool_name, tool_id, False, msg)
        self.memory.add_tool_result(name=tool_name, tool_call_id=tool_id, content=msg)

    async def _handle_setup(self, user_input: str | list[dict], config: RunnerConfig, observer: AgentEventObserver):
        """Handles the setup of the agent runner for a new turn."""

        if config.system_prompt:
            toolset_prompt = self.tools.get_toolset_prompt(config.toolset) if config.toolset else None
            if toolset_prompt:
                combined_prompt = f"{toolset_prompt}\n\n{config.system_prompt}"
                self.memory.set_system_prompt(combined_prompt)
            else:
                self.memory.set_system_prompt(config.system_prompt)
        else:
            toolset_prompt = self.tools.get_toolset_prompt(config.toolset) if config.toolset else None
            if toolset_prompt:
                if not self.memory.system_prompt_exists():
                    self.memory.set_system_prompt(toolset_prompt)
                else:
                    # Prevent infinite concatenation across turns
                    current_prompt = self.memory.system_prompt['content']
                    if not self._toolset_prompt_loaded:
                        self.memory.set_system_prompt(toolset_prompt + "\n\n" + current_prompt)
                        self._toolset_prompt_loaded = True

        messages = [{"role": "user", "content": user_input}] if isinstance(user_input, str) else user_input
        for message in messages:
            self.memory.add_message(message)

        observer.on_turn_start()
        await self.tools.prepare_turn(config)

    def _get_active_tools(self, config: RunnerConfig):
        active_tools = list(config.tools) if config.tools else self.tools.get_tools_from_toolset(config.toolset)
        
        if config.mcp_use_loaded_tools:
            active_tools.extend([t for t in self.tools.get_mcp_loaded_tools() if t not in active_tools])
        
        if config.mcp_enable_discovery:
            active_tools.extend([t for t in self.tools.get_discovery_tools() if t not in active_tools])
            
        return active_tools
    
    # ================
    #  Entry point 
    # ================

    async def stream_turn(
        self, 
        user_input: str | list[dict], 
        observer: AgentEventObserver | None = None, 
        config: RunnerConfig | None = None,
        tool_args_parser: Callable[[str], dict[str, Any]] | None = None
        ) -> AsyncGenerator[StreamEvent, None]:
        """
        Executes a turn of the agent, streaming events as they occur.

        Args:
            user_input: The user's input, either as a string or a list of message dictionaries.
            observer: An optional observer to receive events during the agent's execution.
            config: Optional configuration that overrides the default runner configuration.

        Yields:
            `StreamEvent`: 
                Events representing the agent's thought process, tool usage, and final responses. 
                Access content type via `StreamEvent.type` and `StreamEvent.content`
                Access original errors via `StreamEvent.error`: `BaseException` | `None`
        """

        if not observer:
            if not self.observer:
                raise ConfigurationError("`observer`: `AgentEventObserver` must be provided either during initialization or at runtime.")
            else:
                observer = self.observer

        config = config or self.config

        await self._handle_setup(user_input, config, observer)

        active_tools = self._get_active_tools(config)
        logger.info(f"Active tools: {[t['function']['name'] for t in active_tools]}")

        max_iterations = config.max_iterations
        iteration = 1
        final_response = AgentResponse()

        try:
            while iteration <= max_iterations and iteration <= AGENTIC_ITERATION_MAXIMUM:
                observer.on_iteration_start(iteration, max_iterations)
                conversation = self.memory.get_history()
                logger.info(f"Tools turn {iteration}: {[t['function']['name'] for t in active_tools]}")

                kwargs = config.kwargs or {}
                response_iterator = self.llm.ask(conversation, active_tools, stream=True, **kwargs)

                turn_response = {"text": "", "reasoning": "", "tool_calls": []}
                
                try:
                    async for response in response_iterator:
                        if response.text:
                            turn_response["text"] += response.text
                            yield StreamEvent(StreamEventType.TEXT, response.text)
                        if response.reasoning:
                            turn_response["reasoning"] += response.reasoning
                            yield StreamEvent(StreamEventType.REASONING, response.reasoning)
                        if response.tool_calls:
                            turn_response["tool_calls"].extend(response.tool_calls)
                            for tc in response.tool_calls:
                                yield StreamEvent(StreamEventType.TOOL_CALL, tc)
                        if response.usage:
                            self.last_usage_meta = response.usage
                except ProviderRateLimitError as e:
                    error_msg = f"Rate Limit Exceeded: {str(e)}"
                    observer.on_error(error_msg)
                    yield StreamEvent(StreamEventType.ERROR, error_msg, error=e)
                    final_response.error = error_msg
                    return
                except ProviderAuthenticationError as e:
                    error_msg = f"Authentication Failed: {str(e)}"
                    observer.on_error(error_msg)
                    yield StreamEvent(StreamEventType.ERROR, error_msg, error=e)
                    final_response.error = error_msg
                    return
                except ProviderTimeoutError as e:
                    error_msg = f"Timeout: {str(e)}"
                    observer.on_error(error_msg)
                    yield StreamEvent(StreamEventType.ERROR, error_msg, error=e)
                    final_response.error = error_msg

                logger.info(f"Turn response: {turn_response}")
                if not turn_response["tool_calls"]:
                    self.memory.add_message({"role": "assistant", "content": turn_response["text"]})
                    final_response.text = turn_response["text"]
                    final_response.reasoning = turn_response["reasoning"]
                    final_response.usage = self.last_usage_meta or {}
                    break

                # ==== If we reach here, means it's a tool calling session ====
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

                # ==== Handle tool-calling ====

                tasks = []
                tc_meta = []

                for tc in turn_response["tool_calls"]:
                    tool_name = tc['function']["name"]
                    tool_args = tc['function'].get("arguments", {})
                    tool_id = tc.get("id", "")

                    decision_event = observer.on_tool_start(tool_name, tool_id, tool_args)
                    match decision_event.action:
                        case ToolStartDecision.SKIP():
                            continue
                        case ToolStartDecision.SKIP_WITH_MSG(msg):
                            self._add_error_tool_result(tool_name, tool_id, msg, observer)
                            continue
                        case ToolStartDecision.ABANDON():
                            iteration = max_iterations + 1
                            break
                        case ToolStartDecision.BREAK_WITH_MSG(msg):
                            self._add_error_tool_result(tool_name, tool_id, msg, observer)
                            break
                        case ToolStartDecision.CONTINUE():
                            try:
                                parser = tool_args_parser or self.tool_args_parser

                                parsed_args = parser(tool_args) if isinstance(tool_args, str) else tool_args
                            except HeuristicFailedToParse as e:
                                error_msg = f"Error: Invalid JSON arguments provided. Please fix the syntax and try again. Details: {str(e)}"
                                observer.on_tool_complete(tool_name, tool_id, False, error_msg)
                                self.memory.add_tool_result(name=tool_name, tool_call_id=tool_id, content=error_msg)
                                continue

                            tasks.append(
                                self.tools.execute(
                                    tool_name, parsed_args, controller=observer,
                                    max_chars=config.max_chars,
                                    extra_context={
                                        **(config.extra_context or {}),
                                        "llm_client": self.llm,
                                        "tools_manager": self.tools,
                                        "memory_manager": self.memory,
                                    },
                                )
                            )
                            tc_meta.append((tc["id"], tool_name))

                if len(tasks) > 0:
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
                        yield StreamEvent(StreamEventType.TOOL_RESULT, {"tool": tool_name, "id": tc_id, "result": tool_result, "success": success})

                iteration += 1
                if iteration == max_iterations:
                    decision_event: DecisionEvent[LastIterationAction] = observer.on_final_iteration()

                    match decision_event.action:
                        case LastIterationDecision.CONTINUE():
                            continue
                        case LastIterationDecision.LEAVE_MSG(msg):
                            self.memory.add_message({
                                "role":'user',
                                "content": msg
                            })
                            continue
                        case LastIterationDecision.ABANDON():
                            break
                        case LastIterationDecision.EXTEND(max_iterations_count):
                            max_iterations += max_iterations_count or max_iterations
                            continue

            if iteration > max_iterations:
                error_msg = f"Agent failed to provide a final answer after {max_iterations} iterations."
                observer.on_error(error_msg)
                limit_error = IterationLimitReachedError(error_msg)
                final_response.error = limit_error
                yield StreamEvent(StreamEventType.ERROR, error_msg, error=limit_error)

        except Exception as e:
            # If we reached here, they are unexpected crashes
            logger.exception("Unexpected error during stream_turn")
            error_msg = f"Engine execution error: {str(e)}"
            observer.on_error(error_msg)
            yield StreamEvent(StreamEventType.ERROR, error_msg, error=e)
            final_response.error = error_msg

        finally:
            observer.on_turn_complete(final_response)
            yield StreamEvent(StreamEventType.FINAL_RESPONSE, final_response)

    async def run_turn(self, user_input: str | list[dict], observer: AgentEventObserver | None = None, config: RunnerConfig | None = None) -> AgentResponse:
        """
        Standard method that wraps the `stream_turn` to return a single block response.

        Args:
            user_input: The user's input, either as a string or a list of message dictionaries.
            observer: An optional observer to receive events during the agent's execution.
            config: Optional configuration that overrides the default runner configuration.

        Returns:
            AgentResponse: The final response from the agent, including text, reasoning, usage, and any errors.

        Raises (package-specific):
            ProviderAuthenticationError: If there's an authentication error with the LLM provider.
            ProviderRateLimitError: If the LLM provider rate limits are exceeded.
        """
        final_response = AgentResponse()
        cached_error = None
        cached_error_msg = None
        
        async for event in self.stream_turn(user_input, observer, config):
            if event.type == StreamEventType.FINAL_RESPONSE:
                final_response = event.content
            elif event.type == StreamEventType.ERROR:
                cached_error = event.error
                cached_error_msg = event.content
        
        if cached_error and not final_response.error:
            final_response.error = cached_error
            final_response.text += cached_error_msg
                
        return final_response

