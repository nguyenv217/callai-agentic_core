from typing import Any, AsyncGenerator, Callable
import asyncio
import logging

from agentic_core.utils import HeuristicFailedToParse, heuristic_json_parse
from ..llm_providers import ILLMClient 
from ..tools import ToolManager
from ..memory.manager import MemoryManager
from ..handlers import AgentEventHandler
from ..decisions import (
    LastIterationDecision, 
    ToolStartDecision,
    ErrorContext,
    ErrorDecision
)
from ..config import ConfigurationError, RunnerConfig
from ..interfaces import (
    AgentResponse, 
    IterationLimitReachedError, 
    StreamEvent, 
    StreamEventType
)

logger = logging.getLogger(__name__)

import os
try:
    agentic_max = int(os.getenv("AGENTIC_ITERATION_MAXIMUM", "50"))
except ValueError:
    logger.warning("Invalid value for AGENTIC_ITERATION_MAXIMUM")
    agentic_max = 50
AGENTIC_ITERATION_MAXIMUM = agentic_max

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
        handler: AgentEventHandler | None = None,
        tool_args_parser: Callable[[str], dict[str, Any]] | None = None
    ):
        """
        Initializes the AgentRunner with the provided LLM client, tools, memory, and configuration.

        Args:
            llm_client (ILLMClient): The LLM client used for generating responses.
            tools (ToolManager): Manages the tools available to the agent.
            memory (MemoryManager): Handles the agent's memory operations.
            config (RunnerConfig): Configuration settings for the agent runner. Can be overwritten at runtime.
            handler (AgentEventHandler): Handler for agent events. Can be overwritten at runtime.
            tool_args_parser (Callable[[str], dict[str,Any]): 
                Custom runtime tool argument parser for LLM `tool_call` whenever the argument is a string and need manual parsing. 
                Defaults to `heuristic_json_parse()` which attemps to extract and parse the string heuristically with regex and ast.
        """
        self.llm = llm_client
        self.tools = tools 
        self.memory = memory
        self.last_usage_meta = None
        self.config = config or RunnerConfig()
        self.handler = handler
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

    async def _add_error_tool_result(self, tool_name: str, tool_id: str, msg: str, handler: AgentEventHandler):
        await handler.on_tool_complete(tool_name, tool_id, False, msg)
        self.memory.add_tool_result(name=tool_name, tool_call_id=tool_id, content=msg)

    async def _handle_setup(self, user_input: str | list[dict], config: RunnerConfig, handler: AgentEventHandler):
        """Handles the setup of the agent runner for a new turn."""
        toolset_prompt = self.tools.get_toolset_prompt(config.toolset) if config.toolset else None
        
        if config.system_prompt:
            combined = f"{toolset_prompt}\n\n{config.system_prompt}" if toolset_prompt else config.system_prompt
            self.memory.set_system_prompt(combined)
        elif toolset_prompt:
            if not self.memory.system_prompt_exists():
                self.memory.set_system_prompt(toolset_prompt)
            elif not self._toolset_prompt_loaded:
                self.memory.set_system_prompt(toolset_prompt + "\n\n" + self.memory.system_prompt['content'])
                self._toolset_prompt_loaded = True

        messages = [{"role": "user", "content": user_input}] if isinstance(user_input, str) else user_input
        for msg in messages:
            self.memory.add_message(msg)

        await handler.on_turn_start()
        await self.tools.prepare_turn(config)

    def _get_active_tools(self, config: RunnerConfig):
        active_tools = []
        
        if config.tools is not None:
            # User explicitly passed a list (even if it's an empty list `[]` meaning no tools)
            if config.tools: 
                missing = None
                specified_by_tool_name = isinstance(config.tools[0], str)
                if specified_by_tool_name:
                    try:
                        active_tools = [s for s in self.tools.tool_schemas if s['function']['name'] in config.tools]
                        missing = set(config.tools) - {s['function']['name'] for s in active_tools}
                    except TypeError: 
                        # User has intertwined ToolSchema and str
                        raise ConfigurationError("RunnerConfig.tools can only contain either a list of `ToolSchema` or str")
                else:
                    active_tools = config.tools
                    missing = set([t['function']['name'] for t in config.tools if t not in self.tools.tool_schemas])

                if missing:
                    logger.warning(f"Requested tools not registered in ToolManager: {missing}")
                
        elif config.toolset is not None:
            # User explicitly passed a string (e.g., "all", "none", or a custom toolset name)
            active_tools = self.tools.get_tools_from_toolset(config.toolset)
            
        else:
            # User passed NEITHER tools nor toolset. They are unspecified.
            if self.tools.get_registered_tools():
                logger.warning(
                    "No tools were provided to RunnerConfig, but tools are registered in ToolManager. "
                    "If you meant to use them all, pass `RunnerConfig(toolset='all')`. "
                    "If you want NO tools, pass `RunnerConfig(toolset='none')` or `tools=[]`."
                )

        if config.mcp_use_loaded_tools:
            active_tools.extend([t for t in self.tools.get_mcp_loaded_tools() if t not in active_tools])
        if config.mcp_enable_discovery:
            active_tools.extend([t for t in self.tools.get_discovery_tools() if t not in active_tools])
            
        return active_tools
    
    async def _create_error_context(
        self, e: BaseException, tool_name: str | None = None, retry_count: int = 0, 
        max_retries: int = 0, **engine_kwargs
    ) -> ErrorContext:
        return ErrorContext(
            error=e, tool_name=tool_name, retry_count=retry_count, max_retries=max_retries,
            engine_state=engine_kwargs
        )
    
    async def _handle_error_decision(
        self, error_context: ErrorContext, handler: AgentEventHandler, final_response: AgentResponse
    ) -> tuple[bool, "AgentResponse"]:
        decision_event = await handler.on_error(error_context)
        action = decision_event.action
        
        if isinstance(action, ErrorDecision.RETRY):
            if action.delay > 0:
                delay = min(action.delay * (action.exponential_base ** error_context.retry_count), 60.0)
                await asyncio.sleep(delay)
            return False, final_response
            
        elif isinstance(action, ErrorDecision.SKIP):
            return False, final_response
            
        elif isinstance(action, ErrorDecision.RESOLVE_WITH):
            self.memory.add_message({"role": "user", "content": action.msg})
            return False, final_response
            
        else: # ABANDON
            return True, final_response


    # ================
    #  Entry point 
    # ================

    async def stream_turn(
        self, 
        user_input: str | list[dict], 
        handler: AgentEventHandler | None = None, 
        config: RunnerConfig | None = None,
        tool_args_parser: Callable[[str], dict[str, Any]] | None = None
        ) -> AsyncGenerator[StreamEvent, None]:
        """
        Executes a turn of the agent, streaming events as they occur.

        Args:
            user_input: The user's input, either as a string or a list of message dictionaries.
            handler: An optional handler to receive events during the agent's execution.
            config: Optional configuration that overrides the default runner configuration.

        Yields:
            `StreamEvent`: 
                Events representing the agent's thought process, tool usage, and final responses. 
                Access content type via `StreamEvent.type` and `StreamEvent.content`
                Access original errors via `StreamEvent.error`: `BaseException` | `None`
        """
        handler = handler or self.handler
        if not handler:
            raise ConfigurationError("`AgentEventHandler` must be provided.")
        config = config or self.config
        await self._handle_setup(user_input, config, handler)

        active_tools = self._get_active_tools(config)
        max_iterations = config.max_iterations
        iteration = 1
        final_response = AgentResponse()
        retry_count = 0
        max_retries = getattr(config, 'max_retries', 1)

        try:
            while iteration <= max_iterations and iteration <= AGENTIC_ITERATION_MAXIMUM:
                await handler.on_iteration_start(iteration, max_iterations)
                self.memory.enforce_context_limits()
                conversation = self.memory.get_history()
                kwargs = config.kwargs or {}
                
                # Default to True, but allow overriding via config.kwargs
                do_stream = kwargs.pop("stream", True)
                response_iterator = self.llm.ask(conversation, active_tools, stream=do_stream, **kwargs)

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
                            turn_response["tool_calls"] = response.tool_calls
                        if response.usage:
                            self.last_usage_meta = response.usage
                except Exception as e:
                    error_context = await self._create_error_context(e, retry_count=retry_count, max_retries=max_retries)
                    should_abort, final_response = await self._handle_error_decision(error_context, handler, final_response)
                    yield StreamEvent(StreamEventType.ERROR, str(e), error=e)
                    if should_abort:
                        final_response.error = e
                        return
                    retry_count += 1
                    continue

                retry_count = 0

                if not turn_response["tool_calls"]:
                    self.memory.add_message({"role": "assistant", "content": turn_response["text"]})
                    final_response.text = turn_response["text"]
                    final_response.reasoning = turn_response["reasoning"]
                    final_response.usage = self.last_usage_meta or {}
                    break

                for tc in turn_response["tool_calls"]:
                    yield StreamEvent(StreamEventType.TOOL_CALL, tc)

                self.memory.add_message({
                    "role": "assistant",
                    "content": turn_response.get("text", ""),
                    "tool_calls": turn_response["tool_calls"]
                })

                reasoning_text = turn_response.get("reasoning") or turn_response.get("text")
                await handler.on_tool_call_session_start(
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

                    decision_event = await handler.on_tool_start(tool_name, tool_id, tool_args)
                    if isinstance(decision_event.action, ToolStartDecision.SKIP):
                        continue
                    elif isinstance(decision_event.action, ToolStartDecision.SKIP_WITH_MSG):
                        await self._add_error_tool_result(tool_name, tool_id, decision_event.action.msg, handler)
                        continue
                    elif isinstance(decision_event.action, ToolStartDecision.ABANDON):
                        iteration = max_iterations + 1
                        break
                    elif isinstance(decision_event.action, ToolStartDecision.BREAK_WITH_MSG):
                        await self._add_error_tool_result(tool_name, tool_id, decision_event.action.msg, handler)
                        break

                    try:
                        parser = tool_args_parser or self.tool_args_parser
                        parsed_args = parser(tool_args) if isinstance(tool_args, str) else tool_args
                    except HeuristicFailedToParse as e:
                        error_context = await self._create_error_context(e, tool_name=tool_name)
                        should_abort, _ = await self._handle_error_decision(error_context, handler, final_response)
                        await self._add_error_tool_result(tool_name, tool_id, f"Invalid JSON: {e}", handler)
                        if should_abort:
                            final_response.error = e
                            return
                        continue

                    tasks.append(self.tools.execute(
                        tool_name, parsed_args, controller=handler, max_chars=config.max_chars,
                        extra_context={**(config.extra_context or {}), "llm_client": self.llm, "tools_manager": self.tools, "memory_manager": self.memory}
                    ))
                    tc_meta.append((tool_id, tool_name))

                if tasks:
                    tool_results = await asyncio.gather(*tasks, return_exceptions=True)
                    for i, tool_result in enumerate(tool_results):
                        tc_id, tool_name = tc_meta[i]
                        success = not isinstance(tool_result, Exception)
                        
                        if not success:
                            error_context = await self._create_error_context(tool_result, tool_name=tool_name)
                            should_abort, _ = await self._handle_error_decision(error_context, handler, final_response)
                            
                            error_msg = f"Tool execution failed: {type(tool_result).__name__} - {str(tool_result)}"
                            await self._add_error_tool_result(tool_name, tc_id, error_msg, handler)
                            yield StreamEvent(StreamEventType.TOOL_RESULT, {"tool": tool_name, "id": tc_id, "result": error_msg, "success": False})
                            
                            if should_abort:
                                final_response.error = tool_result
                                return
                            continue
                                
                        await handler.on_tool_complete(tool_name, tc_id, success, tool_result)
                        self.memory.add_tool_result(tool_call_id=tc_id, name=tool_name, content=str(tool_result))
                        yield StreamEvent(StreamEventType.TOOL_RESULT, {"tool": tool_name, "id": tc_id, "result": tool_result, "success": success})

                iteration += 1
                if iteration == max_iterations:
                    decision_event = await handler.on_final_iteration()
                    if isinstance(decision_event.action, LastIterationDecision.LEAVE_MSG):
                        self.memory.add_message({"role": 'user', "content": decision_event.action.msg})
                    elif isinstance(decision_event.action, LastIterationDecision.ABANDON):
                        break
                    elif isinstance(decision_event.action, LastIterationDecision.EXTEND):
                        max_iterations += decision_event.action.extra_iterations_count or max_iterations

            if iteration > max_iterations:
                limit_error = IterationLimitReachedError(f"Agent failed after {max_iterations} iterations.")
                error_context = await self._create_error_context(limit_error)
                should_abort, final_response = await self._handle_error_decision(error_context, handler, final_response)
                if should_abort:
                    final_response.error = limit_error
                    yield StreamEvent(StreamEventType.ERROR, str(limit_error), error=limit_error)

        except Exception as e:
            logger.exception("Unexpected error during stream_turn")
            error_context = await self._create_error_context(e)
            should_abort, final_response = await self._handle_error_decision(error_context, handler, final_response)
            if should_abort:
                yield StreamEvent(StreamEventType.ERROR, str(e), error=e)
                final_response.error = e
        finally:
            await handler.on_turn_complete(final_response)
            import sys
            if sys.exc_info()[0] is not GeneratorExit:
                yield StreamEvent(StreamEventType.FINAL_RESPONSE, final_response)

    async def run_turn(self, user_input: str | list[dict], handler: AgentEventHandler | None = None, config: RunnerConfig | None = None) -> AgentResponse:
        """
        Standard method that wraps the `stream_turn` to return a single block response.

        Args:
            user_input: The user's input, either as a string or a list of message dictionaries.
            handler: An optional handler to receive events during the agent's execution.
            config: Optional configuration that overrides the default runner configuration.

        Returns:
            AgentResponse: The final response from the agent, including text, reasoning, usage, and any errors.

        Raises (package-specific):
            ProviderAuthenticationError: If there's an authentication error with the LLM provider.
            ProviderRateLimitError: If the LLM provider rate limits are exceeded.
        """
        final_response = AgentResponse()
        final_response.tool_calls = []
        cached_error = None
        cached_error_msg = None

        async for event in self.stream_turn(user_input, handler, config):
            if event.type == StreamEventType.FINAL_RESPONSE: 
                final_response = event.content
            elif event.type == StreamEventType.TOOL_CALL: 
                final_response.tool_calls.append(event.content)
            elif event.type == StreamEventType.ERROR:
                cached_error = event.error
                cached_error_msg = event.content
        
        if cached_error and not final_response.error:
            final_response.error = cached_error
            final_response.text += cached_error_msg
            
        return final_response