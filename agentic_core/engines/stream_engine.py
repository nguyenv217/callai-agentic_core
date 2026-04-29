from typing import AsyncGenerator
import json
import asyncio

from ..tools import ToolManager
from ..memory.manager import MemoryManager
from ..observers import AgentEventObserver, DecisionEvent, LastIterationDecision, ToolStartDecision
from ..config import ConfigurationError, RunnerConfig
from ..interfaces import AgentResponse, StreamEvent, StreamEventType
from .engine import AgentRunner

import logging
logger = logging.getLogger(__name__)


class AgentStreamRunner(AgentRunner):
    """
    A specialized AgentRunner that emphasizes streaming capabilities.
    """
    async def stream(self, user_input: str | list[dict], observer: AgentEventObserver | None = None, config: RunnerConfig | None = None) -> AsyncGenerator[StreamEvent, None]:
        return self.stream_turn(user_input, observer, config)
    
    # ======== Entry point ========

    async def stream_turn(self, user_input: str | list[dict], observer: AgentEventObserver | None = None, config: RunnerConfig | None = None) -> AsyncGenerator[StreamEvent, None]:
        if not observer:
            if not self.observer:
                raise ConfigurationError("`observer`: `AgentEventObserver` must be provided either during initialization or at runtime.")
            else:
                observer = self.observer

        config = config or self.config

        if config.mcp_clear_loaded_tools:
            self.tools.clear_loaded_tools()

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

        active_tools = config.tools or self.tools.get_tools_from_toolset(config.toolset)
        active_tools.extend(self.tools.get_mcp_loaded_tools())

        if config.mcp_enable_discovery:
            active_tools.extend(self.tools.get_discovery_tools())

        max_iterations = config.max_iterations
        iteration = 1
        final_response = AgentResponse()

        try:
            while iteration <= max_iterations:
                observer.on_iteration_start(iteration, max_iterations)
                conversation = self.memory.get_history()
                logger.info(f"Tools turn {iteration}: {[t['function']['name'] for t in active_tools]}")
                response_iterator = self.llm.ask(conversation, active_tools, stream=True)

                turn_response = {"text": "", "reasoning": "", "tool_calls": []}

                async for response in response_iterator:
                    if response.success:
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
                    else:
                        observer.on_error(response.error or "Unknown LLM error")
                        yield StreamEvent(StreamEventType.ERROR, response.error)
                        return

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
                        # To break from an async generator and return final_response
                        # we can just break the while loop.
                        iteration = max_iterations + 1
                        break
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

                        tasks.append(
                            self.tools.execute(
                                tool_name, parsed_args, controller=observer,
                                max_chars=config.max_chars,
                                extra_context=config.extra_context,
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
            logger.exception("Error during stream_turn")
            observer.on_error(str(e))
            yield StreamEvent(StreamEventType.ERROR, f"Engine execution error: {str(e)}")

        finally:
            observer.on_turn_complete(final_response)
            yield StreamEvent(StreamEventType.FINAL_RESPONSE, final_response)

    async def run_turn(self, user_input: str | list[dict], observer: AgentEventObserver | None = None, config: RunnerConfig | None = None) -> AgentResponse:
        final_response = AgentResponse()
        async for event in self.stream_turn(user_input, observer, config):
            if event.type == StreamEventType.FINAL_RESPONSE:
                final_response = event.content
            elif event.type == StreamEventType.ERROR:
                final_response.error = event.content
        return final_response