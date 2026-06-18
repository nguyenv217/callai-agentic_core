"""
Agent Builder - High-level agent constructors for quick setup.

This module provides a fluent builder API to construct agents without
managing the underlying instantiation of components manually.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Any
from dataclasses import dataclass

if TYPE_CHECKING:
    from openai import OpenAI

from ..engines.engine import AgentRunner
from ..memory.manager import MemoryManager
from ..tools import ToolManager
from ..tools.base import BaseTool
from ..handlers.standard import SilentHandler, PrintHandler, AgentEventHandler
from ..models import AgentResponse
from ..interfaces import Message
from ..llm_providers import OpenAILLM, AnthropicLLM, OllamaLLM
from ..config import RunnerConfig


class AgentBuilder:
    """Fluent API for constructing AgentRunners."""
    def __init__(self):
        self._provider = "openai"
        self._api_key: str | None = None
        self._model: str | None = None
        self._base_url: str | None = None
        self._client: Any = None
        self._llm_kwargs: dict[str, Any] = {}
        
        self._system_prompt = "You are a helpful assistant."
        self._tenant_id = "default"
        
        self._tools: list[BaseTool] = []
        self._mcp_config_path: str | None = None
        self._mcp_config_dict: dict | None = None
        self._enable_mcp_discovery = True
        self._mcp_extra_env: dict | None = None
        self._mcp_initialize_timeout = 15.0
        
        self._memory_max_chars = 80000
        self._memory_strategy = None
        
        self._handler: AgentEventHandler | None = None

    def with_provider_openai(self, api_key: str | None = None, model: str = "gpt-4o", base_url: str | None = None, client: Any | None = None, timeout: float = 30.0, **kwargs) -> AgentBuilder:
        self._provider = "openai"
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._client = client
        self._llm_kwargs = {"timeout": timeout, **kwargs}
        return self

    def with_provider_anthropic(self, api_key: str, model: str = "claude-3-5-sonnet-20241022", **kwargs) -> AgentBuilder:
        self._provider = "anthropic"
        self._api_key = api_key
        self._model = model
        self._llm_kwargs = kwargs
        return self

    def with_provider_ollama(self, model: str = "llama3.1", base_url: str = "http://localhost:11434", **kwargs) -> AgentBuilder:
        self._provider = "ollama"
        self._model = model
        self._base_url = base_url
        self._llm_kwargs = kwargs
        return self

    def with_system_prompt(self, prompt: str) -> AgentBuilder:
        self._system_prompt = prompt
        return self

    def with_tools(self, tools: list[BaseTool]) -> AgentBuilder:
        self._tools.extend(tools)
        return self

    def with_mcp(self, config_path: str | None = None, config_dict: dict | None = None, enable_discovery: bool = True, extra_env: dict | None = None, initialize_timeout: float = 15.0) -> AgentBuilder:
        self._mcp_config_path = config_path
        self._mcp_config_dict = config_dict
        self._enable_mcp_discovery = enable_discovery
        self._mcp_extra_env = extra_env
        self._mcp_initialize_timeout = initialize_timeout
        return self

    def with_memory(self, max_chars: int = 80000, strategy: Any = None) -> AgentBuilder:
        self._memory_max_chars = max_chars
        self._memory_strategy = strategy
        return self

    def with_handler(self, handler: AgentEventHandler | None) -> AgentBuilder:
        self._handler = handler
        return self
        
    def with_tenant(self, tenant_id: str) -> AgentBuilder:
        self._tenant_id = tenant_id
        return self

    def build(self) -> AgentRunner:
        if self._provider == "openai":
            base_url = self._base_url or "https://api.openai.com/v1"
            llm = OpenAILLM(api_key=self._api_key, model=self._model, base_url=base_url, client=self._client, **self._llm_kwargs)
        elif self._provider == "anthropic":
            llm = AnthropicLLM(api_key=self._api_key, model=self._model, **self._llm_kwargs)
        elif self._provider == "ollama":
            llm = OllamaLLM(model=self._model, base_url=self._base_url, **self._llm_kwargs)
        else:
            raise ValueError(f"Unknown provider: {self._provider}")

        memory = MemoryManager(max_chars=self._memory_max_chars, strategy=self._memory_strategy)
        memory.set_system_prompt(self._system_prompt)

        tools = ToolManager(
            mcp_config_path=self._mcp_config_path,
            enable_mcp_discovery=self._enable_mcp_discovery,
            extra_env=self._mcp_extra_env,
            tenant_id=self._tenant_id,
            mcp_initialize_timeout=self._mcp_initialize_timeout
        )
        if self._mcp_config_dict:
            tools._mcp_config_dict = self._mcp_config_dict
            
        for t in self._tools:
            tools.register_tool(t)

        handler = self._handler or SilentHandler()
        
        return AgentRunner(llm_client=llm, tools=tools, memory=memory, handler=handler)

@dataclass
class ChatResult:
    response: AgentResponse
    session_id: str | None = None
    tenant_id: str | None = None

async def chat(
    message: str | list[Message],
    runner: AgentRunner | None = None,
    provider: str = "openai",
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    system_prompt: str = "You are a helpful assistant.",
    mcp_config_path: str | None = None,
    verbose: bool = False,
    config: RunnerConfig | None = None,
    session_id: str | None = None,
    tenant_id: str = "default",   
    **kwargs
) -> ChatResult:
    """
    The absolute simplest way to start an agentic flow.
    """
    handler = PrintHandler() if verbose else SilentHandler()

    if runner: 
        agent = runner
    else:
        async def _create_agent() -> AgentRunner:
            builder = AgentBuilder().with_system_prompt(system_prompt).with_mcp(config_path=mcp_config_path).with_handler(handler).with_tenant(tenant_id)
            if provider == 'openai':
                builder.with_provider_openai(api_key=api_key, model=model or "gpt-4o", base_url=base_url, **kwargs)
            elif provider == "anthropic":
                builder.with_provider_anthropic(api_key=api_key, model=model or "claude-3-5-sonnet-20241022", **kwargs)
            elif provider == "ollama":
                builder.with_provider_ollama(model=model or "llama3.1", base_url=base_url or "http://localhost:11434", **kwargs)
            else:
                raise ValueError(f"Unknown provider: {provider}. Use 'openai', 'anthropic', or 'ollama'")
            return builder.build()

        if session_id:
            from ..engines.session import global_session_manager
            agent = await global_session_manager.get_runner(
                session_id=session_id, 
                creator_func=_create_agent,
                tenant_id=tenant_id
            )
        else:
            agent = await _create_agent()

    result = await agent.run_turn(message, handler=handler, config=config)

    return ChatResult(
        response=result,
        session_id=session_id,
        tenant_id=tenant_id
    )
