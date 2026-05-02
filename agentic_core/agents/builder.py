"""
Agent Builder - High-level agent constructors for quick setup.

This module provides simple, idiot-proof ways to create agents without
understanding the underlying protocols.
"""
from __future__ import annotations
from typing import TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from openai import OpenAI

from ..engines.engine import AgentRunner
from ..memory.manager import MemoryManager
from ..tools import ToolManager
from ..observers.standard import SilentObserver, PrintObserver, AgentEventObserver
from ..interfaces import AgentResponse
from ..llm_providers import OpenAILLM, AnthropicLLM, OllamaLLM
from ..config import RunnerConfig

def create_openai_agent(
    api_key: str,
    model: str = "gpt-4o",
    system_prompt: str = "You are a helpful assistant.",
    mcp_config_path: str | None = None,
    observer: AgentEventObserver | None = None,
    base_url: str | None = None,
    timeout: float = 30,
    client: OpenAI | None = None,
    tenant_id: str = "default",
    **kwargs
) -> AgentRunner:
    """
    Create an OpenAI agent in one line.

    Args:
        system_prompt: The agent's persona 
        client: An OpenAI client instance. If this is supplied, you don't need to use the below arguments.
        api_key: Your OpenAI API key
        base_url: For OpenAI-compatible endpoints
        model: Model name (defaults to provider's recommended)
        timeout: Timeout in seconds (defaults to 30s)
        mcp_config_path: Path to MCP config (optional)
        observer: An AgentEventObserver instance.
        kwargs: Any additional arguments to pass to the agent creation function (will be passed as extra_body to client request).
    
    Example:
        agent = create_openai_agent(
            api_key="sk-...",
            model="gpt-4o",
            system_prompt="You are a helpful coding assistant."
        )
        result = await agent.run_turn("Hello!", SilentObserver())
    """
    base_url = base_url or "https://api.openai.com/v1"
    llm = OpenAILLM(api_key=api_key, model=model, base_url=base_url, client=client, timeout=timeout, **kwargs)
    memory = MemoryManager()
    memory.set_system_prompt(system_prompt)
    tools = ToolManager(mcp_config_path=mcp_config_path, tenant_id=tenant_id)
    observer = observer or SilentObserver()
    
    return AgentRunner(llm_client=llm, tools=tools, memory=memory)

def create_anthropic_agent(
    api_key: str,
    model: str = "claude-3-5-sonnet-20241022",
    system_prompt: str = "You are a helpful assistant.",
    mcp_config_path: str | None = None,
    observer: AgentEventObserver | None = None,
    tenant_id: str = "default",
    **kwargs
) -> AgentRunner:
    """Create an Anthropic Claude agent in one line."""
    llm = AnthropicLLM(api_key=api_key, model=model, **kwargs)
    memory = MemoryManager()
    memory.set_system_prompt(system_prompt)
    tools = ToolManager(mcp_config_path=mcp_config_path, tenant_id=tenant_id)
    observer = observer or SilentObserver()
    
    return AgentRunner(llm_client=llm, tools=tools, memory=memory)

def create_ollama_agent(
    model: str = "llama3.1",
    system_prompt: str = "You are a helpful assistant.",
    base_url: str | None = None,
    mcp_config_path: str | None = None,
    observer: AgentEventObserver | None = None,
    tenant_id: str = "default",
    **kwargs
) -> AgentRunner:
    """Create an Ollama local agent in one line."""
    llm_kwargs = {"base_url": base_url} if base_url else {}
    llm = OllamaLLM(model=model, **llm_kwargs, **kwargs)
    memory = MemoryManager()
    memory.set_system_prompt(system_prompt)
    tools = ToolManager(mcp_config_path=mcp_config_path, tenant_id=tenant_id)
    observer = observer or SilentObserver()
    
    return AgentRunner(llm_client=llm, tools=tools, memory=memory)

@dataclass
class ChatResult:
    response: AgentResponse
    session_id: str | None = None
    tenant_id: str | None = None

async def chat(
    message: str,
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
) -> AgentResponse:
    """
    The absolute simplest way to start an agentic flow.
    """
    observer = PrintObserver() if verbose else SilentObserver()

    if runner: 
        agent = runner
    else:
        # Wrap the synchronous agent creation in an async factory for the session manager
        async def _create_agent() -> AgentRunner:
            if provider == 'openai':
                return create_openai_agent(
                    api_key=api_key, model=model, base_url=base_url,
                    system_prompt=system_prompt, mcp_config_path=mcp_config_path,
                    observer=observer, tenant_id=tenant_id, **kwargs
                )
            elif provider == "anthropic":
                return create_anthropic_agent(
                    api_key=api_key, model=model, system_prompt=system_prompt,
                    mcp_config_path=mcp_config_path, observer=observer, 
                    tenant_id=tenant_id, **kwargs
                )
            elif provider == "ollama":
                return create_ollama_agent(
                    model=model, base_url=base_url, system_prompt=system_prompt,
                    mcp_config_path=mcp_config_path, observer=observer, 
                    tenant_id=tenant_id, **kwargs
                )
            else:
                raise ValueError(f"Unknown provider: {provider}. Use 'openai', 'anthropic', or 'ollama'")

        if session_id:
            from ..engines.session import global_session_manager
            # Fetch from cache or create new using the session manager
            agent = await global_session_manager.get_runner(
                session_id=session_id, 
                creator_func=_create_agent,
                tenant_id=tenant_id
            )
        else:
            # Throwaway agent for single-turn executions
            agent = await _create_agent()

    # Run the turn
    result = await agent.run_turn(message, observer=observer, config=config)

    return ChatResult(
        response= result,
        session_id=session_id,
        tenant_id=tenant_id
    )
