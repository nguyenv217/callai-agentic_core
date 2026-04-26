"""
Agent Builder - High-level agent constructors for quick setup.

This module provides simple, idiot-proof ways to create agents without
understanding the underlying protocols.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import OpenAI

from ..engines.engine import AgentRunner
from ..memory.manager import MemoryManager
from ..tools import ToolManager
from ..observers.standard import SilentObserver, PrintObserver, AgentEventObserver

# Import our new isolated providers
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
    tools = ToolManager(mcp_config_path=mcp_config_path)
    observer = observer or SilentObserver()
    
    return AgentRunner(llm_client=llm, tools=tools, memory=memory)


def create_anthropic_agent(
    api_key: str,
    model: str = "claude-3-5-sonnet-20241022",
    system_prompt: str = "You are a helpful assistant.",
    mcp_config_path: str | None = None,
    observer: AgentEventObserver | None = None,
    **kwargs
) -> AgentRunner:
    """Create an Anthropic Claude agent in one line."""
    llm = AnthropicLLM(api_key=api_key, model=model, **kwargs)
    memory = MemoryManager()
    memory.set_system_prompt(system_prompt)
    tools = ToolManager(mcp_config_path=mcp_config_path)
    observer = observer or SilentObserver()
    
    return AgentRunner(llm_client=llm, tools=tools, memory=memory)


def create_ollama_agent(
    model: str = "llama3.1",
    system_prompt: str = "You are a helpful assistant.",
    base_url: str | None = None,
    mcp_config_path: str | None = None,
    observer: AgentEventObserver | None = None,
    **kwargs
) -> AgentRunner:
    """Create an Ollama local agent in one line."""
    llm_kwargs = {"base_url": base_url} if base_url else {}
    llm = OllamaLLM(model=model, **llm_kwargs, **kwargs)
    memory = MemoryManager()
    memory.set_system_prompt(system_prompt)
    tools = ToolManager(mcp_config_path=mcp_config_path)
    observer = observer or SilentObserver()
    
    return AgentRunner(llm_client=llm, tools=tools, memory=memory)


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
    **kwargs
) -> str:
    """
    The absolute simplest way to start an agentic flow.
    
    Example:
        result = await chat("What's the weather?", provider="openai", api_key="sk-...")
    
    Args:
        message: The user's message
        system_prompt: The agent's persona 
        mcp_config_path: Path to MCP config (optional)
        verbose: If True, prints all events
        runner: An AgentRunner instance. If this is supplied, you don't need to use the below arguments.
        provider: "openai", "anthropic", or "ollama"
        api_key: Your API key (required for openai/anthropic)
        base_url: If using provider 'openai', for custom inference endpoint (OpenAI-compatible)
        model: Model name (defaults to provider's recommended)
        **kwargs: Any additional arguments to pass to the agent creation function.
    
    Returns:
        The agent's text response
    """
    observer = PrintObserver() if verbose else SilentObserver()

    if runner: agent=runner
    elif provider == 'openai':
        agent = create_openai_agent(
            api_key=api_key,
            model=model,
            base_url=base_url,
            system_prompt=system_prompt,
            mcp_config_path=mcp_config_path,
            observer=observer,
            **kwargs
        )
    elif provider == "anthropic":
        agent = create_anthropic_agent(
            api_key=api_key,
            model=model,
            system_prompt=system_prompt,
            mcp_config_path=mcp_config_path,
            observer=observer,
            **kwargs
        )
    elif provider == "ollama":
        agent = create_ollama_agent(
            model=model,
            base_url=base_url,
            system_prompt=system_prompt,
            mcp_config_path=mcp_config_path,
            observer=observer,
            **kwargs
        )
           
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'openai', 'anthropic', or 'ollama'")
    
    # Run the turn
    
    result = await agent.run_turn(message, observer=observer, config=config)
    
    return result.get("text", result.get("error", "No response"))