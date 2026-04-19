"""
Agent Builder - High-level agent constructors for quick setup.

This module provides simple, idiot-proof ways to create agents without
understanding the underlying protocols.
"""
from typing import Optional

from ..engine import AgentRunner
from ..memory.manager import MemoryManager
from ..tools.manager import ToolManager
from ..observers.base import AgentEventObserver
from ..observers.standard import DefaultObserver, PrintObserver

# Import our new isolated providers
from ..llm_providers.openai import OpenAILLM
from ..llm_providers.anthropic import AnthropicLLM
from ..llm_providers.ollama import OllamaLLM
from ..engine import RunnerConfig

def create_openai_agent(
    api_key: str,
    model: str = "gpt-4o",
    system_prompt: str = "You are a helpful assistant.",
    mcp_config_path: Optional[str] = None,
    observer: Optional[AgentEventObserver] = None,
    base_url: Optional[str] = None,
    **kwargs
) -> AgentRunner:
    """
    Create an OpenAI agent in one line.
    
    Example:
        agent = create_openai_agent(
            api_key="sk-...",
            model="gpt-4o",
            system_prompt="You are a helpful coding assistant."
        )
        result = await agent.run_turn("Hello!", DefaultObserver())
    """

    llm_kwargs = {"base_url": base_url} if base_url else {}
    llm = OpenAILLM(api_key=api_key, model=model, **llm_kwargs, **kwargs)
    llm = OpenAILLM(api_key=api_key, model=model, **kwargs)
    memory = MemoryManager()
    memory.set_system_prompt(system_prompt)
    tools = ToolManager(mcp_config_path=mcp_config_path)
    observer = observer or DefaultObserver()
    
    return AgentRunner(llm_client=llm, tool_manager=tools, memory=memory)


def create_anthropic_agent(
    api_key: str,
    model: str = "claude-3-5-sonnet-20241022",
    system_prompt: str = "You are a helpful assistant.",
    mcp_config_path: Optional[str] = None,
    observer: Optional[AgentEventObserver] = None,
    **kwargs
) -> AgentRunner:
    """Create an Anthropic Claude agent in one line."""
    llm = AnthropicLLM(api_key=api_key, model=model, **kwargs)
    memory = MemoryManager()
    memory.set_system_prompt(system_prompt)
    tools = ToolManager(mcp_config_path=mcp_config_path)
    observer = observer or DefaultObserver()
    
    return AgentRunner(llm_client=llm, tool_manager=tools, memory=memory)


def create_ollama_agent(
    model: str = "llama3.1",
    system_prompt: str = "You are a helpful assistant.",
    base_url: Optional[str] = None,
    mcp_config_path: Optional[str] = None,
    observer: Optional[AgentEventObserver] = None,
    **kwargs
) -> AgentRunner:
    """Create an Ollama local agent in one line."""
    llm_kwargs = {"base_url": base_url} if base_url else {}
    llm = OllamaLLM(model=model, **llm_kwargs, **kwargs)
    memory = MemoryManager()
    memory.set_system_prompt(system_prompt)
    tools = ToolManager(mcp_config_path=mcp_config_path)
    observer = observer or DefaultObserver()
    
    return AgentRunner(llm_client=llm, tool_manager=tools, memory=memory)


async def chat(
    message: str,
    runner: Optional[AgentRunner] = None,
    provider: str = "openai",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    system_prompt: str = "You are a helpful assistant.",
    mcp_config_path: Optional[str] = None,
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
        model: Model name (defaults to provider's recommended)
        **kwargs: Any additional arguments to pass to the agent creation function.
    
    Returns:
        The agent's text response
    """
    observer = PrintObserver() if verbose else DefaultObserver()

    if runner: agent=runner
    elif provider == 'openai':
        agent = create_openai_agent(
            api_key=api_key or "dummy",
            model=model or "gpt-4o",
            base_url=base_url,
            system_prompt=system_prompt,
            mcp_config_path=mcp_config_path,
            observer=observer,
            **kwargs
        )
    elif provider == "anthropic":
        agent = create_anthropic_agent(
            api_key=api_key or "dummy",
            model=model or "claude-3-5-sonnet-20241022",
            system_prompt=system_prompt,
            mcp_config_path=mcp_config_path,
            observer=observer,
            **kwargs
        )
    elif provider == "ollama":
        agent = create_ollama_agent(
            model=model or "llama3.1",
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