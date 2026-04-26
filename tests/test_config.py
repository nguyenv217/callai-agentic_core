import pytest
from agentic_core.config import RunnerConfig, ConfigurationError

def test_runner_config_invalid_iterations():
    """Test that max_iterations < 1 raises a ValueError."""
    with pytest.raises(ValueError, match="`max_iterations` must be >= 1"):
        RunnerConfig(max_iterations=0)

def test_runner_config_mcp_preload_without_servers():
    """Test that preloading tools without active servers raises an error."""
    with pytest.raises(ConfigurationError, match="The hosting servers of some tools.*are not found"):
        RunnerConfig(mcp_preload_tools=["github_create_issue"])

def test_runner_config_mcp_preload_mismatched_servers():
    """Test that preloading tools with mismatched active servers raises an error."""
    with pytest.raises(ConfigurationError, match="The hosting servers of some tools.*are not found"):
        RunnerConfig(
            mcp_preload_tools=["github_create_issue"], 
            mcp_active_servers=["sqlite"]
        )

def test_runner_config_mcp_valid_preload():
    """Test that valid MCP preload combinations do not raise errors."""
    config = RunnerConfig(
        mcp_preload_tools=["github_create_issue"], 
        mcp_active_servers=["github", "sqlite"]
    )
    assert "github_create_issue" in config.mcp_preload_tools
    assert "github" in config.mcp_active_servers