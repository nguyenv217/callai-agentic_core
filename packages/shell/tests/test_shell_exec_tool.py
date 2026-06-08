import asyncio
import os

import pytest

from agentic_core_shell import ShellExecConfig, ShellExecTool

@pytest.mark.asyncio
async def test_shell_exec_echo_basic():
    tool = ShellExecTool(
        ShellExecConfig(
            allowlist_commands=None,  # command allowlisting is configured by the app using the tool
            blocklist_commands=None,
            timeout_s=5.0,
        )
    )

    res = await tool.execute({"command": "echo hello"}, context={})
    assert "hello" in res.lower()



@pytest.mark.asyncio
async def test_shell_exec_blocklist_blocks_executable():
    tool = ShellExecTool(
        ShellExecConfig(
            allowlist_commands=None,
            blocklist_commands=["python"],
            timeout_s=5.0,
        )
    )

    res = await tool.execute({"command": "python -c \"print(123)\""}, context={})
    assert "blocked" in res.lower()


@pytest.mark.asyncio
async def test_shell_exec_allowlist_blocks_non_allowed_executable():
    tool = ShellExecTool(
        ShellExecConfig(
            allowlist_commands=["python"],
            blocklist_commands=None,
            timeout_s=5.0,
        )
    )

    res = await tool.execute({"command": "echo hello"}, context={})
    assert "not in the allowlist" in res.lower()

@pytest.mark.asyncio
async def test_shell_exec_allowlist_blocks_chained_bypasses():
    tool = ShellExecTool(
        ShellExecConfig(
            allowlist_commands=["echo"],
            blocklist_commands=None,
            timeout_s=5.0,
        )
    )

    # The first command is allowed, but the chained one is not
    res = await tool.execute({"command": "echo hello ; rm -rf /"}, context={})
    assert "not in the allowlist" in res.lower()
    assert "rm" in res.lower()

    res2 = await tool.execute({"command": "echo hello | grep hello"}, context={})
    assert "not in the allowlist" in res2.lower()
    assert "grep" in res2.lower()


@pytest.mark.asyncio
async def test_shell_exec_timeout():

    # Use a command that should exceed the timeout. Keep it short to avoid long hangs.
    tool = ShellExecTool(
        ShellExecConfig(
            
            allowlist_commands=None,
            blocklist_commands=None,
            timeout_s=0.3,
        )
    )

    if os.name == "nt":
        # ping localhost with 5 echoes -> ~5 seconds
        cmd = "ping 127.0.0.1 -n 6 > nul"
    else:
        cmd = "sleep 2"

    res = await tool.execute({"command": cmd}, context={})
    assert "timeout" in res.lower()

