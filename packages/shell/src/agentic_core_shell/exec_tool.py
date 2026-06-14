from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass

from typing import Any

from .isolation.backends import DockerIsolationBackend, DockerIsolationConfig


from agentic_core.tools import BaseTool


@dataclass
class ShellExecConfig:
    # Security
    allowlist_commands: list[str] | None = None  # exact executable names allowed
    blocklist_commands: list[str] | None = None  # exact executable names blocked

    # Optional OS enforcement for this tool instance.
    # - None: no explicit enforcement
    # - "windows" / "linux" / "darwin": require that platform
    # - "auto": accept current platform (default if you want to be explicit)
    os_support: str | None = None

    # True isolation (optional)
    # When set to {"type": "docker", ...} the command is run inside a container.
    # When None, the tool runs locally (best-effort isolation only).
    isolation: dict[str, Any] | None = None

    chdir: str | None = None
    env: dict[str, str] | None = None

    # Execution limits
    timeout_s: float = 15.0




import shlex

def _extract_executables(cmd_str: str) -> list[str]:
    """
    Extracts all executable names from a shell command string, parsing through
    pipes and logical operators to prevent command chaining bypasses.
    """
    cmd_str = cmd_str.strip()
    if not cmd_str:
        return []
        
    try:
        # posix=(os.name != "nt") preserves backslashes on Windows while still parsing quotes
        tokens = shlex.split(cmd_str, posix=(os.name != "nt"))
    except ValueError as e:
        raise ValueError(f"Malformed shell command: {e}")
        
    executables = []
    expect_exe = True
    
    for token in tokens:
        if token in (";", "&", "|", "&&", "||"):
            expect_exe = True
        elif expect_exe:
            # Skip environment variable assignments before a command
            if "=" in token and not token.startswith("="):
                continue
            executables.append(token)
            expect_exe = False
            
    return executables


class ShellExecTool(BaseTool):
    """Configurable shell command execution tool.

    Agent must only receive this tool name and its schema.
    Configuration is provided at tool-instance creation time.

    WARNING: This tool executes arbitrary commands and should be used with allowlists.
    """

    name = "shell_exec"

    schema = {
        "type": "function",
        "function": {
            "name": "shell_exec",
            "description": "Execute a shell command and return combined stdout/stderr. Uses ShellExecConfig allow/block lists and optional isolation backend.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute. If isolated, files created in the workspace persist and sync with host. First word validated against allowlist.",
                    },
                    "timeout_s": {
                        "type": "number",
                        "description": "Optional per-call timeout override in seconds (overrides ShellExecConfig.timeout_s).",
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Optional per-call working directory (overrides ShellExecConfig.chdir).",
                    },
                    "env": {
                        "type": "object",
                        "description": "Optional per-call extra environment variables (merged with ShellExecConfig.env).",
                        "additionalProperties": {"type": "string"},
                    },
                },
                "required": ["command"],
            },
        },
    }


    def __init__(
        self,
        config: ShellExecConfig | None = None,
    ):
        self._config = config or ShellExecConfig()
        self._backend = None
        self._backend_error = None
        
        iso_cfg = self._config.isolation
        if iso_cfg and str(iso_cfg.get("type", "")).lower() == "docker":
            try:
                docker_cfg = DockerIsolationConfig(
                    image=iso_cfg.get("image", "ubuntu:latest"),
                    container_name=iso_cfg.get("container_name") or None,
                    container_cmd=iso_cfg.get("container_cmd", ""),
                    workdir=iso_cfg.get("workdir", "/workspace"),
                    mount_cwd=bool(iso_cfg.get("mount_cwd", False)),
                    disable_network=bool(iso_cfg.get("disable_network", False)),
                    persistent_container=bool(iso_cfg.get("persistent_container", True)),
                    setup_commands=iso_cfg.get("setup_commands", []),
                    volumes=iso_cfg.get("volumes", []),
                    publish_ports=iso_cfg.get("publish_ports", []),
                    privileged=bool(iso_cfg.get("privileged", False)),
                    extra_args=iso_cfg.get("extra_args", []),
                    user=iso_cfg.get("user", ""),
                    env=iso_cfg.get("env", {}),
                )
                self._backend = DockerIsolationBackend(docker_cfg)
            except Exception as e:
                self._backend_error = str(e)
                self._backend = None

    async def execute(self, args: dict, context: dict) -> str:
        cmd = str(args.get("command", "")).strip()
        if not cmd:
            return "Error: 'command' is required and must be a non-empty string."

        # OS-support enforcement (tool-instance config)
        if self._config.os_support:
            os_support = self._config.os_support.lower()
            current_os = "windows" if os.name == "nt" else ("darwin" if sys.platform == "darwin" else "linux")

            if os_support == "auto":
                pass
            elif os_support != current_os:
                return f"Error: ShellExecConfig.os_support='{self._config.os_support}' not supported on this OS (current: {current_os})."

        try:
            executables = _extract_executables(cmd)
        except ValueError as e:
            return f"Error: {e}"

        if not executables:
            return "Error: No executable command found in the input."

        allowlist = {c.lower() for c in self._config.allowlist_commands} if self._config.allowlist_commands is not None else None
        blocklist = {c.lower() for c in self._config.blocklist_commands} if self._config.blocklist_commands is not None else None

        for exe in executables:
            exe_name = os.path.basename(exe).lower()
            
            if allowlist is not None and exe_name not in allowlist:
                return f"Error: Command executable '{exe_name}' is not in the allowlist. All chained commands must be allowed."
                
            if blocklist is not None and exe_name in blocklist:
                return f"Error: Command executable '{exe_name}' is blocked."

        timeout_s = float(args.get("timeout_s", self._config.timeout_s))

        cwd = args.get("cwd") or self._config.chdir
        env = {}
        if self._config.env:
            env.update(self._config.env)
        extra_env = args.get("env")
        if isinstance(extra_env, dict):
            env.update({str(k): str(v) for k, v in extra_env.items()})

        if getattr(self, "_backend_error", None):
            return f"Error: Failed to initialize docker isolation backend: {self._backend_error}. If you don't have Docker installed, please change the isolation type to 'None (Local)' in the TUI Settings."

        if self._backend:
            try:
                rc, out = await self._backend.run(
                    command=cmd,
                    timeout_s=timeout_s,
                    cwd=cwd,
                    env=env if env else None,
                )
                if rc is None:
                    return out or "Error: Unknown docker timeout failure."
                return out or f"(Command exited with code {rc} with no output.)"
            except Exception as e:
                return f"Error: {type(e).__name__}: {e}"

        # Local execution path (best-effort)
        try:
            if os.name != "nt":
                # Use explicit SHELL on POSIX if available
                shell = os.environ.get("SHELL") or "/bin/sh"
                proc = await asyncio.create_subprocess_exec(
                    shell, "-c", cmd,
                    cwd=cwd,
                    env={**os.environ, **env} if env else None,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
            else:
                # Native shell execution prevents list2cmdline quote mutilation on Windows
                proc = await asyncio.create_subprocess_shell(
                    cmd,
                    cwd=cwd,
                    env={**os.environ, **env} if env else None,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )

            try:
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                return f"Error: Timeout after {timeout_s}s while executing command."

            out = (stdout or b"").decode(errors="replace")
            return out.strip() if out.strip() else (f"(Command exited with code {proc.returncode} with no output.)")

        except FileNotFoundError:
            return "Error: Shell executable not found on this system."
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"


