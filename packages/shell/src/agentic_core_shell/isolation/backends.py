from __future__ import annotations

import asyncio
import os
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


class IsolationBackend(ABC):
    """Executes commands using an isolation mechanism.

    Implementations must be OS-universal *in API shape*; actual isolation may be OS-specific.
    """

    @abstractmethod
    async def run(
        self,
        *,
        command: str,
        timeout_s: float,
        cwd: str | None,
        env: dict[str, str] | None,
        # lower-level shell selection is handled by the backend or tool
    ) -> tuple[int | None, str]:
        """Return (exit_code, combined_output)."""


class LocalIsolationBackend(IsolationBackend):
    async def run(
        self,
        *,
        command: str,
        timeout_s: float,
        cwd: str | None,
        env: dict[str, str] | None,
    ) -> tuple[int | None, str]:
        # Local is implemented by the tool (shell_cmd selection). This backend is a stub.
        # The tool should bypass shell logic and call the tool's local execution path.
        raise RuntimeError("LocalIsolationBackend.run should be handled by ShellExecTool.")


@dataclass
class DockerIsolationConfig:
    image: str = "alpine:3.20"
    workdir: str = "/workspace"
    # If True, mounts cwd into container. Host-path is provided by ShellExecTool via cwd.
    mount_cwd: bool = False
    # Network
    disable_network: bool = True


class DockerIsolationBackend(IsolationBackend):
    def __init__(self, config: DockerIsolationConfig | None = None):
        self.config = config or DockerIsolationConfig()

        if shutil.which("docker") is None:
            raise RuntimeError("Docker is not installed or not on PATH.")

    async def run(
        self,
        *,
        command: str,
        timeout_s: float,
        cwd: str | None,
        env: dict[str, str] | None,
    ) -> tuple[int | None, str]:
        # We run inside Docker using `sh -lc` to interpret the user command.
        # For portability, we avoid relying on host shell.
        cfg = self.config

        # Prepare env args
        env_args: list[str] = []
        if env:
            # Keep it safe: docker -e only accepts KEY=VAL
            for k, v in env.items():
                env_args.extend(["-e", f"{k}={v}"])

        mount_args: list[str] = []
        workdir = cfg.workdir
        if cfg.mount_cwd and cwd:
            # Docker Desktop on Windows uses special mounting semantics; user must configure.
            # We'll still attempt a bind mount.
            mount_args = ["-v", f"{cwd}:{workdir}"]

        # Disable network if requested
        net_args: list[str] = []
        if cfg.disable_network:
            net_args = ["--network", "none"]

        # Command execution inside container
        # Use `sh -lc` so that typical shell syntax works.
        docker_cmd = [
            "docker",
            "run",
            "--rm",
            "-w",
            workdir,
            *net_args,
            *env_args,
            *mount_args,
            cfg.image,
            "sh",
            "-lc",
            command,
        ]

        proc = await asyncio.create_subprocess_exec(
            *docker_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return None, f"Error: Timeout after {timeout_s}s while executing command in docker."

        out = (stdout or b"").decode(errors="replace")
        return proc.returncode, out.strip() if out.strip() else ""

