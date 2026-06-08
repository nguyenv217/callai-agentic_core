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
    image: str = "ubuntu:latest"
    workdir: str = "/workspace"
    mount_cwd: bool = True
    disable_network: bool = False
    persistent_container: bool = True
    container_name: str | None = None
    setup_commands: list[str] | None = None
    volumes: list[str] | None = None
    user: str | None = None
    env: dict[str, str] | None = None

class DockerIsolationBackend(IsolationBackend):
    def __init__(self, config: DockerIsolationConfig | None = None):
        self.config = config or DockerIsolationConfig()
        if shutil.which("docker") is None:
            raise RuntimeError("Docker is not installed or not on PATH.")
        self._container_started = False
        self._container_name = self.config.container_name or f"callai_sandbox_{os.getpid()}_{id(self)}"
        self._cleaned_up = False
        
        if self.config.persistent_container:
            import atexit
            atexit.register(self._sync_cleanup)

    def _sync_cleanup(self):
        if self._container_started and not self._cleaned_up:
            import subprocess
            # Execute asynchronously from the GC thread's perspective via Popen 
            # to prevent hanging the interpreter shutdown phase
            subprocess.Popen(
                ["docker", "rm", "-f", self._container_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            self._cleaned_up = True

    def __del__(self):
        self._sync_cleanup()
        
    async def _ensure_container(self, cwd: str | None = None):
        if self._container_started:
            return
        
        cfg = self.config
        mount_args = []
        self._mounted_cwd = None
        if cfg.mount_cwd and cwd:
            self._mounted_cwd = os.path.abspath(cwd)
            mount_args = ["-v", f"{self._mounted_cwd}:{cfg.workdir}"]
            
        net_args = ["--network", "none"] if cfg.disable_network else []
        
        env_args = []
        if cfg.env:
            for k, v in cfg.env.items():
                env_args.extend(["-e", f"{k}={v}"])
                
        vol_args = []
        if cfg.volumes:
            for v in cfg.volumes:
                vol_args.extend(["-v", v])
                
        user_args = ["-u", cfg.user] if cfg.user else []

        cmd = [
            "docker", "run", "-d", "--rm",
            "--name", self._container_name,
            "-w", cfg.workdir,
            *net_args,
            *user_args,
            *env_args,
            *vol_args,
            *mount_args,
            cfg.image,
            "tail", "-f", "/dev/null"
        ]
        
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=600.0)
        except asyncio.TimeoutError:
            proc.kill()
            raise RuntimeError("Timeout while pulling docker image or starting container.")
            
        if proc.returncode != 0:
            raise RuntimeError(f"Failed to start persistent docker container: {out.decode('utf-8', errors='replace')}")
            
        if cfg.setup_commands:
            for scmd in cfg.setup_commands:
                setup_proc = await asyncio.create_subprocess_exec(
                    "docker", "exec", "-w", cfg.workdir, self._container_name, "sh", "-lc", scmd,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
                )
                out, _ = await setup_proc.communicate()
                if setup_proc.returncode != 0:
                    raise RuntimeError(f"Setup command '{scmd}' failed: {out.decode('utf-8', errors='replace')}")
        
        self._container_started = True

    async def run(
        self,
        *,
        command: str,
        timeout_s: float,
        cwd: str | None,
        env: dict[str, str] | None,
    ) -> tuple[int | None, str]:
        cfg = self.config
        env_args: list[str] = []
        if env:
            for k, v in env.items():
                env_args.extend(["-e", f"{k}={v}"])

        if cfg.persistent_container:
            await self._ensure_container(cwd)
            
            container_cwd = cfg.workdir
            if cfg.mount_cwd and cwd and self._mounted_cwd:
                abs_cwd = os.path.abspath(cwd)
                if abs_cwd.startswith(self._mounted_cwd):
                    rel_path = os.path.relpath(abs_cwd, self._mounted_cwd)
                    if rel_path != ".":
                        container_cwd = os.path.join(cfg.workdir, rel_path).replace("\\", "/")
                        
            docker_cmd = [
                "docker", "exec",
                "-w", container_cwd,
                *env_args,
                self._container_name,
                "sh", "-lc", command
            ]
        else:
            mount_args: list[str] = []
            container_cwd = cfg.workdir
            if cfg.mount_cwd and cwd:
                abs_cwd = os.path.abspath(cwd)
                mount_args = ["-v", f"{abs_cwd}:{cfg.workdir}"]
            net_args: list[str] = ["--network", "none"] if cfg.disable_network else []
            
            vol_args = []
            if cfg.volumes:
                for v in cfg.volumes:
                    vol_args.extend(["-v", v])
            user_args = ["-u", cfg.user] if cfg.user else []
            container_env_args = []
            if cfg.env:
                for k, v in cfg.env.items():
                    container_env_args.extend(["-e", f"{k}={v}"])

            full_command = command
            if cfg.setup_commands:
                chained_setup = " && ".join(cfg.setup_commands)
                full_command = f"{chained_setup} && {command}"

            docker_cmd = [
                "docker", "run", "--rm",
                "-w", container_cwd,
                *net_args,
                *user_args,
                *container_env_args,
                *env_args,
                *vol_args,
                *mount_args,
                cfg.image,
                "sh", "-lc", full_command
            ]

        proc = await asyncio.create_subprocess_exec(
            *docker_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        except asyncio.TimeoutError:
            try:
                proc.kill()
                await proc.communicate()
            except Exception:
                pass
            return None, f"Error: Timeout after {timeout_s}s while executing command in docker."

        out = (stdout or b"").decode(errors="replace")
        return proc.returncode, out.strip() if out.strip() else ""

