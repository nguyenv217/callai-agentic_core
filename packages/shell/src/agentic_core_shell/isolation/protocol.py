from __future__ import annotations

from typing import TypedDict, Literal


class IsolationDockerConfig(TypedDict, total=False):
    type: Literal["docker"]
    image: str | None = None
    container_name: str | None = None
    container_cmd: str | None = None
    workdir: str | None = None
    mount_cwd: bool | None = None
    disable_network: bool | None = None
    persistent_container: bool | None = None
    setup_commands: list[str] | None = None
    volumes: list[str] | None = None
    publish_ports: list[str] | None = None
    privileged: bool | None = None
    extra_args: list[str] | None = None
    user: str | None = None
    env: dict[str, str] | None = None


class IsolationConfig(TypedDict, total=False):
    """Tool-instance config for shell isolation."""

    # None means no isolation backend; tool runs locally.
    type: str | None = None


# Helper union shape; used for typing only.
IsolationBackendConfig = IsolationDockerConfig

