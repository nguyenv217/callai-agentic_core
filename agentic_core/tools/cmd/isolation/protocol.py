from __future__ import annotations

from typing import TypedDict, Literal, NotRequired


class IsolationDockerConfig(TypedDict, total=False):
    type: Literal["docker"]
    image: NotRequired[str]
    workdir: NotRequired[str]
    mount_cwd: NotRequired[bool]
    disable_network: NotRequired[bool]


class IsolationConfig(TypedDict, total=False):
    """Tool-instance config for shell isolation."""

    # None means no isolation backend; tool runs locally.
    type: NotRequired[str]


# Helper union shape; used for typing only.
IsolationBackendConfig = IsolationDockerConfig

