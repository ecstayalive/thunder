from typing import Any, Dict, Tuple

import gymnasium

from .loader import EnvLoaderSpec, ThunderEnvWrapper, register_loader


class MjLabLoaderSpec(EnvLoaderSpec):
    framework: str = "mjlab"
    task: str = "default"
    num_envs: int = 1
    num_agents: int = 1


@register_loader("mjlab")
def load_mjlab(spec: EnvLoaderSpec | MjLabLoaderSpec) -> ThunderEnvWrapper: ...
