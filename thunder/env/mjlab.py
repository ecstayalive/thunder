from typing import Any, Dict, Tuple

import gymnasium

from .interface import EnvSpec, EnvWrapper
from .loader import register_loader


class MjLabSpec(EnvSpec):
    framework: str = "mjlab"
    task: str = "default"
    num_envs: int = 1
    num_agents: int = 1


@register_loader("mjlab")
def load_mjlab(spec: EnvSpec | MjLabSpec) -> EnvWrapper: ...
