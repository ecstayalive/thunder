from __future__ import annotations

from typing import Any, Dict, Tuple

import gymnasium

from thunder.utils import ArgBase


class EnvSpec(ArgBase):
    framework: str = ...
    task: str = ...
    num_envs: int = 1
    num_agents: int = 1
    seed: int = 0


class EnvWrapper:
    """ """

    def __init__(self, env: gymnasium.Env):
        self.env = env

    def reset(self) -> Tuple[Any, Dict]:
        return self.env.reset()

    def step(self, action: Any) -> Tuple[Any, Any, Any, Any, Dict]:
        return self.env.step(action)

    def render(self): ...

    @property
    def unwrapped(self) -> Any:
        return self.env.unwrapped
