from typing import Any, Dict, Tuple

import gymnasium

from .loader import EnvSpec, register_loader


class DmcSpec(EnvSpec):
    """ """

    framework: str = "dmc"
    task: str = ...
    num_envs: int = 1
    headless: bool = False


class DmcAdapter(gymnasium.Env):
    """ """

    def __init__(self, env: gymnasium.Env):
        self.env = env

    def reset(self) -> Tuple[Any, Dict]:
        return self.env.reset()

    def step(self, action: Any) -> Tuple[Any, Any, Any, Any, Dict]:
        return self.env.step(action)

    @property
    def unwrapped(self) -> gymnasium.Env:
        return self.env.unwrapped


@register_loader("dmc")
def load_dmc(spec: DmcSpec | EnvSpec) -> DmcAdapter:
    """ """
    spec = DmcSpec().parse(final=True)
    from dm_control import suite

    domain, task = spec.task.split("_", 1)
    env = suite.load(domain_name=domain, task_name=task)
    return DmcAdapter(env)
