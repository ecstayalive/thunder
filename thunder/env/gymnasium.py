from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np

from .loader import EnvSpec, register_loader


class GymnasiumSpec(EnvSpec):
    framework: str = "gymnasium"
    task: str = "CartPole-v1"
    num_envs: int = 1
    num_agents: int = 1
    render_mode: str = "human"
    vectorize_mode: str = "async"


class GymnasiumAdaptor(gym.ObservationWrapper):
    def observation(self, observation):
        return {"policy": observation}


@register_loader("gymnasium")
def load_gym(spec: EnvSpec | GymnasiumSpec) -> gym.Env | gym.vector.VectorEnv:
    spec = GymnasiumSpec.parse(final=True)
    if spec.num_envs > 1:
        env = gym.make_vec(
            spec.task, spec.num_envs, spec.vectorize_mode, render_mode=spec.render_mode
        )
    else:
        env = gym.make(spec.task, render_mode=spec.render_mode)
    return GymnasiumAdaptor(env)
