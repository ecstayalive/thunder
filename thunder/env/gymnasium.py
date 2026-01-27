from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from thunder.utils import ArgBase

from .loader import EnvLoaderSpec, ObservationWrapper, register_loader


class GymVecLoaderSpec(ArgBase):
    autoreset_mode: gym.vector.AutoresetMode = gym.vector.AutoresetMode.NEXT_STEP


class GymnasiumLoaderSpec(EnvLoaderSpec):
    framework: str = "gymnasium"
    task: str = "CartPole-v1"
    num_envs: int = 1
    num_agents: int = 1
    render_mode: str = "human"
    vectorize_mode: str = "async"
    vector_kwargs: Optional[GymVecLoaderSpec] = None


class GymnasiumAdaptor(ObservationWrapper):
    def observation(self, observation):
        return {"policy": observation}


@register_loader("gymnasium")
def load_gym(spec: EnvLoaderSpec | GymnasiumLoaderSpec) -> GymnasiumAdaptor:
    spec: GymnasiumLoaderSpec = spec.to(GymnasiumLoaderSpec, final=True)
    if spec.num_envs > 1:
        env = gym.make_vec(
            spec.task,
            spec.num_envs,
            spec.vectorize_mode,
            render_mode=spec.render_mode,
            vector_kwargs=spec.vector_kwargs.to_dict(),
        )
    else:
        env = gym.make(spec.task, render_mode=spec.render_mode)
    env = GymnasiumAdaptor(env)
    env.autoreset_mode = spec.vector_kwargs.autoreset_mode
    return env
