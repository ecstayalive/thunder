from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym

from thunder.core import Executor
from thunder.utils import ArgBase


class EnvSpec(ArgBase):
    framework: str = ...
    task: str = ...
    num_envs: int = 1
    num_agents: int = 1
    seed: int = 0


class BoundWrapper(gym.Wrapper):

    _FORMAT_MAP = {
        "numpy": "numpy",
        "torch": "torch",
        "jax": "jax",
        "jaxlib": "jax",
        "warp": "warp",
        "builtins": "numpy",
    }

    def __init__(self, env):
        super().__init__(env)
        self._data_format: Optional[str] = None
        self._inbound_fn: callable = None
        self._outbound_fn: callable = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self._data_format is None:
            self._setup_bound(obs)
        return self._outbound_fn(obs), info

    def step(self, action):
        env_action = self._inbound_fn(action)
        next_obs, reward, done, timeouts, info = self.env.step(env_action)
        return (
            self._outbound_fn(next_obs),
            self._outbound_fn(reward),
            self._outbound_fn(done),
            self._outbound_fn(timeouts),
            info,
        )

    def _setup_bound(self, sample_obs):
        """ """
        self._data_format = self.get_dtype(sample_obs)
        if self._data_format == Executor.backend:
            self._inbound_fn = lambda x: x
            self._outbound_fn = lambda x: x
            return
        match self._data_format:
            case "numpy" | "list":
                self._inbound_fn = Executor.to_numpy
                self._outbound_fn = Executor.from_numpy
            case "torch":
                self._inbound_fn = Executor.to_torch
                self._outbound_fn = Executor.from_torch
            case "warp":
                self._inbound_fn = Executor.to_warp
                self._outbound_fn = Executor.from_warp
            case _:
                raise ValueError(f"Unsupported format: {self._data_format}")

    @staticmethod
    def get_dtype(data: Any) -> str:
        """ """
        if isinstance(data, (dict, list, tuple)):
            if not data:
                return "dict" if isinstance(data, dict) else "list"
            first = next(iter(data.values())) if isinstance(data, dict) else data[0]
            return BoundWrapper.get_dtype(first)
        root_module = type(data).__module__.partition(".")[0]
        return BoundWrapper._FORMAT_MAP.get(root_module, "unknown")
