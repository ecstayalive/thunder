from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Type

import gymnasium as gym

from thunder.core import Executor
from thunder.utils import ArgBase

_LOADER_REGISTRY: Dict[str, Callable[[Any], gym.Env]] = {}


class EnvLoaderSpec(ArgBase):
    """
    Args:

    """

    framework: str = ...
    task: str = ...
    num_envs: int = 1
    num_agents: int = 1
    seed: int = 0


class ThunderWrapper:
    """_summary_

    Raises:
        ValueError: _description_
        AttributeError: _description_

    Returns:
        _type_: _description_
    """

    _FORMAT_MAP = {
        "numpy": "numpy",
        "torch": "torch",
        "jax": "jax",
        "jaxlib": "jax",
        "warp": "warp",
        "builtins": "numpy",
    }

    def __init__(self, env):
        self.env: gym.Env | gym.vector.VectorEnv = env
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

    def close(self):
        return self.env.close()

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
            return ThunderWrapper.get_dtype(first)
        root_module = type(data).__module__.partition(".")[0]
        return ThunderWrapper._FORMAT_MAP.get(root_module, "unknown")

    @property
    def unwrapped(self):
        return self.env.unwrapped

    @property
    def num_envs(self):
        return self.env.num_envs

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    def __getattr__(self, name: str):
        """
        If a method/attribute is not found in ThunderWrapper,
        automatically look for it in the underlying self.env.
        This enables calling env.render(), env.close(), env.num_envs,
        or simulator-specific methods like env.update_terrain() directly.
        """
        if name.startswith("_"):
            raise AttributeError(f"attempted to get missing private attribute '{name}'")
        return getattr(self.env, name)


def register_loader(framework: str):
    """_summary_

    Args:
        framework (str): _description_
    """

    def decorator(func: Callable):
        _LOADER_REGISTRY[framework] = func
        return func

    return decorator


def make_env(spec: EnvLoaderSpec, wrappers: Optional[List[Type[ThunderWrapper]]] = None):
    """ """
    if spec.framework not in _LOADER_REGISTRY:
        import importlib

        try:
            importlib.import_module(f"thunder.env.{spec.framework}")
        except ModuleNotFoundError:
            print(f"No framework named {spec.framework}")
            raise
    env = ThunderWrapper(_LOADER_REGISTRY[spec.framework](spec))
    if wrappers:
        for wrapper_cls in wrappers:
            env = wrapper_cls(env)
    return env
