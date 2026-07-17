from __future__ import annotations

from dataclasses import dataclass, is_dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, TypeVar

import gymnasium as gym

from thunder.core import Executor

from .typing import ActionType, ArrayType, ObservationType

if TYPE_CHECKING:
    from gymnasium import Space
    from gymnasium.envs.registration import EnvSpec

_LOADER_REGISTRY: Dict[str, Callable[[Any], gym.Env]] = {}


@dataclass
class EnvLoaderSpec:
    """
    Attributes:
        framework: Framework identifier
        task: Registered environment.
        num_envs: Number of parallel environments to spawn.
        num_agents: Number of agents per environment (for multi-agent envs).
        seed: Random seed for environment initialization.
        device: Compute device string.
    """

    framework: str
    task: str
    num_envs: int = 1
    num_agents: int = 1
    seed: int = 0
    device: str = "cuda"


class ThunderEnv(gym.Env[ObservationType, ActionType]):
    """_summary_

    Raises:
        ValueError: _description_
        AttributeError: _description_

    Returns:
        _type_: _description_
    """

    _TYPE_MAP = {
        "numpy": "numpy",
        "torch": "torch",
        "jax": "jax",
        "jaxlib": "jax",
        "warp": "warp",
        "builtins": "numpy",
    }
    autoreset_mode: gym.vector.AutoresetMode = gym.vector.AutoresetMode.SAME_STEP
    # _train: bool = False

    def __init__(self, env: gym.Env | gym.vector.VectorEnv):
        self.env: gym.Env | gym.vector.VectorEnv = env
        self._data_type: Optional[str] = None
        self._inbound_fn: Callable[[Any], Any] = None
        self._outbound_fn: Callable[[Any], Any] = None

        self._action_space: Space[WrapperActType] | None = None
        self._observation_space: Space[WrapperObsType] | None = None
        self._metadata: dict[str, Any] | None = None
        self._cached_spec: EnvSpec | None = None

    def reset(
        self, seed=None, indices=None, options=None
    ) -> Tuple[ObservationType, Dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        if self._data_type is None:
            self._setup_bound(obs)
        return self._outbound_fn(obs), info

    def step(
        self, action: ActionType
    ) -> Tuple[ObservationType, ArrayType, ArrayType, ArrayType, Dict[str, Any]]:
        next_obs, reward, terminated, timeouts, info = self.env.step(
            self._inbound_fn(action)
        )
        return (
            self._outbound_fn(next_obs),
            self._outbound_fn(reward),
            self._outbound_fn(terminated),
            self._outbound_fn(timeouts),
            info,
        )

    def close(self):
        return self.env.close()

    def _setup_bound(self, sample_obs):
        """ """
        self._data_type = self.get_dtype(sample_obs)
        if self._data_type == Executor.backend:
            self._inbound_fn = lambda x: x
            self._outbound_fn = lambda x: x
            return
        match self._data_type:
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
                raise ValueError(f"Unsupported format: {self._data_type}")

    @staticmethod
    def get_dtype(data: Any) -> str:
        """ """
        if isinstance(data, (dict, list, tuple)):
            if not data:
                return "dict" if isinstance(data, dict) else "list"
            first = next(iter(data.values())) if isinstance(data, dict) else data[0]
            return ThunderEnv.get_dtype(first)
        root_module = type(data).__module__.partition(".")[0]
        return ThunderEnv._TYPE_MAP.get(root_module, "unknown")

    @property
    def unwrapped(self) -> gym.Env:
        return self.env.unwrapped

    @property
    def num_envs(self) -> int:
        return self.env.num_envs

    @property
    def action_space(self) -> gym.Space:
        return self.env.action_space

    @property
    def observation_space(self) -> gym.spaces.Dict:
        return self.env.observation_space

    @property
    def single_observation_space(self) -> gym.spaces.Dict:
        space = getattr(self.env.unwrapped, "single_observation_space", None)
        return space or self._unbatch_space(self.observation_space, self.num_envs)

    @property
    def single_action_space(self) -> gym.Space:
        space = getattr(self.env.unwrapped, "single_action_space", None)
        return space or self._unbatch_space(self.action_space, self.num_envs)

    @staticmethod
    def _unbatch_space(space: gym.Space, num_envs: int) -> gym.Space:
        """Strip the leading ``num_envs`` axis to yield a single-env space.

        Only removes the first axis when it actually equals ``num_envs``, so a
        space that is already per-env is returned unchanged.
        """
        if isinstance(space, gym.spaces.Dict):
            return gym.spaces.Dict(
                {
                    key: ThunderEnv._unbatch_space(sub, num_envs)
                    for key, sub in space.spaces.items()
                }
            )
        if isinstance(space, gym.spaces.Box) and space.shape[:1] == (num_envs,):
            return gym.spaces.Box(
                low=space.low[0],
                high=space.high[0],
                shape=space.shape[1:],
                dtype=space.dtype,
            )
        return space

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


WrapperObsType = TypeVar("WrapperObsType")
WrapperActType = TypeVar("WrapperActType")


class ObservationWrapper(ThunderEnv):
    """"""

    def reset(
        self, seed=None, indices=None, options=None
    ) -> Tuple[ObservationType, Dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        if self._data_type is None:
            self._setup_bound(obs)
        return self.observation(self._outbound_fn(obs)), info

    def step(
        self, action: ActionType
    ) -> Tuple[ObservationType, ArrayType, ArrayType, ArrayType, Dict[str, Any]]:
        next_obs, reward, terminated, timeouts, info = self.env.step(
            self._inbound_fn(action)
        )
        return (
            self.observation(self._outbound_fn(next_obs)),
            self._outbound_fn(reward),
            self._outbound_fn(terminated),
            self._outbound_fn(timeouts),
            info,
        )

    def observation(self, observation: ObservationType) -> WrapperObsType:
        """Returns a modified observation.

        Args:
            observation: The :attr:`env` observation

        Returns:
            The modified observation
        """
        raise NotImplementedError


class RewardWrapper(ThunderEnv):
    def step(
        self, action: ActionType
    ) -> Tuple[ObservationType, ArrayType, ArrayType, ArrayType, Dict[str, Any]]:
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        next_obs, reward, terminated, timeouts, info = self.env.step(
            self._inbound_fn(action)
        )
        return (
            self._outbound_fn(next_obs),
            self.reward(self._outbound_fn(reward)),
            self._outbound_fn(terminated),
            self._outbound_fn(timeouts),
            info,
        )

    def reward(self, reward: ArrayType) -> ArrayType:
        """Returns a modified environment ``reward``.

        Args:
            reward: The :attr:`env` :meth:`step` reward

        Returns:
            The modified `reward`
        """
        raise NotImplementedError


class ActionWrapper(ThunderEnv):
    """ """

    def step(
        self, action: ActionType
    ) -> Tuple[ObservationType, ArrayType, ArrayType, ArrayType, Dict[str, Any]]:
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        next_obs, reward, done, timeouts, info = self.env.step(
            self._inbound_fn(self.action(action))
        )
        return (
            self._outbound_fn(next_obs),
            self.reward(self._outbound_fn(reward)),
            self._outbound_fn(done),
            self._outbound_fn(timeouts),
            info,
        )

    def action(self, action: WrapperActType) -> ActionType:
        """Returns a modified action before :meth:`step` is called.

        Args:
            action: The original :meth:`step` actions

        Returns:
            The modified actions
        """
        raise NotImplementedError


def register_loader(framework: str):
    """_summary_

    Args:
        framework (str): _description_
    """

    def decorator(func: Callable):
        _LOADER_REGISTRY[framework] = func
        return func

    return decorator


def make_env(
    spec: EnvLoaderSpec, wrappers: Optional[List[ThunderEnv]] = None
) -> ThunderEnv:
    """ """
    if spec.framework not in _LOADER_REGISTRY:
        import importlib

        try:
            importlib.import_module(f"thunder.env.{spec.framework}")
        except ModuleNotFoundError:
            print(f"No framework named {spec.framework}")
            raise
    env = _LOADER_REGISTRY[spec.framework](spec)
    if wrappers:
        for wrapper_cls in wrappers:
            env = wrapper_cls(env)
    return env
