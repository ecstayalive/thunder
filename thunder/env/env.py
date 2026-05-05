from __future__ import annotations

from dataclasses import dataclass, is_dataclass
from types import UnionType
from typing import (
    Any,
    Callable,
    Dict,
    get_args,
    get_type_hints,
    List,
    Optional,
    Tuple,
    TYPE_CHECKING,
    TypeVar,
)

import gymnasium as gym

from thunder.core import Executor

if TYPE_CHECKING:
    from gymnasium import Space
    from gymnasium.envs.registration import EnvSpec

    from thunder.env.typing import ActionType, ArrayType, ObservationType

_LOADER_REGISTRY: Dict[str, Callable[[Any], gym.Env]] = {}
_LOADER_SPEC_REGISTRY: Dict[str, type] = {}


@dataclass
class EnvLoaderSpec:
    """

    Args:

    """

    framework: str
    task: str
    num_envs: int = 1
    num_agents: int = 1
    seed: int = 0


class ThunderEnv(gym.Env):
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

    def __init__(self, env: gym.Env | gym.vector.VectorEnv):
        self.env: gym.Env | gym.vector.VectorEnv = env
        self._data_type: Optional[str] = None
        self._inbound_fn: callable = None
        self._outbound_fn: callable = None

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
        next_obs, reward, done, timeouts, info = self.env.step(self._inbound_fn(action))
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
        next_obs, reward, done, timeouts, info = self.env.step(self._inbound_fn(action))
        return (
            self.observation(self._outbound_fn(next_obs)),
            self._outbound_fn(reward),
            self._outbound_fn(done),
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
        next_obs, reward, done, timeouts, info = self.env.step(self._inbound_fn(action))
        return (
            self._outbound_fn(next_obs),
            self.reward(self._outbound_fn(reward)),
            self._outbound_fn(done),
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


def _iter_annotation_types(annotation: Any):
    if annotation is None:
        return
    origin = getattr(annotation, "__origin__", None)
    if origin is None:
        origin = getattr(annotation, "__class__", None)
    if origin is UnionType or str(origin) == "typing.Union":
        for arg in get_args(annotation):
            yield from _iter_annotation_types(arg)
        return
    yield annotation


def _infer_loader_spec_cls(loader: Callable) -> type | None:
    try:
        annotation = get_type_hints(loader).get("spec")
    except Exception:
        annotation = getattr(loader, "__annotations__", {}).get("spec")

    for candidate in _iter_annotation_types(annotation):
        if not isinstance(candidate, type):
            continue
        if candidate is EnvLoaderSpec:
            continue
        if issubclass(candidate, EnvLoaderSpec) and is_dataclass(candidate):
            return candidate
    return None


def register_loader(framework: str, spec_cls: type | None = None):
    """_summary_

    Args:
        framework (str): _description_
    """

    def decorator(func: Callable):
        _LOADER_REGISTRY[framework] = func
        resolved_spec_cls = spec_cls or _infer_loader_spec_cls(func)
        if resolved_spec_cls is not None:
            _LOADER_SPEC_REGISTRY[framework] = resolved_spec_cls
        return func

    return decorator


def get_loader_spec_cls(framework: str, default: type | None = None) -> type | None:
    if framework not in _LOADER_SPEC_REGISTRY:
        import importlib

        try:
            importlib.import_module(f"thunder.env.{framework}")
        except ModuleNotFoundError:
            return default
    return _LOADER_SPEC_REGISTRY.get(framework, default)


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
