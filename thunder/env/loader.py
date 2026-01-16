from typing import Any, Callable, Dict, List, Optional, Type

import gymnasium as gym

from .interface import BoundWrapper, EnvSpec

_LOADER_REGISTRY: Dict[str, Callable[[Any], gym.Env]] = {}


def register_loader(framework: str):
    def decorator(func: Callable):
        _LOADER_REGISTRY[framework] = func
        return func

    return decorator


def make_env(spec: EnvSpec, wrappers: Optional[List[Type[gym.Env]]] = None) -> gym.Env:
    """ """
    if spec.framework not in _LOADER_REGISTRY:
        import importlib

        try:
            importlib.import_module(f"thunder.env.{spec.framework}")
        except ModuleNotFoundError:
            print(f"No framework named {spec.framework}")
            raise
    env = BoundWrapper(_LOADER_REGISTRY[spec.framework](spec))
    if wrappers:
        for wrapper_cls in wrappers:
            env = wrapper_cls(env)
    return env
