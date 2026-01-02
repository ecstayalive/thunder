from typing import Any, Callable, Dict, List, Optional, Type

from .interface import EnvSpec, EnvWrapper

_LOADER_REGISTRY: Dict[str, Callable[[Any], EnvWrapper]] = {}


def register_loader(framework: str):
    def decorator(func: Callable):
        _LOADER_REGISTRY[framework] = func
        return func

    return decorator


def make_env(spec: EnvSpec, wrappers: Optional[List[Type[EnvWrapper]]] = None) -> EnvWrapper:
    """ """
    if spec.framework not in _LOADER_REGISTRY:
        import importlib

        importlib.import_module(f"thunder.env.{spec.framework}")

    env = _LOADER_REGISTRY[spec.framework](spec)
    if wrappers:
        for wrapper_cls in wrappers:
            env = wrapper_cls(env)

    return env
