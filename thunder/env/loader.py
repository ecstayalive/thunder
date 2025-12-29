from __future__ import annotations

import importlib
from typing import TYPE, Dict, Optional

from .interface import EnvWrapper

_LOADER_REGISTRY: Dict[str, str] = {
    "isaaclab": "thunder.env.isaaclab.isaaclab_loader",
    "dm_control": "thunder.env.dmc.dmc_loader",
    "gym": "thunder.env.gym.gym_loader",
}


def make_env(
    framework: str, task: str, wrapper: Optional[TYPE[EnvWrapper]] = None, **kwargs
) -> EnvWrapper:
    """Create an environment based on the specified framework and task.

    Args:
        framework (str): The environment framework to use (e.g., 'isaaclab', 'gym', 'dm_control').
        task (str): The specific task or environment name.
        wrapper (Optional[TYPE[EnvWrapper]]): An optional wrapper class to wrap the environment.

    Returns:
        An instance of the created environment, possibly wrapped.
    """
    if framework not in _LOADER_REGISTRY:
        raise ValueError(f"Unsupported environment framework: {framework}")
    module_path, loader_name = _LOADER_REGISTRY[framework].rsplit(".", 1)
    module = importlib.import_module(module_path)
    loader = getattr(module, loader_name)
    env = loader(task, **kwargs)
    if wrapper is not None:
        env = wrapper(env)
    return env
