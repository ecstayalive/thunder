import importlib
import os
from typing import TYPE_CHECKING

_BACKEND = os.getenv("THUNDER_BACKEND", "torch").lower()

_REGISTRY = {
    "torch": ("._torch_impl", "TorchExecutor"),
    "jax": ("._jax_impl", "JaxExecutor"),
    "warp": ("._warp_impl", "WarpExecutor"),
}

if TYPE_CHECKING:
    if _BACKEND == "torch":
        from ._torch_impl import TorchExecutor as Executor
    elif _BACKEND == "jax":
        from ._jax_impl import JaxExecutor as Executor
    elif _BACKEND == "warp":
        from ._warp_impl import WarpExecutor as Executor
    else:
        from .executor import ExecutorProtocol as Executor

else:
    if _BACKEND not in _REGISTRY:
        raise ValueError(
            f"Unknown THUNDER_BACKEND: {_BACKEND}. " f"Supported: {list(_REGISTRY.keys())}"
        )

    module_path, class_name = _REGISTRY[_BACKEND]

    try:
        _mod = importlib.import_module(module_path, package=__name__)
        Executor = getattr(_mod, class_name)

    except (ImportError, ModuleNotFoundError) as e:
        _hints = {
            "torch": "pip install thunder-rl[torch]",
            "jax": "pip install thunder-rl[jax]",
            "warp": "Make sure nvidia-warp is installed.",
        }
        raise ImportError(
            f"Failed to load backend '{_BACKEND}'.\n"
            f"Required implementation '{class_name}' in '{module_path}' could not be imported.\n"
            f"Hint: {_hints.get(_BACKEND, 'Check installation')}"
        ) from e

__all__ = ["Executor"]
