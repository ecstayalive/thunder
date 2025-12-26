import importlib
import os
from typing import TYPE_CHECKING

_BACKEND = os.getenv("THUNDER_BACKEND", "torch").lower()

_REGISTRY = {
    "torch": ("._torch_impl", "TorchModule"),
    "jax": ("._jax_impl", "JaxModule"),
    "warp": ("._warp_impl", "WarpModule"),
}

if TYPE_CHECKING:
    if _BACKEND == "torch":
        from ._torch_impl import TorchModule as ThunderModule
    elif _BACKEND == "jax":
        from ._jax_impl import JaxModule as ThunderModule
    elif _BACKEND == "warp":
        from ._warp_impl import WarpModule as ThunderModule
    else:
        from .interface import ModuleProtocol as ThunderModule
else:
    if _BACKEND not in _REGISTRY:
        raise ValueError(
            f"Unknown THUNDER_BACKEND: {_BACKEND}. Available: {list(_REGISTRY.keys())}"
        )

    module_path, class_name = _REGISTRY[_BACKEND]

    try:
        _mod = importlib.import_module(module_path, package=__name__)
        ThunderModule = getattr(_mod, class_name)

    except (ImportError, ModuleNotFoundError) as e:
        _libs = {"torch": "torch", "jax": "jax/flax", "warp": "warp-lang"}
        raise ImportError(
            f"Current backend is set to '{_BACKEND}', but required libraries are missing.\n"
            f"Please install {_libs.get(_BACKEND, _BACKEND)} or run: pip install thunder[{_BACKEND}]"
        ) from e

__all__ = ["ThunderModule"]
