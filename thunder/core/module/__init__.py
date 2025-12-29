import importlib
import os
from collections import namedtuple
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


class ModelPack:
    """ """

    def __new__(cls, **kwargs):

        wrapped_kwargs = {}
        for k, v in kwargs.items():
            if not isinstance(v, ThunderModule):
                v = ThunderModule(v, k)
            else:
                if hasattr(v, "_name") and v._name != k:
                    v = v.replace(_name=k) if hasattr(v, "replace") else setattr(v, "_name", k) or v
            wrapped_kwargs[k] = v
        fields = sorted(wrapped_kwargs.keys())
        Pack = namedtuple("ModelPack", fields)
        return Pack(**wrapped_kwargs)


__all__ = ["ThunderModule"]
