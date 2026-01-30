import importlib
import os
from typing import TYPE_CHECKING, Any

_BACKEND = os.getenv("THUNDER_BACKEND", "torch").lower()

_REGISTRY = {
    "torch": ("._torch_impl", "TorchModule", "TorchModelPack"),
    "jax": ("._jax_impl", "JaxModule", "JaxModelPack"),
    "warp": ("._warp_impl", "WarpModule", "WarpModelPack"),
}

if TYPE_CHECKING:
    if _BACKEND == "torch":
        from ._torch_impl import TorchModelPack as ModelPack
        from ._torch_impl import TorchModule as ThunderModule
    elif _BACKEND == "jax":
        from ._jax_impl import JaxModelPack as ModelPack
        from ._jax_impl import JaxModule as ThunderModule
    elif _BACKEND == "warp":
        # from ._warp_impl import WarpModelPack as ModelPack
        from ._warp_impl import WarpModule as ThunderModule
    else:
        from .interface import ModulePackProtocol as ModelPack
        from .interface import ModuleProtocol as ThunderModule

else:
    if _BACKEND not in _REGISTRY:
        raise ValueError(
            f"Unknown THUNDER_BACKEND: {_BACKEND}. Available: {list(_REGISTRY.keys())}"
        )
    module_path, tm_class_name, mp_class_name = _REGISTRY[_BACKEND]

    try:
        _mod = importlib.import_module(module_path, package=__name__)

        ThunderModule = getattr(_mod, tm_class_name)
        if mp_class_name and hasattr(_mod, mp_class_name):
            ModelPack = getattr(_mod, mp_class_name)
        else:
            ModelPack = None
    except (ImportError, ModuleNotFoundError) as e:
        _libs = {"torch": "torch", "jax": "jax/flax", "warp": "warp-lang"}
        raise ImportError(
            f"Current backend is set to '{_BACKEND}', but required libraries are missing.\n"
            f"Please install {_libs.get(_BACKEND, _BACKEND)} or run: pip install thunder[{_BACKEND}]"
        ) from e

__all__ = ["ThunderModule", "ModelPack"]


# class ModelPack:
#     """ """

#     def __new__(cls, **kwargs):

#         wrapped_kwargs = {}
#         for k, v in kwargs.items():
#             # if not isinstance(v, ThunderModule):
#             #     v = ThunderModule(v)
#             wrapped_kwargs[k] = v
#         fields = sorted(wrapped_kwargs.keys())
#         Pack = namedtuple("ModelPack", fields)
#         return Pack(**wrapped_kwargs)
