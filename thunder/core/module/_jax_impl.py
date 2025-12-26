from functools import lru_cache, partial
from typing import Any, Dict, Optional, Tuple

import flax.linen as nn
from flax import struct


@struct.dataclass
class JaxModule:
    """ """

    _module: Any = struct.field(pytree_node=False)
    _name: str = struct.field(pytree_node=False)
    _backend: str = struct.field(pytree_node=False, default="jax")

    def init(self, key, *args, **kwargs):
        return self._module.init(key, *args, **kwargs)

    def _prepare_params(self, state: Any) -> Dict[str, Any]:
        """ """
        if isinstance(state, dict) and self._name in state:
            p = state[self._name]
        else:
            p = state.params if hasattr(state, "params") else state

        return {"params": p}

    def __call__(self, x: Any, state: Any, carry: Any = None, **kwargs: Any) -> Any:
        params = self._prepare_params(state)
        if carry is None:
            return self._module.apply(params, x, **kwargs)
        else:
            return self._module.apply(params, x, carry, **kwargs)

    @lru_cache(maxsize=None)
    def __getattr__(self, name: str):
        """ """
        if not hasattr(self._module, name):
            raise AttributeError(f"Flax module has no attribute '{name}'")
        method_fn = getattr(self._module, name)

        def wrapper(*args, state: Any, carry: Any = None, **kwargs: Any):
            params = self._prepare_params(state)
            if carry is None:
                return self._module.apply(params, *args, method=method_fn, **kwargs)
            else:
                return self._module.apply(params, *args, carry, method=method_fn, **kwargs)

        return wrapper

    def get_params(self):
        """ """
        return None
