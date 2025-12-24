from functools import partial
from typing import Any, Dict, Optional, Tuple

import flax.linen as nn
from flax import struct


@struct.dataclass
class JaxModule:
    """ """

    _module: Any = struct.field(pytree_node=False)
    _backend: str = struct.field(pytree_node=False, default="jax")

    def init(self, key, *args, **kwargs):
        return self._module.init(key, *args, **kwargs)

    def _prepare_params(self, state: Any) -> Dict[str, Any]:
        p = state.params if hasattr(state, "params") else state
        return {"params": p}

    def __call__(self, x: Any, state: Any, carry: Any = None, **kwargs: Any) -> Any:
        params = self._prepare_params(state)
        if carry is None:
            return self._module.apply(params, x, **kwargs)
        else:
            return self._module.apply(params, x, carry, **kwargs)

    def __getattr__(self, name: str):
        """ """
        if not hasattr(self._module, name):
            raise AttributeError(
                f"Flax module '{self._module.__class__.__name__}' has no attribute '{name}'"
            )
        method_fn = getattr(self._module, name)

        def wrapper(*args, state: Any, carry: Any = None, **kwargs: Any):
            """ """
            variables = self._prepare_params(state)
            if carry is None:
                return self._module.apply(variables, *args, method=method_fn, **kwargs)
            else:
                return self._module.apply(variables, *args, carry, method=method_fn, **kwargs)

        return wrapper

    def get_params(self):
        """ """
        return None
