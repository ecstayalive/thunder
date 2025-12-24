from typing import Any, Optional


class WarpModule:
    """ """

    def __init__(self, module: Any, backend: str = "warp"):
        self._backend = backend
        self._module = module
        self.get_params = getattr(module, "parameters", lambda: [])

    def __call__(self, x, state: Optional[Any] = None, **kwargs):
        """ """
        if state is not None:
            return self._module.forward(x, state, **kwargs)
        return self._module.forward(x, **kwargs)

    def to(self, device):
        if hasattr(self._module, "to"):
            self._module.to(device)
