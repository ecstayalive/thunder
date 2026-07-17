import os
from typing import Any, Dict, TYPE_CHECKING, TypeAlias

_BACKEND = os.getenv("THUNDER_BACKEND", "torch").lower()

if TYPE_CHECKING:
    import jax
    import torch

    ArrayType: TypeAlias = torch.Tensor | jax.Array
    ActionType: TypeAlias = ArrayType
    ObservationType: TypeAlias = Dict[str, ArrayType]

else:
    if _BACKEND == "torch":
        import torch

        ArrayType = torch.Tensor
    elif _BACKEND == "jax":
        import jax

        ArrayType = jax.Array
    elif _BACKEND == "warp":
        import warp as wp

        ArrayType = wp.array
    else:
        ArrayType = Any

    ActionType = ArrayType
    ObservationType = dict[str, ArrayType]

__all__ = ["ActionType", "ArrayType", "ObservationType"]
