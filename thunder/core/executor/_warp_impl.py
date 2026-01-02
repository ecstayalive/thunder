from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

import numpy as np
import warp as wp

if TYPE_CHECKING:
    from ..context import ExecutionContext
    from ..data import Batch
    from ..module._warp_impl import WarpModule


class WarpExecutor:
    """ """

    def __init__(self, device: str = "cuda", mixed_precision: bool = False):
        wp.init()
        self.device = device
        self.mixed_precision = mixed_precision

    def to_device(self, data: Any) -> Any:
        if isinstance(data, np.ndarray):
            return wp.array(data, device=self.device)
        return data

    def to_numpy(self, data: Any) -> Any:
        if isinstance(data, wp.array):
            return data.numpy()
        return data

    def optimize(
        self,
        ctx: ExecutionContext,
        opt: str,
        objectives: list,
        max_grad_norm: float = 1.0,
    ) -> tuple[dict, Any, Any]:

        optimizer = ctx.opt_states[opt]
        model = ctx.models
        batch = ctx.batch

        tape = wp.Tape()
        metrics = {}

        with tape:
            total_loss = wp.zeros(1, dtype=wp.float32, device=self.device, requires_grad=True)
            for obj in objectives:
                l, m = obj.forward(batch, model)
                pass
            pass
        tape.backward(loss=total_loss)
        pass
        return metrics

    def init(model: WarpModule, batch: Batch, optim_config: Dict[str, Any]):
        pass
