import torch
import torch.nn as nn

from thunder.core import Batch, ExecutionContext, ModelPack, Objective, Operation


class LeJepaObjective(Objective):
    def __init__(self, name="lejepa", weight=1):
        super().__init__(name, weight)

    def compute(self, batch: Batch, model: ModelPack):
        return 0.0, {}


class VicRegular(Objective):
    def __init__(self, name="vic_reg", weight=1, **kwargs):
        super().__init__(name, weight)

    def compute(self, batch: Batch, model: ModelPack):
        return 0.0, {}
