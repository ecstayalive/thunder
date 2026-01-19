import torch
import torch.nn as nn

from thunder.core import Batch, ExecutionContext, ModelPack, Objective, Operation

class SIGRegObj(Objective):
    def __init__(self, name="sigreg", weight=1):
        super().__init__(name, weight)

    def compute(self, batch: Batch, model: ModelPack): ...


class LeJepaObjective(Objective):
    def __init__(self, name="lejepa", weight=1):
        super().__init__(name, weight)

    def compute(self, batch: Batch, model: ModelPack):
        return 0.0, {}


class VicReg(Objective):
    def __init__(self, name="vic_reg", weight=1, **kwargs):
        super().__init__(name, weight)

    def compute(self, batch: Batch, model: ModelPack):
        return 0.0, {}
