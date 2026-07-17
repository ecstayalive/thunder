from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Iterable, Tuple, Type

import torch
import torch.nn as nn

__all__ = ["ModelSpec", "MultiModelSpec", "SequentialSpec"]


@dataclass
class ModelSpec(ABC):
    """Self-building spec for a feature-extracting model.

    Args:
        out_shape: Feature dim the model emits; consumed by a downstream head.
    """

    class_type: ClassVar[Type[nn.Module]]

    out_shape: int = 256

    @abstractmethod
    def factory(self, *in_shapes, **ctx) -> nn.Module:
        """Build the module. ``in_shapes`` are the per-input feature shapes
        (an ``int`` feature dim for vector inputs, a ``(C, H, W)`` tuple for
        images), one per obs key the owning Actor/Critic feeds in."""
        raise NotImplementedError


class Sequential(nn.Module):
    """Runs contract modules back-to-back, threading a per-block ``carry`` tuple."""

    def __init__(self, blocks: nn.Module):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *args):
        *inputs, carry = args
        carries = carry if carry is not None else (None,) * len(self.blocks)
        new_carries, step_inputs = [], tuple(inputs)
        for block, c in zip(self.blocks, carries):
            out, c = block(*step_inputs, c)
            step_inputs = (out,)
            new_carries.append(c)
        return out, tuple(new_carries)


@dataclass
class SequentialSpec(ModelSpec):
    """Composes several :class:`ModelSpec` blocks into one backbone."""

    blocks: Tuple[ModelSpec, ...] = ()

    class_type: ClassVar[Type[nn.Module]] = Sequential

    def __post_init__(self):
        if self.blocks:
            self.out_shape = self.blocks[-1].out_shape

    def factory(self, *in_shapes, **ctx) -> Sequential:
        modules, cur = [], in_shapes
        for block in self.blocks:
            modules.append(block.factory(*cur, **ctx))
            cur = (block.out_shape,)
        return Sequential(modules)


class MultiModel(nn.Module):
    """Encodes each input with its own block, concatenates the features, then runs a
    trunk — the multi-modal / asymmetric-input backbone (N inputs -> 1 feature).

    Threads a per-block carry tuple ``(*encoder_carries, trunk_carry)``, like
    :class:`Sequential`. Non-recurrent encoders pass their ``None`` carry through;
    the recurrent trunk holds the temporal memory.
    """

    def __init__(self, encoders: Iterable[nn.Module], trunk: nn.Module):
        super().__init__()
        self.encoders = nn.ModuleList(encoders)
        self.trunk = trunk

    def forward(self, *args):
        *inputs, carry = args
        carries = carry if carry is not None else (None,) * (len(self.encoders) + 1)
        features, new_carries = [], []
        for encoder, x, c in zip(self.encoders, inputs, carries):
            feature, c = encoder(x, c)
            features.append(feature)
            new_carries.append(c)
        out, trunk_carry = self.trunk(torch.cat(features, dim=-1), carries[-1])
        return out, (*new_carries, trunk_carry)


@dataclass
class MultiModelSpec(ModelSpec):
    """Encodes each obs key with its own model, concatenates the features, then a trunk.

    Maps N inputs to one feature — the multi-modal (e.g. image + state) and
    asymmetric-input case that a single block can't express.

    Args:
        encoders: One :class:`ModelSpec` per obs key, in ``obs_keys`` order.
        trunk: The model that consumes the concatenated encoder features.
    """

    encoders: Tuple[ModelSpec, ...] = ()
    trunk: ModelSpec | None = None

    class_type: ClassVar[Type[nn.Module]] = MultiModel

    def __post_init__(self):
        if self.trunk is not None:
            self.out_shape = self.trunk.out_shape

    def factory(self, *in_shapes, **ctx) -> MultiModel:
        if len(in_shapes) != len(self.encoders):
            raise ValueError(
                f"MultiModelSpec has {len(self.encoders)} encoders but received "
                f"{len(in_shapes)} inputs; one encoder per obs key is required."
            )
        encoders = [
            spec.factory(shape, **ctx) for spec, shape in zip(self.encoders, in_shapes)
        ]
        fused_shape = sum(spec.out_shape for spec in self.encoders)
        return MultiModel(encoders, self.trunk.factory(fused_shape, **ctx))
