from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, replace
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, TypeVar

if TYPE_CHECKING:
    from .executor.interface import Executor

_BACKEND = os.getenv("THUNDER_BACKEND", "torch").lower()
TBatch = TypeVar("TBatch", bound="Batch")


@dataclass(slots=True)
class Batch:
    obs: Dict[str, Any] = None
    actions: Optional[Any] = None
    rewards: Optional[Any] = None
    dones: Optional[Any] = None
    timeouts: Optional[Any] = None
    mask: Optional[Any] = None
    next_obs: Optional[Dict[str, Any]] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, name: str) -> Any:
        """ """
        try:
            extra = object.__getattribute__(self, "extra")
            return extra[name]
        except (AttributeError, KeyError):
            raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        """
        WARNING: In JAX loops (scan/while_loop), adding NEW keys dynamically
        changes the Pytree structure and will cause compilation errors.
        """
        if name in self.__class__.__dataclass_fields__:
            object.__setattr__(self, name, value)
        else:
            self.extra[name] = value

    def __setitem__(self, key: str, value: Any) -> None:
        """ """
        setattr(self, key, value)

    def __getitem__(self, key: str) -> Any:
        """ """
        return getattr(self, key)

    def __dir__(self):
        """ """
        return list(self.__class__.__dataclass_fields__.keys()) + list(self.extra.keys())

    def map(self: TBatch, fn: Callable[[Any], Any]) -> TBatch:
        """ """

        def _recursive_apply(val):
            if val is None:
                return None
            if isinstance(val, dict):
                return {k: _recursive_apply(v) for k, v in val.items()}
            if isinstance(val, (list, tuple)):
                return type(val)(_recursive_apply(v) for v in val)
            return fn(val)

        changes = {}
        for f in fields(self):
            if f.name == "extra":
                continue
            val = getattr(self, f.name)
            changes[f.name] = _recursive_apply(val)
        if self.extra:
            changes["extra"] = _recursive_apply(self.extra)
        return replace(self, **changes)

    def to(self: TBatch, executor: Executor) -> TBatch:
        """ """
        return self.map(executor.to_device)

    def __repr__(self) -> str:
        def _fmt(v):
            if hasattr(v, "shape"):
                return f"Arr{tuple(v.shape)}"
            if isinstance(v, dict):
                return f"Dict[{len(v)}]"
            if isinstance(v, (list, tuple)):
                return f"{type(v).__name__}[{len(v)}]"
            return str(v)

        core = []
        for f in fields(self):
            if f.name == "extra":
                continue
            val = getattr(self, f.name)
            if val is not None:
                core.append(f"{f.name}={_fmt(val)}")
        extra_items = [f"{k}={_fmt(v)}" for k, v in self.extra.items()]
        return f"Batch({', '.join(core + extra_items)})"

    def replace(self: TBatch, **kwargs) -> TBatch:
        core_changes = {}
        extra_updates = {}
        valid_fields = self.__class__.__dataclass_fields__
        for k, v in kwargs.items():
            if k in valid_fields:
                core_changes[k] = v
            else:
                extra_updates[k] = v
        if extra_updates:
            base_extra = core_changes.get("extra", self.extra)
            new_extra = base_extra.copy()
            new_extra.update(extra_updates)
            core_changes["extra"] = new_extra
        return replace(self, **core_changes)


if _BACKEND == "torch":
    import torch.utils._pytree as pytree

    def _flatten_batch(batch: Batch):
        core_fields = [f.name for f in fields(batch) if f.name != "extra"]
        core_values = [getattr(batch, f) for f in core_fields]
        extra_keys = sorted(batch.extra.keys())
        extra_values = [batch.extra[k] for k in extra_keys]
        children = tuple(core_values + extra_values)
        aux_data = (tuple(core_fields), tuple(extra_keys))
        return children, aux_data

    def _unflatten_batch(children, aux_data):
        core_names, extra_keys = aux_data
        n_core = len(core_names)
        core_vals = children[:n_core]
        extra_vals = children[n_core:]
        init_kwargs = dict(zip(core_names, core_vals))
        extra_dict = dict(zip(extra_keys, extra_vals))
        return Batch(extra=extra_dict, **init_kwargs)

    pytree.register_pytree_node(Batch, _flatten_batch, _unflatten_batch)

if _BACKEND == "jax":
    import jax

    def _flatten_batch(batch: Batch):
        core_fields = [f.name for f in fields(batch) if f.name != "extra"]
        core_values = [getattr(batch, f) for f in core_fields]
        extra_keys = sorted(batch.extra.keys())
        extra_values = [batch.extra[k] for k in extra_keys]
        children = tuple(core_values + extra_values)
        aux_data = (tuple(core_fields), tuple(extra_keys))
        return children, aux_data

    def _unflatten_batch(aux_data, children):
        core_names, extra_keys = aux_data
        n_core = len(core_names)
        core_vals = children[:n_core]
        extra_vals = children[n_core:]
        init_kwargs = dict(zip(core_names, core_vals))
        extra_dict = dict(zip(extra_keys, extra_vals))
        return Batch(extra=extra_dict, **init_kwargs)

    jax.tree_util.register_pytree_node(Batch, _flatten_batch, _unflatten_batch)


__all__ = ["Batch"]
