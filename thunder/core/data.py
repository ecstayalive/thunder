from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from functools import wraps
from typing import Any, Dict, Optional, TypeVar

_BACKEND = os.getenv("THUNDER_BACKEND", "torch").lower()
TAttrData = TypeVar("TAttrData", bound="AttrData")
TBatch = TypeVar("TBatch", bound="Batch")
_REGISTERED_ATTRDATA_TYPES: set[type] = set()


@dataclass(slots=True, init=False, repr=False)
class AttrData:
    """Dataclass container with attribute-style access to dynamic fields.

    `AttrData` is the common base type for Thunder data containers. Explicit
    dataclass fields describe the stable structure, while `_data` stores
    dynamically attached attributes.

    Only dataclass subclasses are supported as pytree nodes. Use
    `@attr_dataclass(...)` for concrete subclasses so Torch/JAX can
    reconstruct the exact runtime type during unflatten.
    """

    _data: Dict[str, Any] = field(default_factory=dict, init=False)

    def __init__(self, _data: Optional[Dict[str, Any]] = None, **kwargs):
        base_data = {} if _data is None else dict(_data)
        base_data.update(kwargs)
        object.__setattr__(self, "_data", base_data)

    def __getattr__(self, name: str) -> Any:
        try:
            data = object.__getattribute__(self, "_data")
            return data[name]
        except (AttributeError, KeyError) as exc:
            raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'") from exc

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.__class__.__dataclass_fields__ or name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        data = object.__getattribute__(self, "_data")
        data[name] = value

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def __dir__(self):
        return list(self.__class__.__dataclass_fields__.keys()) + list(self._data.keys())

    def __repr__(self) -> str:
        def _fmt(v: Any) -> str:
            if hasattr(v, "shape"):
                return f"Arr{tuple(v.shape)}"
            if isinstance(v, dict):
                return f"Dict[{len(v)}]"
            if isinstance(v, (list, tuple)):
                return f"{type(v).__name__}[{len(v)}]"
            return str(v)

        core = []
        for name in type(self).__attrdata_field_names__:
            val = getattr(self, name)
            if val is not None:
                core.append(f"{name}={_fmt(val)}")
        dynamic_items = [f"{k}={_fmt(v)}" for k, v in self._data.items()]
        return f"{type(self).__name__}({', '.join(core + dynamic_items)})"

    def get(self, key: str, default=None) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            return default

    def update(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)

    def replace(self: TAttrData, **kwargs) -> TAttrData:
        explicit_data = kwargs.pop("_data", None)
        base_data = dict(self._data if explicit_data is None else explicit_data)

        init_kwargs = {}
        for name in type(self).__attrdata_field_names__:
            init_kwargs[name] = kwargs.pop(name, getattr(self, name))

        base_data.update(kwargs)
        return type(self)(_data=base_data, **init_kwargs)


def _flatten_attrdata_instance(data: AttrData):
    core_fields = type(data).__attrdata_field_names__
    core_values = tuple(getattr(data, name) for name in core_fields)
    data_keys = tuple(sorted(data._data.keys()))
    data_values = tuple(data._data[key] for key in data_keys)
    children = core_values + data_values
    aux_data = (type(data), core_fields, data_keys)
    return children, aux_data


def _torch_unflatten_attrdata(children, aux_data):
    cls, core_fields, data_keys = aux_data
    n_core = len(core_fields)
    core_vals = children[:n_core]
    data_vals = children[n_core:]
    init_kwargs = dict(zip(core_fields, core_vals))
    data_dict = dict(zip(data_keys, data_vals))
    return cls(_data=data_dict, **init_kwargs)


def _jax_unflatten_attrdata(aux_data, children):
    cls, core_fields, data_keys = aux_data
    n_core = len(core_fields)
    core_vals = children[:n_core]
    data_vals = children[n_core:]
    init_kwargs = dict(zip(core_fields, core_vals))
    data_dict = dict(zip(data_keys, data_vals))
    return cls(_data=data_dict, **init_kwargs)


def register_attrdata_type(cls: type[TAttrData]) -> type[TAttrData]:
    """Register a concrete AttrData dataclass as a pytree node."""

    if cls in _REGISTERED_ATTRDATA_TYPES:
        return cls
    if not is_dataclass(cls):
        raise TypeError(f"{cls.__name__} must be a dataclass before pytree registration.")
    if not issubclass(cls, AttrData):
        raise TypeError(f"{cls.__name__} must inherit from AttrData.")

    cls.__attrdata_field_names__ = tuple(f.name for f in fields(cls) if f.name != "_data")
    cls.__attrdata_field_name_set__ = frozenset(cls.__attrdata_field_names__)

    if _BACKEND == "torch":
        import torch.utils._pytree as pytree

        pytree.register_pytree_node(cls, _flatten_attrdata_instance, _torch_unflatten_attrdata)
    elif _BACKEND == "jax":
        import jax

        jax.tree_util.register_pytree_node(cls, _flatten_attrdata_instance, _jax_unflatten_attrdata)

    _REGISTERED_ATTRDATA_TYPES.add(cls)
    return cls


def attr_dataclass(_cls=None, **dataclass_kwargs):
    """Dataclass decorator that also registers the concrete AttrData subtype."""

    def wrap(cls):
        dataclass_kwargs.setdefault("init", True)
        dataclass_kwargs.setdefault("repr", False)
        cls = dataclass(cls, **dataclass_kwargs)
        original_init = cls.__init__
        init_field_names = tuple(f.name for f in fields(cls) if f.name != "_data" and f.init)
        field_name_set = frozenset(init_field_names)

        @wraps(original_init)
        def __init__(self, *args, _data: Optional[Dict[str, Any]] = None, **kwargs):
            base_data = {} if _data is None else dict(_data)
            core_kwargs = {}
            extra_kwargs = {}
            for key, value in kwargs.items():
                if key in field_name_set:
                    core_kwargs[key] = value
                else:
                    extra_kwargs[key] = value

            consumed_names = init_field_names[: len(args)]
            for name in init_field_names[len(consumed_names) :]:
                if name not in core_kwargs and name in base_data:
                    core_kwargs[name] = base_data.pop(name)

            original_init(self, *args, **core_kwargs)
            base_data.update(extra_kwargs)
            object.__setattr__(self, "_data", base_data)

        cls.__init__ = __init__
        return register_attrdata_type(cls)

    if _cls is None:
        return wrap
    return wrap(_cls)


register_attrdata_type(AttrData)
Cache = AttrData


@attr_dataclass(slots=True)
class Batch(AttrData):
    obs: Optional[Dict[str, Any]] = None
    actions: Optional[Any] = None
    rewards: Optional[Any] = None
    dones: Optional[Any] = None
    timeouts: Optional[Any] = None
    mask: Optional[Any] = None
    next_obs: Optional[Dict[str, Any]] = None


__all__ = ["AttrData", "Cache", "Batch", "attr_dataclass", "register_attrdata_type"]
