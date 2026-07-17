from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, MISSING, replace
from typing import Any, Dict, Optional, TypeVar

_BACKEND = os.getenv("THUNDER_BACKEND", "torch").lower()

TAttrData = TypeVar("TAttrData", bound="AttrData")
_REGISTERED_ATTR_DATA_TYPES: set[type[Any]] = set()


def _flatten_attr_data(
    obj: AttrData,
) -> tuple[tuple[Any, ...], tuple[tuple[str, ...], tuple[str, ...]]]:
    core_names = tuple(name for name, _, _, _ in type(obj).__attr_data_core_fields__)
    data_keys = tuple(sorted(obj._data))
    children = tuple(getattr(obj, name) for name in core_names) + tuple(
        obj._data[key] for key in data_keys
    )
    return children, (core_names, data_keys)


if _BACKEND == "torch":
    import torch.utils._cxx_pytree as pytree

    _register_pytree_node = pytree.register_pytree_node

elif _BACKEND == "jax":
    import jax.tree_util as jtu

    _register_pytree_node = jtu.register_pytree_node

else:
    _register_pytree_node = None


def _register_attr_data_type(cls: type["AttrData"]) -> None:
    if cls in _REGISTERED_ATTR_DATA_TYPES:
        return

    if _register_pytree_node is not None:

        if _BACKEND == "torch":

            def _unflatten(children, aux_data):
                core_names, data_keys = aux_data
                n_core = len(core_names)
                kwargs = dict(zip(core_names, children[:n_core]))
                kwargs["_data"] = dict(zip(data_keys, children[n_core:]))
                return cls(**kwargs)

        else:

            def _unflatten(aux_data, children):
                core_names, data_keys = aux_data
                n_core = len(core_names)
                kwargs = dict(zip(core_names, children[:n_core]))
                kwargs["_data"] = dict(zip(data_keys, children[n_core:]))
                return cls(**kwargs)

        _register_pytree_node(cls, _flatten_attr_data, _unflatten)

    _REGISTERED_ATTR_DATA_TYPES.add(cls)


def attr_dataclass(cls: type[TAttrData] | None = None, **dataclass_kwargs):
    dataclass_kwargs.setdefault("slots", True)
    dataclass_kwargs.setdefault("repr", False)
    dataclass_kwargs["init"] = False

    def wrap(cls: type[TAttrData]) -> type[TAttrData]:
        if not issubclass(cls, AttrData):
            raise TypeError("attr_dataclass can only be used with AttrData subclasses.")
        if "__dataclass_fields__" not in cls.__dict__:
            cls = dataclass(**dataclass_kwargs)(cls)
        core_fields = tuple(
            (f.name, f.default, f.default_factory, f.kw_only)
            for f in fields(cls)
            if f.name != "_data"
        )
        cls.__attr_data_core_fields__ = core_fields
        cls.__attr_data_field_set__ = frozenset(cls.__dataclass_fields__)

        no_default, has_default, has_factory = 0, 1, 2
        init_plan = []
        for name, default, default_factory, kw_only in core_fields:
            if default is not MISSING:
                init_plan.append((name, has_default, default, kw_only))
            elif default_factory is not MISSING:
                init_plan.append((name, has_factory, default_factory, kw_only))
            else:
                init_plan.append((name, no_default, None, kw_only))
        init_plan = tuple(init_plan)
        positional_plan = tuple(field for field in init_plan if not field[3])
        cls_name = cls.__name__
        post_init = getattr(cls, "__post_init__", None)

        def __init__(self, *args, _data=None, **kwargs):
            if type(self) is not cls:
                raise TypeError(
                    f"{type(self).__name__} must be decorated with @attr_dataclass."
                )

            data = {} if _data is None else dict(_data)
            if len(args) > len(positional_plan):
                raise TypeError(
                    f"{cls_name} expected at most {len(positional_plan)} "
                    f"positional arguments, got {len(args)}"
                )
            for (name, _, _, _), value in zip(positional_plan, args):
                if name in data or name in kwargs:
                    raise TypeError(f"Got multiple values for argument: '{name}'")
                data[name] = value
            data.update(kwargs)

            for name, default_kind, default_value, _ in init_plan:
                if name in data:
                    object.__setattr__(self, name, data.pop(name))
                elif default_kind == has_default:
                    object.__setattr__(self, name, default_value)
                elif default_kind == has_factory:
                    object.__setattr__(self, name, default_value())
                else:
                    raise TypeError(f"Missing required argument: '{name}'")

            object.__setattr__(self, "_data", data)
            if post_init is not None:
                post_init(self)

        cls.__init__ = __init__
        _register_attr_data_type(cls)
        return cls

    return wrap if cls is None else wrap(cls)


@dataclass(slots=True, init=False, repr=False)
class AttrData:
    """Base class for attribute-based data structures.
    Supports both statically declared fields and dynamic
    fields stored in a dictionary. Provides dict-like
    access to dynamic fields anda custom __repr__ implementation."""

    _data: Dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, name: str) -> Any:
        if name == "_data":
            raise AttributeError(
                f"'{self.__class__.__name__}' has no attribute '{name}'"
            )
        try:
            return self._data[name]
        except KeyError as exc:
            raise AttributeError(
                f"'{self.__class__.__name__}' has no attribute '{name}'"
            ) from exc
        except AttributeError as exc:
            raise AttributeError(
                f"'{self.__class__.__name__}' has no attribute '{name}'"
            ) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.__class__.__attr_data_field_set__:
            object.__setattr__(self, name, value)
        else:
            self._data[name] = value

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def get(self, key: str, default=None) -> Any:
        return self._data.get(key, default)

    def update(self, **kwargs) -> None:
        data = kwargs.pop("_data", None)
        if data is not None:
            object.__setattr__(self, "_data", data.copy())
        for key, value in kwargs.items():
            if key in self.__class__.__attr_data_field_set__:
                object.__setattr__(self, key, value)
            else:
                self._data[key] = value

    def replace(self: TAttrData, **kwargs) -> TAttrData:
        declared = {}
        dynamic = {}
        for key, value in kwargs.items():
            if key in self.__class__.__attr_data_field_set__:
                declared[key] = value
            else:
                dynamic[key] = value
        if dynamic:
            base_data = declared.get("_data", self._data)
            new_data = base_data.copy()
            new_data.update(dynamic)
            declared["_data"] = new_data
        return replace(self, **declared)

    def __dir__(self):
        return list(
            dict.fromkeys([*self.__class__.__dataclass_fields__, *self._data.keys()])
        )

    def __repr__(self) -> str:
        def _fmt(value: Any) -> str:
            if hasattr(value, "shape"):
                return f"Arr{tuple(value.shape)}"
            if isinstance(value, dict):
                return f"Dict[{len(value)}]"
            if isinstance(value, (list, tuple)):
                return f"{type(value).__name__}[{len(value)}]"
            return str(value)

        items = []
        for name, _, _, _ in self.__class__.__attr_data_core_fields__:
            value = getattr(self, name)
            if value is not None:
                items.append(f"{name}={_fmt(value)}")
        items.extend(f"{key}={_fmt(value)}" for key, value in self._data.items())
        return f"{self.__class__.__name__}({', '.join(items)})"


attr_dataclass(AttrData)


@attr_dataclass(slots=True)
class Batch(AttrData):
    """A basic data structure for Markov Chain ,
    with some common fields for convenience.
    """

    obs: Optional[Dict[str, Any]] = None
    actions: Optional[Any] = None
    rewards: Optional[Any] = None
    next_obs: Optional[Dict[str, Any]] = None
    terminated: Optional[Any] = None
    timeouts: Optional[Any] = None
    # Optional
    mask: Optional[Any] = None
    values: Optional[Any] = None
    next_values: Optional[Any] = None
    advantages: Optional[Any] = None
    returns: Optional[Any] = None


__all__ = ["AttrData", "Batch", "attr_dataclass"]
