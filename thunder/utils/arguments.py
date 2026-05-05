import argparse
import dataclasses
import enum
import re
import sys
import textwrap
from dataclasses import asdict, field, fields, is_dataclass
from types import UnionType
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

from rich.table import Table
from rich.text import Text
from rich_argparse import RichHelpFormatter

T = TypeVar("T")
_TYPE_CACHE = {}
_DOC_CACHE = {}


class TyroStyleHelpFormatter(RichHelpFormatter):
    """A tyro style help formatter"""

    def __init__(self, *args, **kwargs):
        kwargs["max_help_position"] = 45
        super().__init__(*args, **kwargs)
        self.styles["argparse.args"] = "bright_cyan"
        self.styles["argparse.groups"] = "bold magenta"
        self.styles["argparse.help"] = "default"
        self.styles["argparse.metavar"] = "bright_yellow"
        self.styles["argparse.prog"] = "bold"
        self.styles["argparse.syntax"] = "bold"
        self.styles["argparse.default"] = "dim"
        self.styles["argparse.required"] = "bold red"

    @staticmethod
    def group_name_formatter(name: str) -> str:
        return name.upper()

    def start_section(self, heading: str) -> None:
        title_text = Text(
            f" {self.group_name_formatter(heading)} ", style=self.styles["argparse.groups"]
        )
        width = self.console.width
        line_len = max(0, width - 2 - title_text.cell_len - 1)
        line = "─" * line_len
        header = Text.assemble(
            ("╭─", self.styles["argparse.groups"]),
            title_text,
            (line, self.styles["argparse.groups"]),
            ("╮", self.styles["argparse.groups"]),
        )
        self.console.print(header)

    def end_section(self) -> None:
        width = self.console.width
        line_len = max(0, width - 2)
        line = "─" * line_len
        self.console.print(f"╰{line}╯", style=self.styles["argparse.groups"])

    def add_arguments(self, actions: list[argparse.Action]) -> None:
        if not actions:
            self.console.print(
                f"│{' ' * (self.console.width - 2)}│", style=self.styles["argparse.groups"]
            )
            return

        max_invocation_width = 0
        for action in actions:
            invocation = self._format_action_invocation(action)
            max_invocation_width = max(max_invocation_width, len(invocation))

        wrapper_table = Table(box=None, show_header=False, padding=0, expand=True)
        wrapper_table.add_column(style=self.styles["argparse.groups"], width=1, no_wrap=True)
        wrapper_table.add_column(width=1)
        wrapper_table.add_column(width=max_invocation_width)
        wrapper_table.add_column(width=2)
        wrapper_table.add_column(ratio=1)
        wrapper_table.add_column(style=self.styles["argparse.groups"], width=1, no_wrap=True)

        for action in actions:
            invocation_str = self._format_action_invocation(action)
            metavar = (
                action.metavar if action.metavar else self._get_default_metavar_for_optional(action)
            )
            if metavar:
                pattern = re.escape(metavar)
                parts = re.split(f"({pattern})", invocation_str)
                invocation_text = Text()
                for part in parts:
                    if part == metavar:
                        invocation_text.append(part, style=self.styles["argparse.metavar"])
                    else:
                        invocation_text.append(part, style=self.styles["argparse.args"])
            else:
                invocation_text = Text(invocation_str, style=self.styles["argparse.args"])
            help_text = Text.from_markup(action.help or "", style=self.styles["argparse.help"])
            wrapper_table.add_row("│", " ", invocation_text, " ", help_text, "│")

        self.console.print(wrapper_table)


def ArgOpt(
    default=dataclasses.MISSING, *, help="", short=None, factory=None, external=False, **kwargs
):
    metadata = {"help": help, "short": short, "external": external, "argparse_kwargs": kwargs}
    if factory is not None:
        return field(default_factory=factory, metadata=metadata)
    if default is not dataclasses.MISSING:
        return field(default=default, metadata=metadata)
    return field(metadata=metadata)


class DocstringResolver:
    @staticmethod
    def resolve(cls: Type) -> Dict[str, str]:
        if cls in _DOC_CACHE:
            return _DOC_CACHE[cls]

        doc_map = {}
        for base in reversed(cls.__mro__):
            if base is object or not base.__doc__:
                continue
            DocstringResolver._parse_class_content(base.__doc__, doc_map)

        _DOC_CACHE[cls] = doc_map
        return doc_map

    @staticmethod
    def _parse_class_content(docstring: str, doc_map: Dict[str, str]):
        content = textwrap.dedent(docstring)
        current_attr, in_section = None, False
        for line in content.splitlines():
            line = line.strip()
            if line.lower() in ("args:", "parameters:", "attributes:"):
                in_section = True
                continue
            if not in_section or not line:
                continue
            match = re.match(r"^(\w+)(?:\s*\(.*\))?\s*:\s*(.*)", line)
            if match:
                current_attr, text = match.groups()
                doc_map[current_attr] = text
            elif current_attr:
                doc_map[current_attr] += " " + line


class TypeReflector:
    @staticmethod
    def is_union(t: Type) -> bool:
        return get_origin(t) in (Union, UnionType)

    @staticmethod
    def optional_inner(t: Type) -> Type:
        if not TypeReflector.is_union(t):
            return t
        valid = [x for x in get_args(t) if x is not type(None)]
        return valid[0] if len(valid) == 1 else t

    @staticmethod
    def sequence_inner(t: Type) -> Type:
        t = TypeReflector.optional_inner(t)
        args = get_args(t)
        return args[0] if args else str

    @staticmethod
    def resolve_hints(cls: Type) -> Dict[str, Any]:
        """ """
        if cls not in _TYPE_CACHE:
            try:
                from typing import get_type_hints

                _TYPE_CACHE[cls] = get_type_hints(cls)
            except Exception:
                _TYPE_CACHE[cls] = getattr(cls, "__annotations__", {})
        return _TYPE_CACHE[cls]

    @staticmethod
    def unwrap(t: Type) -> Type:
        origin = get_origin(t)
        args = get_args(t)

        if TypeReflector.is_union(t):
            valid = [x for x in args if x is not type(None)]
            if len(valid) == 1:
                return TypeReflector.unwrap(valid[0])
        elif origin in (list, List, tuple, Tuple):
            if args:
                return TypeReflector.unwrap(args[0])

        return t

    @staticmethod
    def get_origin_type(t: Type) -> Any:
        origin = get_origin(t)
        if TypeReflector.is_union(t):
            args = get_args(t)
            valid = [x for x in args if x is not type(None)]
            if len(valid) == 1:
                return get_origin(valid[0]) or valid[0]
        return origin

    @staticmethod
    def to_str(t: Type) -> str:
        try:
            origin = get_origin(t)
            args = get_args(t)

            # 1. Literal -> {a, b, c}
            if origin is Literal:
                return f"{{{','.join(map(str, args))}}}"

            # 2. Sequence -> list[int] / tuple[int]
            if origin in (list, List, tuple, Tuple):
                inner = TypeReflector.to_str(args[0]) if args else "Any"
                return f"{origin.__name__}[{inner}]"
            if isinstance(t, type) and issubclass(t, enum.Enum):
                return t.__name__
            if TypeReflector.is_union(t):
                valid = [x for x in args if x is not type(None)]
                if len(valid) == 1:
                    return TypeReflector.to_str(valid[0])
                return "|".join(TypeReflector.to_str(x) for x in valid)
            if hasattr(t, "__name__"):
                return t.__name__

            return str(t).replace("typing.", "")
        except:
            return str(t)


class ArgumentAdapter:
    def __init__(self, parser: argparse.ArgumentParser):
        self.groups = {
            True: parser.add_argument_group("Required arguments"),
            False: parser.add_argument_group("Optional arguments"),
        }

    def add_field(
        self, field: dataclasses.Field, field_type: Type, prefix: str, default: Any, help_text: str
    ):
        name = f"{prefix}{field.name}"
        flag = f"--{name.replace('_', '-')}"

        real_type = TypeReflector.unwrap(field_type)
        origin = TypeReflector.get_origin_type(field_type)
        type_str = TypeReflector.to_str(field_type)

        kwargs = {
            "dest": name,
            "default": default if default is not dataclasses.MISSING else None,
        }

        if real_type is bool:
            self._handle_bool(kwargs, default)
            kwargs.pop("metavar", None)
        elif origin in (list, List, tuple, Tuple):
            self._handle_sequence(kwargs, real_type, default)
            inner_type = TypeReflector.sequence_inner(field_type)
            kwargs["metavar"] = TypeReflector.to_str(inner_type).upper()
        elif origin is Literal:
            self._handle_literal(kwargs, field_type)
            # Literal: {a,b}
            # kwargs["metavar"] = type_str
            kwargs["metavar"] = f"{{{','.join(map(str, kwargs['choices']))}}}"
        elif isinstance(real_type, type) and issubclass(real_type, enum.Enum):
            self._handle_enum(kwargs, real_type)
            # Enum : {SGD,ADAM}
            kwargs["metavar"] = f"{{{','.join(map(str, kwargs['choices']))}}}"
        else:
            self._handle_primitive(kwargs, real_type, default)
            # INT, STR
            if "metavar" not in kwargs:
                kwargs["metavar"] = TypeReflector.to_str(real_type).upper()
        kwargs.update(field.metadata.get("argparse_kwargs", {}))
        if real_type is bool:
            kwargs.pop("metavar", None)
        is_required = kwargs.get("required", False)
        kwargs["help"] = self._format_help(help_text, type_str, default, is_required)

        flags = [flag]
        if field.metadata.get("short"):
            flags.append(field.metadata.get("short"))

        self.groups[is_required].add_argument(*flags, **kwargs)

    def _handle_bool(self, kwargs, default):
        kwargs["action"] = "store_false" if default is True else "store_true"
        kwargs.pop("default", None)

    def _handle_sequence(self, kwargs, inner_type, default):
        kwargs["type"] = inner_type
        kwargs["nargs"] = "*" if default is not dataclasses.MISSING else "+"

    def _handle_literal(self, kwargs, full_type):
        args = get_args(full_type)
        if TypeReflector.is_union(full_type):
            args = get_args(next(t for t in get_args(full_type) if t is not type(None)))
        kwargs["choices"] = args
        kwargs["type"] = type(args[0])

    def _handle_enum(self, kwargs, enum_type):
        choices = [e.value for e in enum_type]
        kwargs["choices"] = choices
        kwargs["type"] = type(choices[0])

    def _handle_primitive(self, kwargs, real_type, default):
        kwargs["type"] = real_type
        if default is dataclasses.MISSING:
            kwargs["required"] = True
            kwargs.pop("default", None)

    def _format_help(self, base_help: str, type_str: str, default: Any, required: bool) -> str:
        parts = []
        # if type_str:
        #     parts.append(type_str.upper())
        if required:
            parts.append("required")
        elif default is not dataclasses.MISSING:
            val = default.name if isinstance(default, enum.Enum) else str(default)
            parts.append(f"default: {val}")

        meta = f"[dim]({', '.join(parts)})[/dim]" if parts else ""
        return f"{base_help}  {meta}" if base_help else meta


class ArgParser:
    def __init__(self, target_cls: Type[T], default_instance: Any = None):
        self.target_cls = target_cls
        self.default_instance = default_instance
        self.parser = argparse.ArgumentParser(
            formatter_class=TyroStyleHelpFormatter, description=target_cls.__doc__
        )
        self.unknown_args = []

        self.adapter = ArgumentAdapter(self.parser)
        self._build_recursive(
            target_cls,
            default_instance=default_instance,
            skip_source_fields=default_instance is not None,
        )

    def parse(self, args: Optional[List[str]] = None) -> T:
        if args is None:
            args = sys.argv[1:]
        namespace, self.unknown_args = self.parser.parse_known_args(args)
        return self._reconstruct(
            self.target_cls, vars(namespace), fallback_instance=self.default_instance
        )

    @classmethod
    def transform(cls, source: Any, target_cls: Type[T], args: Optional[List[str]] = None) -> T:
        if isinstance(source, target_cls):
            return source
        parser = cls(target_cls, default_instance=source)
        return parser.parse(args)

    @staticmethod
    def as_dict(cfg: Any) -> Dict[str, Any]:
        return asdict(cfg)

    def _build_recursive(
        self,
        cls: Type,
        prefix: str = "",
        default_instance: Any = None,
        skip_source_fields: bool = False,
    ):
        hints = TypeReflector.resolve_hints(cls)
        docs = DocstringResolver.resolve(cls)
        for f in fields(cls):
            if f.metadata.get("external"):
                continue
            f_type = hints.get(f.name, f.type)
            real_type = TypeReflector.unwrap(f_type)
            if is_dataclass(real_type):
                nested_def = self._get_default_value(f, default_instance)
                nested_type = self._resolve_nested_dataclass_type(real_type, nested_def)
                self._build_recursive(
                    nested_type,
                    prefix=f"{prefix}{f.name}.",
                    default_instance=nested_def,
                    skip_source_fields=skip_source_fields
                    and self._source_has_field(default_instance, f.name),
                )
            else:
                if skip_source_fields and self._source_has_field(default_instance, f.name):
                    continue
                default_val = self._get_effective_default(f, default_instance)
                help_text = f.metadata.get("help") or docs.get(f.name, "")
                self.adapter.add_field(f, f_type, prefix, default_val, help_text)

    def _reconstruct(
        self,
        cls: Type,
        data: Dict[str, Any],
        prefix: str = "",
        fallback_instance: Any = None,
    ) -> Any:
        init_kwargs = {}
        hints = TypeReflector.resolve_hints(cls)

        for f in fields(cls):
            full_key = f"{prefix}{f.name}"
            f_type = hints.get(f.name, f.type)
            real_type = TypeReflector.unwrap(f_type)

            if is_dataclass(real_type):
                nested_fallback = self._get_default_value(f, fallback_instance)
                nested_type = self._resolve_nested_dataclass_type(
                    real_type, nested_fallback, data, f"{full_key}."
                )
                init_kwargs[f.name] = self._reconstruct(
                    nested_type,
                    data,
                    prefix=f"{full_key}.",
                    fallback_instance=nested_fallback,
                )
            elif full_key in data:
                init_kwargs[f.name] = self._coerce_value(data[full_key], f_type, real_type)
            elif self._source_has_field(fallback_instance, f.name):
                init_kwargs[f.name] = self._get_source_value(fallback_instance, f.name)

        return cls(**init_kwargs)

    def _source_has_field(self, instance: Any, name: str) -> bool:
        if instance is None:
            return False
        if isinstance(instance, dict):
            return name in instance
        return hasattr(instance, name)

    def _get_source_value(self, instance: Any, name: str) -> Any:
        if isinstance(instance, dict):
            return instance[name]
        return getattr(instance, name)

    def _get_default_value(self, f: dataclasses.Field, instance: Any) -> Any:
        value = self._get_effective_default(f, instance)
        return None if value is dataclasses.MISSING else value

    def _coerce_value(self, value: Any, field_type: Type, real_type: Type) -> Any:
        if isinstance(real_type, type) and issubclass(real_type, enum.Enum):
            return real_type(value)

        origin = TypeReflector.get_origin_type(field_type)
        if origin in (tuple, Tuple) and isinstance(value, list):
            return tuple(value)

        return value

    def _resolve_nested_dataclass_type(
        self,
        annotated_type: Type,
        default_instance: Any,
        data: Dict[str, Any] | None = None,
        prefix: str = "",
    ) -> Type:
        if is_dataclass(default_instance) and isinstance(default_instance, annotated_type):
            if type(default_instance) is not annotated_type:
                return type(default_instance)

        framework = self._get_optional_source_value(default_instance, "framework")
        if framework is None and data is not None:
            framework = data.get(f"{prefix}framework")
        if framework is None or getattr(annotated_type, "__name__", None) != "EnvLoaderSpec":
            return annotated_type

        try:
            from thunder.env.env import get_loader_spec_cls

            spec_cls = get_loader_spec_cls(framework, annotated_type)
        except Exception:
            spec_cls = annotated_type
        return spec_cls or annotated_type

    def _get_optional_source_value(self, instance: Any, name: str) -> Any:
        if not self._source_has_field(instance, name):
            return None
        return self._get_source_value(instance, name)

    def _get_effective_default(self, f: dataclasses.Field, instance: Any) -> Any:
        if self._source_has_field(instance, f.name):
            return self._get_source_value(instance, f.name)

        if f.default is not dataclasses.MISSING:
            return f.default
        if f.default_factory is not dataclasses.MISSING:
            return f.default_factory()
        return dataclasses.MISSING
