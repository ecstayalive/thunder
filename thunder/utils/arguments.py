import argparse
import dataclasses
import enum
import re
import sys
import textwrap
from dataclasses import asdict, field, fields, is_dataclass
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
    if factory:
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

        if origin is Union:
            valid = [x for x in args if x is not type(None)]
            if len(valid) == 1:
                return TypeReflector.unwrap(valid[0])
        elif origin in (list, List):
            if args:
                return TypeReflector.unwrap(args[0])

        return t

    @staticmethod
    def get_origin_type(t: Type) -> Any:
        origin = get_origin(t)
        if origin is Union:
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

            # 2. List -> list[int]
            if origin in (list, List):
                inner = TypeReflector.to_str(args[0]) if args else "Any"
                return f"list[{inner}]"
            if isinstance(t, type) and issubclass(t, enum.Enum):
                return t.__name__
            if origin is Union:
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
        elif origin in (list, List):
            self._handle_list(kwargs, real_type, default)
            inner_type = get_args(field_type)[0] if get_args(field_type) else str
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

    def _handle_list(self, kwargs, inner_type, default):
        kwargs["type"] = inner_type
        kwargs["nargs"] = "*" if default is not dataclasses.MISSING else "+"

    def _handle_literal(self, kwargs, full_type):
        args = get_args(full_type)
        if get_origin(full_type) is Union:
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
        self.parser = argparse.ArgumentParser(
            formatter_class=TyroStyleHelpFormatter, description=target_cls.__doc__
        )
        self.unknown_args = []

        self.adapter = ArgumentAdapter(self.parser)
        self._build_recursive(target_cls, default_instance=default_instance)

    def parse(self, args: Optional[List[str]] = None) -> T:
        if args is None:
            args = sys.argv[1:]
        namespace, self.unknown_args = self.parser.parse_known_args(args)
        return self._reconstruct(self.target_cls, vars(namespace))

    @classmethod
    def transform(cls, source: Any, target_cls: Type[T], args: Optional[List[str]] = None) -> T:
        parser = cls(target_cls, default_instance=source)
        return parser.parse(args)

    @staticmethod
    def as_dict(cfg: Any) -> Dict[str, Any]:
        return asdict(cfg)

    def _build_recursive(self, cls: Type, prefix: str = "", default_instance: Any = None):
        hints = TypeReflector.resolve_hints(cls)
        docs = DocstringResolver.resolve(cls)
        for f in fields(cls):
            if f.metadata.get("external"):
                continue
            f_type = hints.get(f.name, f.type)
            real_type = TypeReflector.unwrap(f_type)
            default_val = self._get_effective_default(f, default_instance)
            if is_dataclass(real_type):
                nested_def = default_val if is_dataclass(default_val) else None
                self._build_recursive(
                    real_type, prefix=f"{prefix}{f.name}.", default_instance=nested_def
                )
            else:
                help_text = f.metadata.get("help") or docs.get(f.name, "")
                self.adapter.add_field(f, f_type, prefix, default_val, help_text)

    def _reconstruct(self, cls: Type, data: Dict[str, Any], prefix: str = "") -> Any:
        init_kwargs = {}
        hints = TypeReflector.resolve_hints(cls)

        for f in fields(cls):
            full_key = f"{prefix}{f.name}"
            f_type = hints.get(f.name, f.type)
            real_type = TypeReflector.unwrap(f_type)

            if is_dataclass(real_type):
                init_kwargs[f.name] = self._reconstruct(real_type, data, prefix=f"{full_key}.")
            elif full_key in data:
                val = data[full_key]
                if isinstance(real_type, type) and issubclass(real_type, enum.Enum):
                    val = real_type(val)
                init_kwargs[f.name] = val

        return cls(**init_kwargs)

    def _get_effective_default(self, f: dataclasses.Field, instance: Any) -> Any:
        if instance and hasattr(instance, f.name):
            return getattr(instance, f.name)

        if f.default is not dataclasses.MISSING:
            return f.default
        if f.default_factory is not dataclasses.MISSING:
            try:
                return f.default_factory()
            except:
                pass
        return dataclasses.MISSING
