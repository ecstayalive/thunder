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
    Generic,
    get_args,
    get_origin,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich_argparse import RichHelpFormatter

T = TypeVar("T")
_TYPE_CACHE = {}
_DOC_CACHE = {}


class BooleanAction(argparse.Action):

    def __init__(
        self,
        option_strings: List[str],
        dest: str,
        default: Any = None,
        required: bool = False,
        help: Optional[str] = None,
        metavar: Optional[str] = None,
        **kwargs,
    ):
        resolved_option_strings = []
        for opt in option_strings:
            resolved_option_strings.append(opt)
            if opt.startswith("--") and not opt.startswith("--no-"):
                neg_opt = f"--no-{opt[2:]}"
                if neg_opt not in resolved_option_strings:
                    resolved_option_strings.append(neg_opt)

        super().__init__(
            option_strings=resolved_option_strings,
            dest=dest,
            nargs=0,
            default=default,
            required=required,
            help=help,
            metavar=metavar,
        )

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: Optional[str] = None,
    ):
        if option_string is not None:
            value = not option_string.startswith("--no-")
            setattr(namespace, self.dest, value)


class TupleAction(argparse.Action):
    """An argparse Action that handles fixed-size or typed tuples with heterogenous elements."""

    def __init__(
        self,
        option_strings: List[str],
        dest: str,
        nargs: Any = None,
        default: Any = None,
        required: bool = False,
        help: Optional[str] = None,
        metavar: Optional[str] = None,
        types: Optional[Tuple[Type, ...]] = None,
        **kwargs,
    ):
        self.types = types or []
        super().__init__(
            option_strings=option_strings,
            dest=dest,
            nargs=nargs,
            default=default,
            required=required,
            help=help,
            metavar=metavar,
        )

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: Optional[str] = None,
    ):
        if values is None:
            setattr(namespace, self.dest, None)
            return

        if not isinstance(values, list):
            values = [values]

        coerced = []
        for i, val in enumerate(values):
            if i < len(self.types):
                t = self.types[i]
            else:
                t = self.types[-1] if self.types else str

            if t is Ellipsis:
                t = self.types[0] if self.types else str

            unwrap_t = TypeReflector.unwrap(t)

            try:
                if isinstance(unwrap_t, type) and issubclass(unwrap_t, enum.Enum):
                    coerced.append(unwrap_t(val))
                elif unwrap_t is bool:
                    coerced.append(val.lower() in ("true", "1", "yes", "on"))
                else:
                    coerced.append(unwrap_t(val) if isinstance(unwrap_t, type) else val)
            except Exception:
                raise argparse.ArgumentTypeError(
                    f"argument {self.dest}: invalid {TypeReflector.to_str(t)} value: {val!r}"
                )

        setattr(namespace, self.dest, tuple(coerced))


class ListAction(argparse.Action):
    """An argparse Action that parses typed lists, including comma-separated literals."""

    def __init__(
        self,
        option_strings: List[str],
        dest: str,
        nargs: Any = None,
        default: Any = None,
        required: bool = False,
        help: Optional[str] = None,
        metavar: Optional[str] = None,
        inner_node: Any = None,
        **kwargs,
    ):
        self.inner_node = inner_node
        super().__init__(
            option_strings=option_strings,
            dest=dest,
            nargs=nargs,
            default=default,
            required=required,
            help=help,
            metavar=metavar,
        )

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: Optional[str] = None,
    ):
        if values is None:
            setattr(namespace, self.dest, None)
            return

        if not isinstance(values, list):
            values = [values]

        parsed = []
        for value in values:
            items = [value]
            if isinstance(self.inner_node, LiteralNode) and isinstance(value, str):
                items = [item.strip() for item in value.split(",") if item.strip()]
            parsed.extend(
                ScalarValueParser.parse(self.inner_node, item) for item in items
            )

        setattr(namespace, self.dest, parsed)


class TyroStyleHelpFormatter(RichHelpFormatter):
    """A tyro style help formatter"""

    _SECTION_PADDING = 1
    _MIN_HELP_WIDTH = 12

    def __init__(self, *args, **kwargs):
        kwargs["max_help_position"] = 45
        super().__init__(*args, **kwargs)
        self._section_heading: Optional[str] = None
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
        self._section_heading = self.group_name_formatter(heading)

    def end_section(self) -> None:
        self._section_heading = None

    def _print_section(self, renderable: Any) -> None:
        title = Text(
            self._section_heading or "",
            style=self.styles["argparse.groups"],
        )
        self.console.print(
            Panel(
                renderable,
                title=title,
                title_align="left",
                border_style=self.styles["argparse.groups"],
                box=box.ROUNDED,
                padding=(0, self._SECTION_PADDING),
                expand=True,
            )
        )

    def _content_width(self) -> int:
        return max(1, self.console.width - 2 - self._SECTION_PADDING * 2)

    def _format_action_invocation(self, action: argparse.Action) -> str:
        if isinstance(action, BooleanAction):
            short_opts = [
                opt for opt in action.option_strings if not opt.startswith("--")
            ]
            long_opts = [
                opt
                for opt in action.option_strings
                if opt.startswith("--") and not opt.startswith("--no-")
            ]
            neg_opts = [opt for opt in action.option_strings if opt.startswith("--no-")]
            parts = []
            if short_opts:
                parts.append(", ".join(short_opts))
            if long_opts and neg_opts:
                parts.append(f"{long_opts[0]}/{neg_opts[0]}")
            else:
                parts.extend(long_opts + neg_opts)
            return ", ".join(parts)

        return super()._format_action_invocation(action)

    def add_arguments(self, actions: list[argparse.Action]) -> None:
        if not actions:
            self._print_section(Text(""))
            return

        max_invocation_width = 0
        for action in actions:
            invocation = self._format_action_invocation(action)
            max_invocation_width = max(max_invocation_width, len(invocation))

        content_width = self._content_width()
        gap_width = 2 if content_width >= 4 else 0
        min_help_width = min(
            self._MIN_HELP_WIDTH, max(1, content_width - gap_width - 1)
        )
        invocation_width = min(
            max_invocation_width,
            max(1, content_width - gap_width - min_help_width),
        )
        help_width = max(1, content_width - invocation_width - gap_width)

        wrapper_table = Table(box=None, show_header=False, padding=0, expand=True)
        wrapper_table.add_column(
            width=invocation_width,
            overflow="fold",
            vertical="top",
        )
        if gap_width:
            wrapper_table.add_column(width=gap_width)
        wrapper_table.add_column(
            ratio=1,
            min_width=help_width,
            overflow="fold",
            vertical="top",
        )

        for action in actions:
            invocation_str = self._format_action_invocation(action)
            metavar = (
                action.metavar
                if action.metavar
                else self._get_default_metavar_for_optional(action)
            )
            if metavar:
                pattern = re.escape(metavar)
                parts = re.split(f"({pattern})", invocation_str)
                invocation_text = Text()
                for part in parts:
                    if part == metavar:
                        invocation_text.append(
                            part, style=self.styles["argparse.metavar"]
                        )
                    else:
                        invocation_text.append(part, style=self.styles["argparse.args"])
            else:
                invocation_text = Text(
                    invocation_str, style=self.styles["argparse.args"]
                )
            help_text = Text.from_markup(
                action.help or "", style=self.styles["argparse.help"]
            )
            row = [invocation_text]
            if gap_width:
                row.append(" ")
            row.append(help_text)
            wrapper_table.add_row(*row)

        self._print_section(wrapper_table)


def ArgOpt(
    default=dataclasses.MISSING,
    *,
    help="",
    short=None,
    factory=None,
    external=False,
    serialize=True,
    **kwargs,
):
    metadata = {
        "help": help,
        "short": short,
        "external": external,
        "serialize": serialize,
        "argparse_kwargs": kwargs,
    }
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


class TypeNode:
    visitor_name = ""

    def accept(self, visitor: Any, *args, **kwargs):
        return getattr(visitor, f"visit_{self.visitor_name}")(self, *args, **kwargs)


@dataclasses.dataclass(frozen=True)
class PrimitiveNode(TypeNode):
    visitor_name = "primitive"
    py_type: Any
    annotation: Any


@dataclasses.dataclass(frozen=True)
class BoolNode(TypeNode):
    visitor_name = "bool"
    annotation: Any = bool


@dataclasses.dataclass(frozen=True)
class LiteralNode(TypeNode):
    visitor_name = "literal"
    choices: Tuple[Any, ...]
    annotation: Any


@dataclasses.dataclass(frozen=True)
class EnumNode(TypeNode):
    visitor_name = "enum"
    enum_type: Type[enum.Enum]
    annotation: Any


@dataclasses.dataclass(frozen=True)
class ListNode(TypeNode):
    visitor_name = "list"
    inner: TypeNode
    annotation: Any


@dataclasses.dataclass(frozen=True)
class FixedTupleNode(TypeNode):
    visitor_name = "fixed_tuple"
    items: Tuple[TypeNode, ...]
    annotation: Any


@dataclasses.dataclass(frozen=True)
class VariadicTupleNode(TypeNode):
    visitor_name = "variadic_tuple"
    inner: TypeNode
    annotation: Any


@dataclasses.dataclass(frozen=True)
class DataclassRefNode(TypeNode):
    visitor_name = "dataclass_ref"
    cls: Type
    annotation: Any


@dataclasses.dataclass(frozen=True)
class OptionalNode(TypeNode):
    visitor_name = "optional"
    inner: TypeNode
    annotation: Any


@dataclasses.dataclass(frozen=True)
class FieldNode:
    name: str
    path: Tuple[str, ...]
    type_node: TypeNode
    default: Any
    bind: bool
    help: str
    metadata: Dict[str, Any]
    field: dataclasses.Field

    @property
    def dest(self) -> str:
        return ".".join(self.path)

    @property
    def flag(self) -> str:
        return f"--{self.dest.replace('_', '-')}"


@dataclasses.dataclass(frozen=True)
class DataclassNode(TypeNode):
    visitor_name = "dataclass"
    cls: Type
    fields: Tuple[FieldNode, ...]
    annotation: Any


@dataclasses.dataclass(frozen=True)
class ArgSpec:
    flags: List[str]
    kwargs: Dict[str, Any]


class TypeTreeBuilder:
    @staticmethod
    def build(annotation: Any) -> TypeNode:
        inner, optional = TypeTreeBuilder._split_optional(annotation)
        node = TypeTreeBuilder._build_required(inner)
        return OptionalNode(node, annotation) if optional else node

    @staticmethod
    def _split_optional(annotation: Any) -> Tuple[Any, bool]:
        origin = get_origin(annotation)
        if origin not in (Union, UnionType):
            return annotation, False

        args = get_args(annotation)
        values = [arg for arg in args if arg is not type(None)]
        if len(values) == 1 and len(values) != len(args):
            return values[0], True

        raise TypeError(f"Unsupported union annotation: {annotation!r}")

    @staticmethod
    def _build_required(annotation: Any) -> TypeNode:
        origin = get_origin(annotation)
        args = get_args(annotation)

        if origin in (list, List):
            inner = TypeTreeBuilder.build(args[0]) if args else PrimitiveNode(str, str)
            return ListNode(inner, annotation)

        if origin in (tuple, Tuple):
            if not args:
                return VariadicTupleNode(PrimitiveNode(str, str), annotation)
            if len(args) == 2 and args[1] is Ellipsis:
                return VariadicTupleNode(TypeTreeBuilder.build(args[0]), annotation)
            return FixedTupleNode(
                tuple(TypeTreeBuilder.build(arg) for arg in args), annotation
            )

        if origin is Literal:
            return LiteralNode(tuple(args), annotation)

        if annotation is bool:
            return BoolNode(annotation)

        if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
            return EnumNode(annotation, annotation)

        if is_dataclass(annotation):
            return DataclassRefNode(annotation, annotation)

        if annotation is Any:
            return PrimitiveNode(str, annotation)

        return PrimitiveNode(annotation, annotation)


class TypeReflector:
    """Compatibility helpers for existing callers and custom argparse actions."""

    @staticmethod
    def is_union(t: Type) -> bool:
        return get_origin(t) in (Union, UnionType)

    @staticmethod
    def optional_inner(t: Type) -> Type:
        try:
            return TypeTreeBuilder._split_optional(t)[0]
        except TypeError:
            return t

    @staticmethod
    def sequence_inner(t: Type) -> Type:
        args = get_args(TypeReflector.optional_inner(t))
        return args[0] if args else str

    @staticmethod
    def is_fixed_tuple(t: Type) -> bool:
        node = TypeTreeBuilder.build(t)
        if isinstance(node, OptionalNode):
            node = node.inner
        return isinstance(node, FixedTupleNode)

    @staticmethod
    def resolve_hints(cls: Type) -> Dict[str, Any]:
        if cls not in _TYPE_CACHE:
            try:
                from typing import get_type_hints

                _TYPE_CACHE[cls] = get_type_hints(cls)
            except Exception:
                _TYPE_CACHE[cls] = getattr(cls, "__annotations__", {})
        return _TYPE_CACHE[cls]

    @staticmethod
    def unwrap(t: Type) -> Type:
        node = TypeTreeBuilder.build(t)
        while isinstance(node, OptionalNode):
            node = node.inner
        while isinstance(node, (ListNode, VariadicTupleNode)):
            node = node.inner
        if isinstance(node, FixedTupleNode) and node.items:
            node = node.items[0]
        if isinstance(node, BoolNode):
            return bool
        if isinstance(node, PrimitiveNode):
            return node.py_type
        if isinstance(node, EnumNode):
            return node.enum_type
        if isinstance(node, DataclassRefNode):
            return node.cls
        return t

    @staticmethod
    def get_origin_type(t: Type) -> Any:
        inner = TypeReflector.optional_inner(t)
        return get_origin(inner) or inner

    @staticmethod
    def to_str(t: Any) -> str:
        if isinstance(t, TypeNode):
            return TypeStringVisitor().stringify(t)
        return TypeStringVisitor().stringify(TypeTreeBuilder.build(t))


class TypeStringVisitor:
    def stringify(self, node: TypeNode) -> str:
        return node.accept(self)

    def visit_optional(self, node: OptionalNode) -> str:
        return self.stringify(node.inner)

    def visit_primitive(self, node: PrimitiveNode) -> str:
        if hasattr(node.py_type, "__name__"):
            return node.py_type.__name__
        return str(node.annotation).replace("typing.", "")

    def visit_bool(self, node: BoolNode) -> str:
        return "bool"

    def visit_literal(self, node: LiteralNode) -> str:
        return f"{{{','.join(map(str, node.choices))}}}"

    def visit_enum(self, node: EnumNode) -> str:
        return node.enum_type.__name__

    def visit_list(self, node: ListNode) -> str:
        return f"list[{self.stringify(node.inner)}]"

    def visit_fixed_tuple(self, node: FixedTupleNode) -> str:
        return f"tuple[{', '.join(self.stringify(item) for item in node.items)}]"

    def visit_variadic_tuple(self, node: VariadicTupleNode) -> str:
        return f"tuple[{self.stringify(node.inner)}, ...]"

    def visit_dataclass_ref(self, node: DataclassRefNode) -> str:
        return node.cls.__name__

    def visit_dataclass(self, node: DataclassNode) -> str:
        return node.cls.__name__


class ScalarValueParser:
    @staticmethod
    def parse(node: TypeNode, value: Any) -> Any:
        if isinstance(node, OptionalNode):
            return None if value is None else ScalarValueParser.parse(node.inner, value)
        if isinstance(node, BoolNode):
            if isinstance(value, bool):
                return value
            return str(value).lower() in ("true", "1", "yes", "on")
        if isinstance(node, EnumNode):
            return node.enum_type(value)
        if isinstance(node, LiteralNode):
            if value not in node.choices and node.choices:
                choice_type = type(node.choices[0])
                value = value if isinstance(value, choice_type) else choice_type(value)
            if value not in node.choices:
                raise argparse.ArgumentTypeError(
                    f"invalid choice: {value!r} (choose from {', '.join(map(str, node.choices))})"
                )
            return value
        if isinstance(node, PrimitiveNode):
            if value is None or not isinstance(node.py_type, type):
                return value
            return value if isinstance(value, node.py_type) else node.py_type(value)
        return value


class ConfigTreeBuilder:
    def build(
        self,
        cls: Type,
        default_instance: Any = None,
        prefix: Tuple[str, ...] = (),
        skip_source_fields: bool = False,
        bind_enabled: bool = True,
    ) -> DataclassNode:
        docs = DocstringResolver.resolve(cls)
        hints = TypeReflector.resolve_hints(cls)
        nodes = []

        for f in fields(cls):
            field_type = hints.get(f.name, f.type)
            type_node = TypeTreeBuilder.build(field_type)
            default = self._get_effective_default(f, default_instance)
            external = f.metadata.get("external", False)
            source_has_field = self._source_has_field(default_instance, f.name)
            should_skip = skip_source_fields and source_has_field
            path = (*prefix, f.name)

            dataclass_cls = self._dataclass_cls(type_node)
            if dataclass_cls is not None:
                nested_default = None if default is dataclasses.MISSING else default
                nested_cls = self._resolve_nested_dataclass_type(
                    dataclass_cls, nested_default
                )
                child_bind_enabled = bind_enabled and not external
                child_skip = skip_source_fields and source_has_field
                nested_node = self.build(
                    nested_cls,
                    default_instance=nested_default,
                    prefix=path,
                    skip_source_fields=child_skip,
                    bind_enabled=child_bind_enabled,
                )
                type_node = self._replace_dataclass_ref(type_node, nested_node)

            bind = bind_enabled and not external and not should_skip
            if isinstance(self._unwrap_optional(type_node), DataclassNode):
                bind = False

            nodes.append(
                FieldNode(
                    name=f.name,
                    path=path,
                    type_node=type_node,
                    default=default,
                    bind=bind,
                    help=f.metadata.get("help") or docs.get(f.name, ""),
                    metadata=dict(f.metadata),
                    field=f,
                )
            )

        return DataclassNode(cls=cls, fields=tuple(nodes), annotation=cls)

    def _unwrap_optional(self, node: TypeNode) -> TypeNode:
        return node.inner if isinstance(node, OptionalNode) else node

    def _dataclass_cls(self, node: TypeNode) -> Type | None:
        node = self._unwrap_optional(node)
        return node.cls if isinstance(node, DataclassRefNode) else None

    def _replace_dataclass_ref(
        self, original: TypeNode, dataclass_node: DataclassNode
    ) -> TypeNode:
        if isinstance(original, OptionalNode):
            return OptionalNode(dataclass_node, original.annotation)
        return dataclass_node

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

    def _get_effective_default(self, f: dataclasses.Field, instance: Any) -> Any:
        if self._source_has_field(instance, f.name):
            return self._get_source_value(instance, f.name)
        if f.default is not dataclasses.MISSING:
            return f.default
        if f.default_factory is not dataclasses.MISSING:
            return f.default_factory()
        return dataclasses.MISSING

    def _resolve_nested_dataclass_type(
        self, annotated_type: Type, default_instance: Any
    ) -> Type:
        if is_dataclass(default_instance) and isinstance(
            default_instance, annotated_type
        ):
            if type(default_instance) is not annotated_type:
                return type(default_instance)
        return annotated_type


class ArgumentAdapter:
    def __init__(self, parser: argparse.ArgumentParser):
        self.groups = {
            True: parser.add_argument_group("Required arguments"),
            False: parser.add_argument_group("Optional arguments"),
        }

    def add(self, arg: ArgSpec):
        required = arg.kwargs.get("required", False)
        self.groups[required].add_argument(*arg.flags, **arg.kwargs)


class ArgparseBindVisitor:
    def __init__(self, adapter: ArgumentAdapter):
        self.adapter = adapter

    def bind(self, node: DataclassNode):
        node.accept(self)

    def visit_dataclass(self, node: DataclassNode, field: FieldNode | None = None):
        for child in node.fields:
            child.type_node.accept(self, child)

    def visit_optional(self, node: OptionalNode, field: FieldNode):
        node.inner.accept(self, field)

    def visit_dataclass_ref(self, node: DataclassRefNode, field: FieldNode):
        return

    def visit_primitive(self, node: PrimitiveNode, field: FieldNode):
        py_type = node.py_type if isinstance(node.py_type, type) else str
        kwargs = self._base_kwargs(field)
        kwargs["type"] = py_type
        if field.default is dataclasses.MISSING:
            kwargs["required"] = True
            kwargs.pop("default", None)
        kwargs.setdefault("metavar", TypeReflector.to_str(node).upper())
        self._add(field, kwargs)

    def visit_bool(self, node: BoolNode, field: FieldNode):
        kwargs = self._base_kwargs(field)
        kwargs["action"] = BooleanAction
        kwargs.pop("metavar", None)
        self._add(field, kwargs)

    def visit_literal(self, node: LiteralNode, field: FieldNode):
        kwargs = self._base_kwargs(field)
        kwargs["choices"] = node.choices
        kwargs["type"] = type(node.choices[0]) if node.choices else str
        kwargs["metavar"] = f"{{{','.join(map(str, node.choices))}}}"
        self._add(field, kwargs)

    def visit_enum(self, node: EnumNode, field: FieldNode):
        choices = [item.value for item in node.enum_type]
        kwargs = self._base_kwargs(field)
        kwargs["choices"] = choices
        kwargs["type"] = type(choices[0]) if choices else str
        kwargs["metavar"] = f"{{{','.join(map(str, choices))}}}"
        self._add(field, kwargs)

    def visit_list(self, node: ListNode, field: FieldNode):
        kwargs = self._base_kwargs(field)
        kwargs["action"] = ListAction
        kwargs["inner_node"] = node.inner
        kwargs["nargs"] = "*" if field.default is not dataclasses.MISSING else "+"
        kwargs["metavar"] = self._list_metavar(node.inner)
        self._add(field, kwargs)

    def visit_fixed_tuple(self, node: FixedTupleNode, field: FieldNode):
        kwargs = self._base_kwargs(field)
        kwargs["action"] = TupleAction
        kwargs["types"] = tuple(item.annotation for item in node.items)
        kwargs["nargs"] = len(node.items)
        kwargs["metavar"] = tuple(
            TypeReflector.to_str(item).upper() for item in node.items
        )
        self._add(field, kwargs)

    def visit_variadic_tuple(self, node: VariadicTupleNode, field: FieldNode):
        kwargs = self._base_kwargs(field)
        kwargs["action"] = TupleAction
        kwargs["types"] = (node.inner.annotation, Ellipsis)
        kwargs["nargs"] = "*" if field.default is not dataclasses.MISSING else "+"
        kwargs["metavar"] = TypeReflector.to_str(node.inner).upper()
        self._add(field, kwargs)

    def _base_kwargs(self, field: FieldNode) -> Dict[str, Any]:
        return {
            "dest": field.dest,
            "default": (
                field.default if field.default is not dataclasses.MISSING else None
            ),
        }

    def _add(self, field: FieldNode, kwargs: Dict[str, Any]):
        if not field.bind:
            return

        kwargs.update(field.metadata.get("argparse_kwargs", {}))
        if kwargs.get("action") is BooleanAction:
            kwargs.pop("metavar", None)
        kwargs["help"] = self._format_help(
            field.help,
            TypeReflector.to_str(field.type_node),
            field.default,
            kwargs.get("required", False),
        )

        flags = [field.flag]
        if field.metadata.get("short"):
            flags.append(field.metadata["short"])
        self.adapter.add(ArgSpec(flags=flags, kwargs=kwargs))

    def _list_metavar(self, node: TypeNode) -> str:
        if isinstance(node, LiteralNode):
            return f"{{{','.join(map(str, node.choices))},...}}"
        return TypeReflector.to_str(node).upper()

    def _format_help(
        self, base_help: str, type_str: str, default: Any, required: bool
    ) -> str:
        parts = []
        if required:
            parts.append("required")
        elif default is not dataclasses.MISSING:
            val = default.name if isinstance(default, enum.Enum) else str(default)
            parts.append(f"default: {val}")

        meta = f"[dim]({', '.join(parts)})[/dim]" if parts else ""
        return f"{base_help}  {meta}" if base_help else meta


class DeserializeVisitor:
    def reconstruct(
        self,
        node: DataclassNode,
        data: Dict[str, Any],
        fallback_instance: Any = None,
    ) -> Any:
        return node.accept(self, data, fallback_instance)

    def visit_dataclass(
        self,
        node: DataclassNode,
        data: Dict[str, Any],
        fallback_instance: Any = None,
    ) -> Any:
        init_kwargs = {}

        for field in node.fields:
            value = self._field_value(field, data, fallback_instance)
            if value is not dataclasses.MISSING:
                init_kwargs[field.name] = value

        return node.cls(**init_kwargs)

    def _field_value(
        self,
        field: FieldNode,
        data: Dict[str, Any],
        fallback_instance: Any,
    ) -> Any:
        node = field.type_node
        bare_node = node.inner if isinstance(node, OptionalNode) else node

        if isinstance(bare_node, DataclassNode):
            fallback = self._default_or_source(field, fallback_instance)
            if fallback is None and not self._has_data_for(bare_node, data):
                return None if isinstance(node, OptionalNode) else dataclasses.MISSING
            return bare_node.accept(self, data, fallback)

        if field.dest in data:
            return node.accept(self, data[field.dest])

        if self._source_has_field(fallback_instance, field.name):
            return self._get_source_value(fallback_instance, field.name)

        return dataclasses.MISSING

    def _default_or_source(self, field: FieldNode, fallback_instance: Any) -> Any:
        if self._source_has_field(fallback_instance, field.name):
            return self._get_source_value(fallback_instance, field.name)
        if field.default is dataclasses.MISSING:
            return None
        return field.default

    def _has_data_for(self, node: DataclassNode, data: Dict[str, Any]) -> bool:
        prefix = ".".join(node.fields[0].path[:-1]) + "." if node.fields else ""
        return any(key.startswith(prefix) for key in data)

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

    def visit_optional(self, node: OptionalNode, value: Any) -> Any:
        return None if value is None else node.inner.accept(self, value)

    def visit_primitive(self, node: PrimitiveNode, value: Any) -> Any:
        return ScalarValueParser.parse(node, value)

    def visit_bool(self, node: BoolNode, value: Any) -> bool:
        return ScalarValueParser.parse(node, value)

    def visit_literal(self, node: LiteralNode, value: Any) -> Any:
        return ScalarValueParser.parse(node, value)

    def visit_enum(self, node: EnumNode, value: Any) -> enum.Enum:
        return ScalarValueParser.parse(node, value)

    def visit_list(self, node: ListNode, value: Any) -> Any:
        if value is None:
            return None
        return [node.inner.accept(self, item) for item in value]

    def visit_fixed_tuple(self, node: FixedTupleNode, value: Any) -> Any:
        if value is None:
            return None
        return tuple(item.accept(self, val) for item, val in zip(node.items, value))

    def visit_variadic_tuple(self, node: VariadicTupleNode, value: Any) -> Any:
        if value is None:
            return None
        return tuple(node.inner.accept(self, item) for item in value)

    def visit_dataclass_ref(self, node: DataclassRefNode, value: Any) -> Any:
        return value


class ArgParser(Generic[T]):
    def __init__(
        self, target_cls: Type[T], default_instance: Any = None, overlay: bool = False
    ):
        self.target_cls = target_cls
        self.default_instance = default_instance
        self.parser = argparse.ArgumentParser(
            formatter_class=TyroStyleHelpFormatter, description=target_cls.__doc__
        )
        self.unknown_args = []

        self.tree = ConfigTreeBuilder().build(
            target_cls,
            default_instance=default_instance,
            skip_source_fields=default_instance is not None and not overlay,
        )
        self.adapter = ArgumentAdapter(self.parser)
        ArgparseBindVisitor(self.adapter).bind(self.tree)

    def parse(self, args: Optional[List[str]] = None) -> T:
        from_cli = args is None
        if args is None:
            args = sys.argv[1:]
        namespace, self.unknown_args = self.parser.parse_known_args(args)
        if from_cli:
            sys.argv = [sys.argv[0], *self.unknown_args]
        return DeserializeVisitor().reconstruct(
            self.tree, vars(namespace), fallback_instance=self.default_instance
        )

    @classmethod
    def transform(
        cls, source: Any, target_cls: Type[T], args: Optional[List[str]] = None
    ) -> T:
        if isinstance(source, target_cls):
            return source
        parser = cls(target_cls, default_instance=source)
        return parser.parse(args)

    @staticmethod
    def as_dict(cfg: Any) -> Dict[str, Any]:
        return asdict(cfg)
