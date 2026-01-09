import argparse
import copy
import enum
import re
from dataclasses import dataclass
from textwrap import dedent
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from rich.table import Table
from rich.text import Text
from rich_argparse import RichHelpFormatter

__all__ = ["ArgsOpt", "ArgBase"]


_ARGS_OPT_DEFAULT_SENTINEL = object()


class TyroStyleHelpFormatter(RichHelpFormatter):
    """A tyro style help formatter"""

    def __init__(self, *args, **kwargs):
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
            metavar = self._get_default_metavar_for_optional(action)
            if metavar and metavar in invocation_str:
                command, _, _ = invocation_str.rpartition(f" {metavar}")
                invocation_text = Text.assemble(
                    (command, self.styles["argparse.args"]),
                    (f" {metavar}", self.styles["argparse.metavar"]),
                )
            else:  # For simple flags like --verbose
                invocation_text = Text(invocation_str, style=self.styles["argparse.args"])
            help_text = Text.from_markup(action.help or "", style=self.styles["argparse.help"])
            wrapper_table.add_row("│", " ", invocation_text, " ", help_text, "│")

        self.console.print(wrapper_table)


@dataclass
class Positional:
    help: Optional[str] = None


@dataclass
class ArgsOpt:
    """
    Utility class to provide defaults and argparse options simultaneously.
    Assign this instead of a direct value to customize parsing.
    Using ArgsOpt generally implies the argument is Optional, unless
    a non-None default is provided.
    Attributes:
        default (Any): The default value. If not set here (remains sentinel),
            the behavior depends on the type hint (Optional[T] -> None,
            List[T] -> [], other T -> Error or handled by argparse).
        help (Optional[str]): Help text. Overrides class docstring help.
        short (Optional[str]): Short alias (e.g., '-a').
    """

    default: Any = _ARGS_OPT_DEFAULT_SENTINEL
    help: Optional[str] = None
    short: Optional[str] = None


class ArgParseMeta(type):
    def __new__(cls, name: str, bases: tuple, namespace: dict):
        new_cls = super().__new__(cls, name, bases, namespace)
        attr_docs = {}
        for base in reversed(bases):
            if hasattr(base, "__mro__"):
                for ancestor in reversed(base.__mro__):
                    if ancestor is object:
                        continue
                    doc = getattr(ancestor, "__doc__", "")
                    attr_docs.update(cls._parse_attr_docs(doc))
        current_doc = namespace.get("__doc__", "")
        attr_docs.update(cls._parse_attr_docs(current_doc))

        argparse_configs = {}
        for base in reversed(bases):
            if hasattr(base, "_argparse_configs"):
                argparse_configs.update(copy.deepcopy(base._argparse_configs))
        try:
            type_hints = get_type_hints(new_cls)
        except Exception:
            type_hints = getattr(new_cls, "__annotations__", {})
        for attr_name, type_hint in type_hints.items():
            if attr_name.startswith("_") or callable(namespace.get(attr_name)):
                continue
            assigned_value = getattr(new_cls, attr_name, ...)
            arg_opt = assigned_value if isinstance(assigned_value, ArgsOpt) else None

            origin = get_origin(type_hint)
            is_optional = origin is Union and type(None) in get_args(type_hint)
            actual_type = type_hint
            if origin is Union:
                actual_type = next(
                    (t for t in get_args(type_hint) if t is not type(None)), type_hint
                )
            if isinstance(actual_type, type) and issubclass(actual_type, ArgBase):
                child_configs = getattr(actual_type, "_argparse_configs", {})
                prefix = attr_name.replace("_", "-")
                for child_dest, (child_flags, child_kwargs) in child_configs.items():
                    new_dest = f"{attr_name}.{child_dest}"
                    new_flags = []
                    for flag in child_flags:
                        if flag.startswith("--"):
                            suffix = flag.lstrip("-")
                            new_flags.append(f"--{prefix}.{suffix}")
                    new_kwargs = copy.deepcopy(child_kwargs)
                    new_kwargs["dest"] = new_dest
                    if is_optional and new_kwargs.get("required"):
                        del new_kwargs["required"]
                    argparse_configs[new_dest] = (new_flags, new_kwargs)
                continue

            has_explicit_default = False
            default_val = ...
            if arg_opt:
                if arg_opt.default is not _ARGS_OPT_DEFAULT_SENTINEL:
                    has_explicit_default = True
                    default_val = arg_opt.default
            elif assigned_value is not ...:
                has_explicit_default = True
                default_val = assigned_value
            help_text = (
                (arg_opt.help if arg_opt else None)
                or attr_docs.get(attr_name)
                or f"Value for {attr_name}"
            )
            long_name = f"--{attr_name.replace('_', '-')}"
            arg_names = [
                alias for alias in [arg_opt.short if arg_opt else None, long_name] if alias
            ]
            kwargs = {"dest": attr_name}

            actual_origin = get_origin(actual_type)
            if actual_origin is Literal:
                choices = get_args(actual_type)
                kwargs.update({"choices": choices, "type": type(choices[0])})
            elif isinstance(actual_type, type) and issubclass(actual_type, enum.Enum):
                choices = [e.value for e in actual_type]
                kwargs.update(
                    {
                        "choices": choices,
                        "type": type(choices[0]),
                    }
                )
                if has_explicit_default:
                    kwargs["default"] = default_val.value
                type_str = f"Enum[{actual_type.__name__}]"
            elif actual_type is bool:
                kwargs["action"] = (
                    "store_false"
                    if (has_explicit_default and default_val is True)
                    else "store_true"
                )
            elif actual_origin in (list, List):
                item_type = get_args(actual_type)[0] if get_args(actual_type) else str
                kwargs.update(
                    {
                        "type": item_type,
                        "nargs": "+" if not has_explicit_default and not is_optional else "*",
                    }
                )
            elif actual_type in [int, float, str]:
                kwargs["type"] = actual_type
            else:
                continue

            is_required = not has_explicit_default and not is_optional
            if is_required:
                kwargs["required"] = True
            else:
                kwargs["default"] = (
                    default_val
                    if has_explicit_default
                    else ([] if actual_origin in (list, List) else None)
                )
            help_parts = [help_text]
            if "action" not in kwargs:
                type_str = getattr(actual_type, "__name__", str(actual_type))
                help_parts.append(f"[dim]({type_str})[/dim]")
            if has_explicit_default and default_val is not None:
                help_parts.append(f"[dim](default: {default_val!r})[/dim]")
            kwargs["help"] = " ".join(help_parts)
            argparse_configs[attr_name] = (arg_names, kwargs)
        new_cls._argparse_configs = argparse_configs
        return new_cls

    @staticmethod
    def _parse_attr_docs(docstring: Optional[str]) -> dict[str, str]:
        """
        Parses the docstring to extract help text for attributes.
        This version looks for a specific section like 'Args:', 'Attributes:', or 'Parameters:'.
        """
        if not docstring:
            return {}

        # Dedent the docstring to handle indentation
        docstring = dedent(docstring)
        docs = {}
        current_attr = None
        in_attr_section = False
        attr_start_regex = re.compile(r"^\s*(\w+)\s*(?:\(.*\))?:\s*(.*)")

        lines = docstring.strip().splitlines()
        for line in lines:
            # Check for the start of the attributes section
            if line.strip().lower() in ("args:", "attributes:", "parameters:"):
                in_attr_section = True
                current_attr = None  # Reset when entering the section
                continue
            if not in_attr_section:
                continue
            match = attr_start_regex.match(line)
            if match:
                # This line starts a new attribute
                current_attr, help_text = match.groups()
                docs[current_attr] = help_text.strip()
            elif current_attr and line.strip() and (line.startswith(" ") or line.startswith("\t")):
                # This is a continuation of the previous attribute's help text
                docs[current_attr] += " " + line.strip()
            else:
                # Not a recognized pattern, reset current attribute
                current_attr = None
        return docs


class ArgBase(metaclass=ArgParseMeta):
    _unknown_args: List[str] = []

    def __init__(self, **kwargs):
        configs = getattr(self.__class__, "_argparse_configs", {})

        nested_data = {}
        for key, value in kwargs.items():
            if "." in key:
                parent, child = key.split(".", 1)
                nested_data.setdefault(parent, {})[child] = value
            else:
                nested_data[key] = value

        # Instantiate Attributes by iterating over Type Hints
        try:
            type_hints = get_type_hints(self.__class__)
        except Exception:
            type_hints = getattr(self.__class__, "__annotations__", {})

        for attr_name, type_hint in type_hints.items():
            if attr_name.startswith("_"):
                continue
            origin = get_origin(type_hint)
            actual_type = type_hint
            if origin is Union:
                actual_type = next(
                    (t for t in get_args(type_hint) if t is not type(None)), type_hint
                )
            is_nested = isinstance(actual_type, type) and issubclass(actual_type, ArgBase)
            if attr_name in nested_data:
                val = nested_data[attr_name]
                if is_nested and isinstance(val, dict):
                    # Recursively instantiate the child class
                    setattr(self, attr_name, actual_type(**val))
                else:
                    # Assign primitive directly
                    setattr(self, attr_name, copy.deepcopy(val))
            else:
                if is_nested:
                    if origin is Union and type(None) in get_args(type_hint):
                        setattr(self, attr_name, None)
                    else:
                        setattr(self, attr_name, actual_type())
                elif attr_name in configs:
                    _, argparse_kwargs = configs[attr_name]
                    default = argparse_kwargs.get("default")
                    if default is ...:
                        raise ValueError(f"Missing required argument: {attr_name}")
                    val = copy.deepcopy(default) if default is not None else None
                    setattr(self, attr_name, val)

    @classmethod
    def add_args_to_parser(cls, parser: argparse.ArgumentParser):
        configs = getattr(cls, "_argparse_configs", {})
        req_group = next(
            (g for g in parser._action_groups if g.title == "Required arguments"), None
        )
        if req_group is None:
            req_group = parser.add_argument_group("Required arguments")

        opt_group = next(
            (g for g in parser._action_groups if g.title == "Optional arguments"), None
        )
        if opt_group is None:
            opt_group = parser.add_argument_group("Optional arguments")

        for _, (arg_names, kwargs) in configs.items():
            target = req_group if kwargs.get("required") else opt_group
            target.add_argument(*arg_names, **kwargs)

    @classmethod
    def parser(cls) -> argparse.ArgumentParser:
        doc = cls.__doc__ or ""
        match = re.split(r"\n\s*(?:Attributes|Args|Parameters):", doc, flags=re.IGNORECASE)[0]
        desc = match[0].strip() if match else f"Arguments for {cls.__name__}"
        parser = argparse.ArgumentParser(
            description=desc, formatter_class=TyroStyleHelpFormatter, add_help=True
        )
        cls.add_args_to_parser(parser)
        return parser

    @classmethod
    def parse(
        cls,
        args_list: Optional[List[str]] = None,
        *,
        parser: Optional[argparse.ArgumentParser] = None,
        final: bool = False,
    ) -> "ArgBase":
        if parser is None:
            parser = cls.parser()

        unknown = []
        if final:
            args_ns = parser.parse_args(args_list)
        else:
            args_ns, unknown = parser.parse_known_args(args_list)
        instance = cls(**vars(args_ns))
        instance._unknown_args = unknown
        return instance

    def to_dict(self, recurse: bool = True) -> Dict[str, Any]:
        configs = getattr(self.__class__, "_argparse_configs", {})
        res = {}
        for k in configs.keys():
            if hasattr(self, k):
                val = getattr(self, k)
                if recurse and isinstance(val, ArgBase):
                    res[k] = val.to_dict()
                else:
                    res[k] = val
        return res

    def to_namespace(self) -> argparse.Namespace:
        """ """
        return argparse.Namespace(**self.to_dict(recurse=False))
