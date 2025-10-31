import argparse
import re
from dataclasses import dataclass
from textwrap import dedent
from typing import (
    Any,
    List,
    Optional,
    Union,
    _GenericAlias,
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
        argparse_configs = {}
        try:
            type_hints = get_type_hints(new_cls)
        except Exception as e:
            type_hints = getattr(new_cls, "__annotations__", {})
        attr_docs = cls._parse_attr_docs(namespace.get("__doc__", ""))

        for attr_name, type_hint in type_hints.items():
            if attr_name.startswith("_") or callable(namespace.get(attr_name)):
                continue
            assigned_value = namespace.get(attr_name, ...)
            arg_opt: Optional[ArgsOpt] = None
            explicit_help = None
            short_alias = None
            has_explicit_default = assigned_value is not ...
            # --- Check if ArgsOpt is used ---
            if isinstance(assigned_value, ArgsOpt):
                arg_opt = assigned_value
                explicit_help = arg_opt.help
                short_alias = arg_opt.short
                # Check if ArgsOpt specifies a default
                if arg_opt.default is not _ARGS_OPT_DEFAULT_SENTINEL:
                    has_explicit_default = True  # ArgsOpt provides the default
                    assigned_value = arg_opt.default  # Use this as the effective assigned value
                else:
                    # ArgsOpt used without default, treat as no default assigned
                    has_explicit_default = False
                    assigned_value = ...
            help_text = explicit_help or attr_docs.get(attr_name) or f"Value for {attr_name}"

            # --- Determine type details ---
            origin = get_origin(type_hint)
            type_args = get_args(type_hint)
            is_optional_syntax = origin is Union and type(None) in type_args
            is_list_syntax = origin is list or origin is List

            actual_type = type_hint
            if is_optional_syntax:
                actual_type = next((t for t in type_args if t is not type(None)), type_hint)
            elif is_list_syntax and type_args:
                actual_type = type_args[0]
            # Tyro style
            type_name = getattr(actual_type, "__name__", str(actual_type))
            if isinstance(type_hint, _GenericAlias):
                type_name = str(type_hint).replace("typing.", "")

            help_parts = [help_text, f"[dim](type: {type_name})[/dim]"]

            is_required = not has_explicit_default and not is_optional_syntax
            final_default = None

            if not is_required:
                if has_explicit_default:
                    # Use the default found (either direct or from ArgsOpt)
                    final_default = assigned_value
                elif is_list_syntax:
                    # Optional list without default -> empty list
                    final_default = []
                # else: final_default is None, which is correct
                if final_default is not None and final_default != []:
                    help_parts.append(f"[dim](default: {final_default!r})[/dim]")

            final_help = " ".join(help_parts)

            # --- Argument Parser Config ---
            long_name = f"--{attr_name.replace('_', '-')}"
            arg_names = [alias for alias in [short_alias, long_name] if alias]
            kwargs = {"dest": attr_name, "help": final_help}

            if is_required:
                kwargs["required"] = True
            else:
                kwargs["default"] = final_default

            if actual_type is bool:
                if not kwargs.get("default", False):
                    kwargs["action"] = "store_true"
                else:
                    kwargs["action"] = "store_false"
            elif is_list_syntax:
                kwargs["type"] = actual_type if actual_type not in [list, List] else str
                kwargs["nargs"] = "+" if is_required else "*"
            elif actual_type in [int, float, str]:
                kwargs["type"] = actual_type
            else:
                print(f"Warning: Unsupported type: {actual_type} for {attr_name}")
                continue

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
        # Regex to find the start of an attribute description
        # It looks for "attribute_name (optional_type): description"
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
    """Base class for argument parsing using argparse.
    `...` indicates the attribution is required and no default assigned.
    `None` indicates the attribution is optional with default None.
    Attributes are defined via type hints. Use `ArgsOpt` to provide
    defaults and argparse options simultaneously.
    """

    def __init__(self, **kwargs):
        # Initialize attributes based on parsed values or defaults held by argparse
        config_keys = getattr(self.__class__, "_argparse_configs", {}).keys()
        type_hints = {}
        try:  # Add try-except for robustness, especially with forward refs
            type_hints = get_type_hints(self.__class__)
        except NameError:
            pass
        # Prioritize values from kwargs (parsed args)
        for attr_name, value in kwargs.items():
            # Ensure we only set attributes that are expected (part of type hints or class vars)
            if hasattr(self.__class__, attr_name):  # A simple check
                setattr(self, attr_name, value)
        for attr_name in type_hints:
            if not hasattr(self, attr_name):  # If not set by kwargs
                if hasattr(self.__class__, attr_name):
                    class_val = getattr(self.__class__, attr_name)
                    if not callable(class_val) and not isinstance(class_val, ArgsOpt):
                        setattr(self, attr_name, class_val)
                # Handle Optional[T] without default assigned -> should be None
                elif get_origin(type_hints[attr_name]) is Union and type(None) in get_args(
                    type_hints[attr_name]
                ):
                    setattr(self, attr_name, None)

    @classmethod
    def add_args_to_parser(cls, parser: argparse.ArgumentParser):
        if not hasattr(cls, "_argparse_configs"):
            return

        required_args = []
        optional_args = []

        for _, (arg_names, kwargs) in cls._argparse_configs.items():
            if kwargs.get("required", False):
                required_args.append((arg_names, kwargs))
            else:
                optional_args.append((arg_names, kwargs))

        if required_args:
            required_group = parser.add_argument_group("Required arguments")
            for arg_names, kwargs in required_args:
                required_group.add_argument(*arg_names, **kwargs)

        if optional_args:
            optional_group = parser.add_argument_group("Optional arguments")
            for arg_names, kwargs in optional_args:
                optional_group.add_argument(*arg_names, **kwargs)

    @classmethod
    def parse(cls, args_list: Optional[List[str]] = None) -> "ArgBase":
        docstring = cls.__doc__ or ""
        parts = re.split(
            r"\n\s*(?:Attributes|Args|Parameters):", docstring, maxsplit=1, flags=re.IGNORECASE
        )
        description = parts[0].strip() if parts else f"Arguments for {cls.__name__}"
        parser = argparse.ArgumentParser(
            description=description,
            formatter_class=TyroStyleHelpFormatter,
            add_help=False,
        )
        cls.add_args_to_parser(parser)
        parser._action_groups[1].add_argument(
            "-h",
            "--help",
            action="help",
            default=argparse.SUPPRESS,
            help="Show this help message and exit.",
        )
        args_ns = parser.parse_args(args_list)
        return cls(**vars(args_ns))
