from __future__ import annotations

import dataclasses
import datetime
import enum
import json
import pathlib
import runpy
from typing import Any, overload, Type, TypeVar

import coolname

T = TypeVar("T")


class ConfigCodec:
    """Serialize a resolved spec dataclass to/from a self-contained ``config.py``.

    ``dumps`` renders the spec as valid Python source: dataclasses become
    ``Cls(field=...)`` calls (preserving the concrete polymorphic subtype),
    enums render as ``Cls.MEMBER``, containers recurse. Every referenced class
    is collected into an import header so the emitted module is importable and
    reconstructs the exact object via ``loads``.
    """

    _HEADER = (
        "# Auto-generated experiment config -- the resolved spec that actually ran.\n"
        "# Reproduce: load this file as the base spec (CLI flags overlay on top).\n"
    )

    @classmethod
    def dumps(cls, spec) -> str:
        imports: set = set()
        expr = cls._render(spec, imports)
        header = [cls._HEADER]
        header += [
            f"from {mod} import {qual.split('.')[0]}" for mod, qual in sorted(imports)
        ]
        return "\n".join([*header, "", f"spec = {expr}", ""])

    @overload
    @classmethod
    def loads(cls, path: str | pathlib.Path, spec_cls: Type[T]) -> T: ...

    @overload
    @classmethod
    def loads(cls, path: str | pathlib.Path, spec_cls: None = None) -> Any: ...

    @classmethod
    def loads(cls, path: str | pathlib.Path, spec_cls: Type[T] | None = None):
        spec = runpy.run_path(str(path))["spec"]
        if spec_cls is not None and not isinstance(spec, spec_cls):
            raise TypeError(
                f"{path} defines a {type(spec).__name__}, expected {spec_cls.__name__}"
            )
        return spec

    @classmethod
    def _render(cls, value, imports: set) -> str:
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            klass = type(value)
            imports.add((klass.__module__, klass.__qualname__))
            args = ", ".join(
                f"{f.name}={cls._render(getattr(value, f.name), imports)}"
                for f in dataclasses.fields(value)
                if f.init and f.metadata.get("serialize", True)
            )
            return f"{klass.__qualname__.split('.')[-1]}({args})"
        if isinstance(value, enum.Enum):
            klass = type(value)
            imports.add((klass.__module__, klass.__qualname__))
            return f"{klass.__qualname__.split('.')[-1]}.{value.name}"
        if isinstance(value, tuple):
            inner = ", ".join(cls._render(v, imports) for v in value)
            return f"({inner},)" if len(value) == 1 else f"({inner})"
        if isinstance(value, list):
            return "[" + ", ".join(cls._render(v, imports) for v in value) + "]"
        if isinstance(value, dict):
            items = ", ".join(
                f"{cls._render(k, imports)}: {cls._render(v, imports)}"
                for k, v in value.items()
            )
            return "{" + items + "}"
        return repr(value)


class SpecDiff:
    """Structural diff between two resolved spec trees.

    Returns a mapping of dotted field path -> ``(old, new)`` for every leaf that
    differs. A polymorphic subtype change (a node whose concrete dataclass type
    differs) is reported as one whole-node difference rather than recursing.
    Used to surface what changed when warm-starting from a saved ``config.py``.
    """

    @classmethod
    def compare(cls, old, new, _path: str = "") -> dict:
        diffs: dict = {}
        if (
            dataclasses.is_dataclass(old)
            and dataclasses.is_dataclass(new)
            and type(old) is type(new)
        ):
            for f in dataclasses.fields(old):
                child = f"{_path}.{f.name}" if _path else f.name
                diffs.update(
                    cls.compare(getattr(old, f.name), getattr(new, f.name), child)
                )
            return diffs
        if old != new:
            diffs[_path or "<root>"] = (old, new)
        return diffs


class GitProbe:
    """Interrogate a git working tree for reproducibility provenance."""

    def __init__(self, repo_root: str | pathlib.Path):
        self.repo_root = pathlib.Path(repo_root)

    def _run(self, *args: str) -> str:
        import subprocess

        return subprocess.run(
            ["git", *args],
            cwd=self.repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    def available(self) -> bool:
        try:
            self._run("rev-parse", "--is-inside-work-tree")
            return True
        except Exception:
            return False

    def diff(self) -> str:
        """Uncommitted changes to tracked files (``git diff HEAD``)."""
        return self._run("diff", "HEAD")

    def toplevel(self) -> pathlib.Path | None:
        """Absolute path to the repository root, or None when not in a repo."""
        try:
            return pathlib.Path(self._run("rev-parse", "--show-toplevel").strip())
        except Exception:
            return None

    def provenance(self) -> dict:
        if not self.available():
            return {"available": False}
        return {
            "available": True,
            "sha": self._run("rev-parse", "HEAD").strip(),
            "branch": self._run("rev-parse", "--abbrev-ref", "HEAD").strip(),
            "has_untracked": bool(
                self._run("ls-files", "--others", "--exclude-standard").strip()
            ),
        }


class Workspace:
    """
    Args:
        ...
    """

    def __init__(
        self, root: str, project: str, run_name: str = None, timestamp: bool = False
    ):
        self.root = pathlib.Path(root)
        self.project = project
        self.run_name = run_name if run_name else coolname.generate_slug(2)
        if timestamp:
            time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            self.run_name = f"{time_stamp}-{self.run_name}"

    @classmethod
    def create(
        cls, root: str, project: str, run_name: str = None, timestamp: bool = False
    ) -> Workspace:
        """Construct a workspace for a brand-new run (directories created lazily)."""
        return cls(root, project, run_name, timestamp)

    def ensure(self) -> Workspace:
        """Single authority for materializing the run directory tree on disk."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return self

    @property
    def config_path(self) -> pathlib.Path:
        return self.run_dir / "config.py"

    def save_config(self, spec) -> pathlib.Path:
        """Persist the resolved spec as a reproducible, importable ``config.py``."""
        self.ensure()
        self.config_path.write_text(ConfigCodec.dumps(spec))
        return self.config_path

    @overload
    def load_config(self, spec_cls: Type[T]) -> T: ...

    @overload
    def load_config(self, spec_cls: None = None) -> Any: ...

    def load_config(self, spec_cls: Type[T] | None = None):
        """Reconstruct the spec from this run's ``config.py``.

        Pass the expected spec class to get a statically-typed result (and a
        runtime check that the config really holds that type).
        """
        return ConfigCodec.loads(self.config_path, spec_cls)

    def checkpoint_path(self, step: int) -> pathlib.Path:
        return self.checkpoint_dir / f"weights_{step}.pth"

    def _checkpoint_steps(self) -> list[int]:
        if not self.checkpoint_dir.exists():
            return []
        steps = []
        for p in self.checkpoint_dir.glob("weights_*.pth"):
            try:
                steps.append(int(p.stem.split("_")[-1]))
            except ValueError:
                continue
        return sorted(steps)

    def latest_checkpoint(self) -> pathlib.Path | None:
        steps = self._checkpoint_steps()
        return self.checkpoint_path(steps[-1]) if steps else None

    def best_checkpoint(self) -> pathlib.Path | None:
        marker = self.checkpoint_dir / "best.json"
        if not marker.exists():
            return None
        return self.checkpoint_path(json.loads(marker.read_text())["step"])

    def mark_best(self, step: int, metric=None) -> None:
        """Record ``step``'s checkpoint as the best one (by ``metric``)."""
        self.ensure()
        (self.checkpoint_dir / "best.json").write_text(
            json.dumps({"step": step, "metric": metric}, default=str)
        )

    def resolve_checkpoint(self, which: str | int = "latest") -> pathlib.Path | None:
        """Resolve ``"latest"`` / ``"best"`` / an explicit step to a checkpoint path."""
        if isinstance(which, int):
            return self.checkpoint_path(which)
        if which == "latest":
            return self.latest_checkpoint()
        if which == "best":
            return self.best_checkpoint()
        raise ValueError(f"Unknown checkpoint selector: {which!r}")

    def capture_git(self, repo_root: str | pathlib.Path = None) -> dict:
        """Record git provenance in meta; dump a patch only when the tree is dirty."""
        probe = GitProbe(repo_root or pathlib.Path.cwd())
        info = probe.provenance()
        if info.get("available"):
            diff = probe.diff()
            info["dirty"] = bool(diff.strip())
            if info["dirty"]:
                self.ensure()
                (self.run_dir / "uncommitted.patch").write_text(diff)
        self.save_meta(git=info)
        return info

    def write_launch_script(
        self,
        entrypoint: str,
        repo_root: str | pathlib.Path = None,
        launcher: str = None,
    ) -> pathlib.Path:
        """Emit a runnable ``launch.sh`` that reproduces this run via its config.py.

        The script ``cd``s to the repo root, then invokes the entrypoint with
        ``--config <config.py> "$@"`` so CLI flags overlay on reproduction. The
        launcher (e.g. ``torchrun ...``) and git provenance -- which the spec
        structurally cannot capture -- are recorded here.
        """
        self.ensure()
        repo_root = pathlib.Path(repo_root) if repo_root else pathlib.Path.cwd()
        try:
            config_ref = self.config_path.resolve().relative_to(repo_root.resolve())
        except ValueError:
            config_ref = self.config_path.resolve()

        git = self.load_meta().get("git", {})
        lines = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            f"# Run: {self.run_name}",
        ]
        if git.get("available"):
            lines.append(f"# Git: {git.get('sha')} ({git.get('branch')})")
            if git.get("dirty"):
                lines += [
                    "# Working tree was DIRTY. For an exact reproduction:",
                    f"#   git checkout {git.get('sha')}",
                    "#   git apply uncommitted.patch",
                ]
        prefix = f"{launcher} " if launcher else ""
        lines += [
            'cd "$(git rev-parse --show-toplevel)"',
            f'{prefix}python {entrypoint} --config {config_ref} "$@"',
            "",
        ]
        path = self.run_dir / "launch.sh"
        path.write_text("\n".join(lines))
        path.chmod(0o755)
        return path

    @property
    def meta_path(self) -> pathlib.Path:
        return self.run_dir / "meta.json"

    def load_meta(self) -> dict:
        """Return the run's metadata, or an empty dict when none has been written."""
        if not self.meta_path.exists():
            return {}
        return json.loads(self.meta_path.read_text())

    def save_meta(self, **fields) -> pathlib.Path:
        """Merge ``fields`` into ``meta.json`` (existing keys are preserved)."""
        self.ensure()
        meta = self.load_meta()
        meta.update(fields)
        self.meta_path.write_text(json.dumps(meta, indent=2, default=str))
        return self.meta_path

    def __repr__(self):
        return f"Workspace(run_dir={self.run_dir})"

    @property
    def project_dir(self):
        return pathlib.Path(self.root / self.project)

    @property
    def run_dir(self):
        return self.root / self.project / self.run_name

    @property
    def checkpoint_dir(self):
        return self.run_dir / "checkpoints"

    def mkdir(self, path=None):
        if path is None:
            pathlib.Path.mkdir(self.run_dir, parents=True, exist_ok=True)
        else:
            pathlib.Path.mkdir(path, parents=True, exist_ok=True)

    def rm(self, path=None):
        if path is None:
            if self.run_dir.exists():
                for item in self.run_dir.iterdir():
                    if item.is_file():
                        item.unlink()
                    else:
                        self.rm(item)
                self.run_dir.rmdir()
        else:
            if path.exists():
                for item in path.iterdir():
                    if item.is_file():
                        item.unlink()
                    else:
                        self.rm(item)
                path.rmdir()

    @classmethod
    def open(cls, run_dir: str | pathlib.Path) -> Workspace:
        """Reopen an existing run directory (for resume / play), no new run created."""
        run_dir = pathlib.Path(run_dir)
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
        return cls(
            root=str(run_dir.parent.parent),
            project=run_dir.parent.name,
            run_name=run_dir.name,
            timestamp=False,
        )
