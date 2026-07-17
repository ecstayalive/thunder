from __future__ import annotations

import pathlib
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import torch

from thunder.env.dexlab import DexLabLoaderSpec
from thunder.rl.torch.agent import PpoAgentSpec

from .arguments import ArgOpt, ArgParser
from .workspace import ConfigCodec, GitProbe, SpecDiff, Workspace

if TYPE_CHECKING:
    from thunder.core import DistributedSpec


@dataclass
class ExperimentSpec:
    """Framework defaults for a training run.

    Task-specific tuning (env task, agent architecture/hyperparameters) lives in
    the env package alongside the other framework configs -- e.g. a DexLab task's
    ``agents/thunder_cfg.py`` subclasses this and is resolved from the gym
    registry via ``--task`` (mirroring ``rsl_rl_cfg_entry_point``). Edit there,
    not here.
    """

    seed: int = 0
    device: str = "cuda:0"
    iteration: int = 42000
    save_interval: int = 500
    # logging / checkpointing are independent switches
    log: bool = True
    save: bool = True
    # workspace placement
    root: str = "./logs"
    project: str | None = None
    run_name: str | None = None
    # warm-start: a run directory (uses its latest checkpoint) or a checkpoint file
    resume: str | None = None
    # reproduce (not persisted): seed the whole spec from a prior run's config.py.
    # The task is selected via env.task, which auto-loads its registered thunder cfg.
    from_config: str | None = ArgOpt(
        default=None,
        serialize=False,
        help="reproduce: seed the spec from a prior run's config.py",
    )
    env: DexLabLoaderSpec = ArgOpt(factory=DexLabLoaderSpec)
    agent: PpoAgentSpec = ArgOpt(factory=PpoAgentSpec)

    @property
    def resolved_project(self) -> str:
        return self.project or f"{self.agent.name}@{self.env.task}"


@dataclass
class PlaySpec(ExperimentSpec):
    """Configuration for replaying a trained policy."""

    run: str | None = ArgOpt(
        default=None, help="run dir to play; default: latest under root"
    )
    checkpoint: str = ArgOpt(default="latest", help="latest | best | <step>")
    iteration: int = 10000
    explore: bool = False
    speed: float = 1.0

    def locate_run(self) -> pathlib.Path:
        if self.run:
            return pathlib.Path(self.run)
        runs = [p.parent for p in pathlib.Path(self.root).glob("*/*/config.py")]
        if not runs:
            raise FileNotFoundError(f"No runs with a config.py under {self.root}")
        return max(runs, key=lambda p: p.stat().st_mtime)


class Experiment:
    """Lifecycle manager for a training run, layered over a :class:`Workspace`."""

    def __init__(
        self,
        spec: ExperimentSpec,
        workspace: Workspace | None,
        is_main: bool = True,
    ):
        self.spec = spec
        self.workspace = workspace
        self.is_main = is_main

    @classmethod
    def parse(
        cls, spec_cls=ExperimentSpec, argv: list[str] | None = None
    ) -> ExperimentSpec:
        """Parse a spec from the CLI (Must run before the Isaac app launches)."""
        from_cli = argv is None
        cls._consumed_argv = list(sys.argv[1:] if from_cli else argv)
        parser = ArgParser(spec_cls)
        prelim = parser.parse(None if from_cli else argv)
        if from_cli:
            stray = [a for a in parser.unknown_args if a.startswith("-")]
            if stray:
                parser.parser.error(f"unrecognized arguments: {' '.join(stray)}")
        if not prelim.from_config:
            return prelim
        base = ConfigCodec.loads(prelim.from_config)
        return ArgParser(type(base), default_instance=base, overlay=True).parse(
            cls._consumed_argv
        )

    @classmethod
    def apply_task_cfg(
        cls, spec: ExperimentSpec, argv: list[str] | None = None
    ) -> ExperimentSpec:
        """Merge the task's registered thunder cfg into ``spec`` (call post-app).

        Resolves the ``thunder_cfg_entry_point`` for ``spec.env.task`` (importing
        it pulls USD, so this must run after ``AppLauncher``), overlays the CLI on
        top, and returns a plain ``ExperimentSpec`` so the persisted ``config.py``
        stays import-light and reproducible pre-app. No-op when reproducing or
        when the task has no thunder cfg.
        """
        if spec.from_config:
            return spec
        cfg = cls._load_task_cfg(spec.env.task)
        if cfg is None:
            return spec
        # The env is already built from `spec.env`; keep it and merge agent/training.
        cfg.env = spec.env
        # sys.argv was stripped at parse(); re-apply the stashed CLI overrides.
        if argv is None:
            argv = getattr(cls, "_consumed_argv", None) or []
        merged = ArgParser(type(cfg), default_instance=cfg, overlay=True).parse(argv)
        return cls._to_base(merged)

    @staticmethod
    def _to_base(spec: ExperimentSpec) -> ExperimentSpec:
        """Copy a thunder-cfg subclass instance into a plain ``ExperimentSpec``.

        Persisting the base class keeps ``config.py`` free of the DexLab task
        package (which imports USD), so it reloads before the app launches.
        """
        import dataclasses

        if type(spec) is ExperimentSpec:
            return spec
        return ExperimentSpec(
            **{
                f.name: getattr(spec, f.name)
                for f in dataclasses.fields(ExperimentSpec)
            }
        )

    @staticmethod
    def _load_task_cfg(task_id: str) -> ExperimentSpec | None:
        """The thunder cfg registered for a gym task, or None if there is none.

        Resolves ``thunder_cfg_entry_point`` from the gym registry and injects
        ``task_id`` as the env task, so one cfg can serve several tasks (mirroring
        a shared rsl_rl runner cfg).
        """
        import importlib

        import gymnasium as gym

        try:
            entry = gym.spec(task_id).kwargs.get("thunder_cfg_entry_point")
        except Exception:
            return None
        if not entry:
            return None
        module_name, attr = entry.split(":")
        cfg_cls = getattr(importlib.import_module(module_name), attr)
        cfg: ExperimentSpec = cfg_cls()
        cfg.env.task = task_id
        return cfg

    @staticmethod
    def bind(spec: ExperimentSpec, dist: DistributedSpec = None) -> None:
        """Wire device/seed/distributed from the runtime onto ``spec`` (in place).

        Idempotent: called before ``make_env`` (so the env builds on the right
        device) and again inside :meth:`start` after the task cfg is merged.
        """
        torch.manual_seed(spec.seed)
        if dist is not None and dist.enabled:
            spec.device = dist.device
            spec.env.seed = spec.seed + dist.rank
            spec.env.distributed = True
        spec.agent.device = spec.device
        spec.env.device = spec.device

    @classmethod
    def start(cls, spec: ExperimentSpec, dist: DistributedSpec = None) -> Experiment:
        """Open a reproducible run: bind the runtime, resolve warm-start, persist.

        The main process snapshots config / git / meta / launch.sh; persistence
        is skipped when both ``log`` and ``save`` are off, so throwaway debug runs
        leave no files behind.
        """
        cls.bind(spec, dist)
        is_main = dist is None or dist.is_main
        # Every rank resolves the same weights; only main warns about spec drift.
        spec.agent.resume = cls.resolve_resume(spec, verbose=is_main)

        # Only main owns the run directory (each rank would draw a different name).
        workspace = None
        if is_main:
            workspace = Workspace.create(
                spec.root, spec.resolved_project, spec.run_name, timestamp=True
            )
            if spec.log or spec.save:
                launcher = (
                    f"torchrun --nproc_per_node={dist.world_size}"
                    if dist is not None and dist.enabled
                    else None
                )
                repo_root = GitProbe(pathlib.Path.cwd()).toplevel()
                workspace.save_config(spec)
                workspace.capture_git(repo_root)
                workspace.save_meta(
                    seed=spec.seed,
                    device=spec.device,
                    argv=list(sys.argv),
                    created_at=datetime.now().isoformat(timespec="seconds"),
                    status="running",
                )
                workspace.write_launch_script(
                    cls._entrypoint(repo_root), repo_root, launcher
                )
        return cls(spec, workspace, is_main)

    @staticmethod
    def _entrypoint(repo_root: pathlib.Path | None) -> str:
        """The running script's path, relative to the repo root when possible."""
        script = pathlib.Path(sys.argv[0]).resolve()
        if repo_root:
            try:
                return str(script.relative_to(pathlib.Path(repo_root).resolve()))
            except ValueError:
                pass
        return script.name

    @classmethod
    def resolve_resume(cls, spec: ExperimentSpec, verbose: bool = True) -> str | None:
        """Resolve ``spec.resume`` to a checkpoint path, warning on spec drift.
        Accepts a run directory (loads its latest checkpoint and diffs the saved
        config against the current spec) or a direct checkpoint file. Stateless
        so every rank can resolve the same weight path; ``verbose`` gates the
        warning to a single process under distributed training.
        """
        if not spec.resume:
            return None
        source = pathlib.Path(spec.resume)
        if source.is_dir():
            source_ws = Workspace.open(source)
            if verbose:
                cls._warn_spec_drift(source_ws, spec)
            checkpoint = source_ws.latest_checkpoint()
            if checkpoint is None:
                raise FileNotFoundError(f"No checkpoint found in run: {source}")
            return str(checkpoint)
        if not source.exists():
            raise FileNotFoundError(f"Resume target does not exist: {source}")
        return str(source)

    @staticmethod
    def _warn_spec_drift(source_ws: Workspace, spec: ExperimentSpec) -> None:
        try:
            old_spec = source_ws.load_config(ExperimentSpec)
        except Exception:
            return
        diff = {
            f"env.{path}": delta
            for path, delta in SpecDiff.compare(old_spec.env, spec.env).items()
        }
        diff.update(
            {
                f"agent.{path}": delta
                for path, delta in SpecDiff.compare(old_spec.agent, spec.agent).items()
            }
        )
        if not diff:
            return
        print(
            f"\033[93m[Experiment] Warm-starting from {source_ws.run_dir} but the "
            f"spec changed in {len(diff)} field(s):\033[0m"
        )
        for path, (old, new) in sorted(diff.items()):
            print(f"  - {path}: {old!r} -> {new!r}")
        print(
            "\033[93m  Architecture changes may cause load_state_dict to fail.\033[0m"
        )

    def finish(self, status: str = "finished") -> None:
        if self.workspace is not None and (self.spec.log or self.spec.save):
            self.workspace.save_meta(status=status)
