import enum
from dataclasses import dataclass, field
from typing import Tuple

from thunder.utils.workspace import SpecDiff, Workspace


class HeadKind(enum.Enum):
    A = "a"
    B = "b"


@dataclass
class HeadSpec:
    pass


@dataclass
class BetaSpec(HeadSpec):
    low: float = -1.0
    high: float = 1.0


@dataclass
class NormalSpec(HeadSpec):
    std: float = 0.2


@dataclass
class ActorSpec:
    obs_keys: Tuple[str, ...] = ("policy",)
    head: HeadSpec = field(default_factory=NormalSpec)
    kind: HeadKind = HeadKind.A


@dataclass
class SampleSpec:
    seed: int = 0
    name: str = "ppo"
    actor: ActorSpec = field(default_factory=ActorSpec)


def test_create_builds_run_dir_structure_without_creating_dirs(tmp_path):
    ws = Workspace.create(str(tmp_path), "ppo@task", run_name="exp1", timestamp=False)

    assert ws.root == tmp_path
    assert ws.project == "ppo@task"
    assert ws.run_name == "exp1"
    assert ws.project_dir == tmp_path / "ppo@task"
    assert ws.run_dir == tmp_path / "ppo@task" / "exp1"
    assert ws.checkpoint_dir == tmp_path / "ppo@task" / "exp1" / "checkpoints"
    # lazy: nothing on disk yet
    assert not ws.run_dir.exists()


def test_create_with_timestamp_prefixes_run_name(tmp_path):
    ws = Workspace.create(str(tmp_path), "proj", run_name="exp1", timestamp=True)
    assert ws.run_name.endswith("-exp1")
    # YYYY_MM_DD_HH_MM_SS- prefix
    assert len(ws.run_name) == len("2026_06_12_00_00_00-") + len("exp1")


def test_create_without_run_name_generates_slug(tmp_path):
    ws = Workspace.create(str(tmp_path), "proj", timestamp=False)
    assert ws.run_name  # non-empty coolname slug
    assert "/" not in ws.run_name


def test_ensure_creates_run_and_checkpoint_dirs(tmp_path):
    ws = Workspace.create(str(tmp_path), "proj", run_name="exp1", timestamp=False)
    ws.ensure()
    assert ws.run_dir.is_dir()
    assert ws.checkpoint_dir.is_dir()
    # idempotent
    ws.ensure()
    assert ws.run_dir.is_dir()


def test_config_codec_skips_non_serializable_fields():
    from thunder.utils.workspace import ConfigCodec

    @dataclass
    class WithDirective:
        seed: int = 0
        # a loader directive that must not be persisted
        task: str = field(default=None, metadata={"serialize": False})

    src = ConfigCodec.dumps(WithDirective(seed=2, task="ignore-me"))
    assert "seed=2" in src
    assert "task=" not in src


def test_save_load_config_roundtrip_preserves_subtype_and_enum(tmp_path):
    spec = SampleSpec(
        seed=7,
        actor=ActorSpec(
            obs_keys=("policy", "critic"),
            head=BetaSpec(low=-2.0, high=2.0),
            kind=HeadKind.B,
        ),
    )
    ws = Workspace.create(str(tmp_path), "proj", run_name="exp1", timestamp=False)
    path = ws.save_config(spec)

    assert path.name == "config.py"
    assert path.exists()

    loaded = ws.load_config()
    # value + concrete polymorphic subtype + enum identity all preserved
    assert loaded == spec
    assert type(loaded.actor.head) is BetaSpec
    assert loaded.actor.kind is HeadKind.B


def _touch_checkpoint(ws, step):
    ws.ensure()
    p = ws.checkpoint_path(step)
    p.write_bytes(b"")
    return p


def test_checkpoint_path_format(tmp_path):
    ws = Workspace.create(str(tmp_path), "proj", run_name="exp1", timestamp=False)
    assert ws.checkpoint_path(500) == ws.checkpoint_dir / "weights_500.pth"


def test_latest_checkpoint_picks_highest_step_numerically(tmp_path):
    ws = Workspace.create(str(tmp_path), "proj", run_name="exp1", timestamp=False)
    _touch_checkpoint(ws, 2)
    _touch_checkpoint(ws, 10)
    _touch_checkpoint(ws, 100)
    # numeric, not lexicographic ('100' > '2' lexically would fail here)
    assert ws.latest_checkpoint() == ws.checkpoint_path(100)


def test_latest_checkpoint_none_when_empty(tmp_path):
    ws = Workspace.create(str(tmp_path), "proj", run_name="exp1", timestamp=False)
    assert ws.latest_checkpoint() is None


def test_resolve_checkpoint_dispatches_latest_int_and_best(tmp_path):
    ws = Workspace.create(str(tmp_path), "proj", run_name="exp1", timestamp=False)
    _touch_checkpoint(ws, 100)
    _touch_checkpoint(ws, 200)
    ws.mark_best(100, metric=0.9)

    assert ws.resolve_checkpoint("latest") == ws.checkpoint_path(200)
    assert ws.resolve_checkpoint(100) == ws.checkpoint_path(100)
    assert ws.resolve_checkpoint("best") == ws.checkpoint_path(100)
    assert ws.best_checkpoint() == ws.checkpoint_path(100)


def test_spec_diff_identical_is_empty():
    a = SampleSpec(seed=1, actor=ActorSpec(head=BetaSpec(low=-1.0, high=1.0)))
    b = SampleSpec(seed=1, actor=ActorSpec(head=BetaSpec(low=-1.0, high=1.0)))
    assert SpecDiff.compare(a, b) == {}


def test_spec_diff_reports_changed_leaf_with_dotted_path():
    a = SampleSpec(seed=1)
    b = SampleSpec(seed=7)
    assert SpecDiff.compare(a, b) == {"seed": (1, 7)}


def test_spec_diff_reports_nested_field_path():
    a = SampleSpec(actor=ActorSpec(head=BetaSpec(low=-1.0, high=1.0)))
    b = SampleSpec(actor=ActorSpec(head=BetaSpec(low=-2.0, high=1.0)))
    assert SpecDiff.compare(a, b) == {"actor.head.low": (-1.0, -2.0)}


def test_spec_diff_reports_polymorphic_subtype_change_as_whole_node():
    a = SampleSpec(actor=ActorSpec(head=NormalSpec(std=0.2)))
    b = SampleSpec(actor=ActorSpec(head=BetaSpec(low=-1.0, high=1.0)))
    diff = SpecDiff.compare(a, b)
    assert set(diff) == {"actor.head"}
    old, new = diff["actor.head"]
    assert type(old) is NormalSpec and type(new) is BetaSpec


def test_spec_diff_reports_tuple_change():
    a = SampleSpec(actor=ActorSpec(obs_keys=("policy",)))
    b = SampleSpec(actor=ActorSpec(obs_keys=("policy", "critic")))
    assert SpecDiff.compare(a, b) == {"actor.obs_keys": (("policy",), ("policy", "critic"))}


def test_write_launch_script_executable_and_reproduces_via_config(tmp_path):
    ws = Workspace.create(str(tmp_path / "logs"), "proj", run_name="exp1", timestamp=False)
    path = ws.write_launch_script("examples/thunder/train.py", repo_root=tmp_path)

    assert path.name == "launch.sh"
    import os

    assert os.access(path, os.X_OK)
    text = path.read_text()
    assert "examples/thunder/train.py" in text
    assert "--config" in text
    assert "logs/proj/exp1/config.py" in text  # relative to repo root
    assert '"$@"' in text  # CLI overlay on reproduction


def test_write_launch_script_includes_launcher_and_dirty_git_hint(tmp_path):
    ws = Workspace.create(str(tmp_path / "logs"), "proj", run_name="exp1", timestamp=False)
    ws.save_meta(
        git={"available": True, "sha": "deadbeef", "branch": "main", "dirty": True}
    )
    path = ws.write_launch_script(
        "train.py", repo_root=tmp_path, launcher="torchrun --nproc_per_node=4"
    )

    text = path.read_text()
    assert "torchrun --nproc_per_node=4" in text
    assert "deadbeef" in text
    assert "uncommitted.patch" in text  # dirty -> patch-apply hint


def _init_repo(path):
    import subprocess

    def run(*args):
        subprocess.run(["git", *args], cwd=path, check=True, capture_output=True)

    run("init")
    run("config", "user.email", "t@t.com")
    run("config", "user.name", "tester")
    (path / "tracked.txt").write_text("hello\n")
    run("add", "-A")
    run("commit", "-m", "init")


def test_capture_git_clean_records_sha_and_writes_no_patch(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    ws = Workspace.create(str(tmp_path / "logs"), "proj", run_name="exp1", timestamp=False)

    ws.capture_git(repo_root=repo)

    git = ws.load_meta()["git"]
    assert len(git["sha"]) == 40
    assert git["dirty"] is False
    assert not (ws.run_dir / "uncommitted.patch").exists()


def test_capture_git_dirty_writes_uncommitted_patch(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    (repo / "tracked.txt").write_text("changed content\n")
    ws = Workspace.create(str(tmp_path / "logs"), "proj", run_name="exp1", timestamp=False)

    ws.capture_git(repo_root=repo)

    git = ws.load_meta()["git"]
    assert git["dirty"] is True
    patch = ws.run_dir / "uncommitted.patch"
    assert patch.exists()
    assert "changed content" in patch.read_text()


def test_capture_git_untracked_flag(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    (repo / "new_file.txt").write_text("brand new\n")
    ws = Workspace.create(str(tmp_path / "logs"), "proj", run_name="exp1", timestamp=False)

    ws.capture_git(repo_root=repo)

    git = ws.load_meta()["git"]
    assert git["has_untracked"] is True


def test_git_probe_toplevel_returns_repo_root(tmp_path):
    from thunder.utils.workspace import GitProbe

    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    sub = repo / "sub" / "dir"
    sub.mkdir(parents=True)

    # toplevel resolves to the repo root even from a nested subdir
    assert GitProbe(sub).toplevel() == repo.resolve()


def test_capture_git_outside_repo_does_not_raise(tmp_path):
    ws = Workspace.create(str(tmp_path / "logs"), "proj", run_name="exp1", timestamp=False)
    ws.capture_git(repo_root=tmp_path / "not_a_repo")  # must not raise
    assert ws.load_meta()["git"]["available"] is False


def test_open_reconstructs_from_existing_run_dir(tmp_path):
    created = Workspace.create(str(tmp_path), "ppo@task", run_name="exp1", timestamp=False)
    created.ensure()

    opened = Workspace.open(created.run_dir)
    assert opened.root == created.root
    assert opened.project == "ppo@task"
    assert opened.run_name == "exp1"
    assert opened.run_dir == created.run_dir
    assert opened.checkpoint_dir == created.checkpoint_dir


def test_open_roundtrips_saved_config_and_checkpoints(tmp_path):
    ws = Workspace.create(str(tmp_path), "proj", run_name="exp1", timestamp=False)
    ws.save_config(SampleSpec(seed=42))
    _touch_checkpoint(ws, 300)

    reopened = Workspace.open(ws.run_dir)
    assert reopened.load_config() == SampleSpec(seed=42)
    assert reopened.latest_checkpoint() == reopened.checkpoint_path(300)


def test_open_missing_dir_raises(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        Workspace.open(tmp_path / "does_not_exist")


def test_load_config_typed_returns_instance_of_expected_class(tmp_path):
    ws = Workspace.create(str(tmp_path), "proj", run_name="exp1", timestamp=False)
    ws.save_config(SampleSpec(seed=3))

    loaded = ws.load_config(SampleSpec)
    assert isinstance(loaded, SampleSpec)
    assert loaded.seed == 3


def test_load_config_typed_rejects_mismatched_class(tmp_path):
    import pytest

    ws = Workspace.create(str(tmp_path), "proj", run_name="exp1", timestamp=False)
    ws.save_config(SampleSpec(seed=3))

    with pytest.raises(TypeError):
        ws.load_config(ActorSpec)  # config holds a SampleSpec, not ActorSpec


def test_load_meta_returns_empty_when_absent(tmp_path):
    ws = Workspace.create(str(tmp_path), "proj", run_name="exp1", timestamp=False)
    assert ws.load_meta() == {}


def test_save_meta_merges_without_clobbering(tmp_path):
    ws = Workspace.create(str(tmp_path), "proj", run_name="exp1", timestamp=False)
    ws.save_meta(seed=7, device="cuda:0")
    ws.save_meta(status="finished", git={"sha": "abc123"})

    meta = ws.load_meta()
    assert meta["seed"] == 7
    assert meta["device"] == "cuda:0"
    assert meta["status"] == "finished"
    assert meta["git"] == {"sha": "abc123"}
