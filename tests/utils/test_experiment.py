from dataclasses import dataclass

import gymnasium as gym

from thunder.utils.experiment import Experiment, ExperimentSpec
from thunder.utils.workspace import Workspace


@dataclass
class _DummyThunderCfg(ExperimentSpec):
    """Stands in for a task's agents/thunder_cfg.py subclass."""

    seed: int = 3


def _register_dummy(task_id):
    if task_id not in gym.registry:
        gym.register(
            task_id,
            entry_point="dummy:NotImported",
            kwargs={"thunder_cfg_entry_point": f"{__name__}:_DummyThunderCfg"},
        )


def test_selectors_are_visible_in_help():
    from thunder.utils import ArgParser

    help_text = ArgParser(ExperimentSpec).parser.format_help()
    assert "--from-config" in help_text  # reproduce selector
    assert "--env.task" in help_text  # task selector (drives cfg loading)


def test_from_config_directive_not_persisted_to_config_py(tmp_path):
    spec = ExperimentSpec(seed=1)
    spec.from_config = "/some/run/config.py"
    ws = Workspace.create(str(tmp_path), "proj", run_name="r1", timestamp=False)
    ws.save_config(spec)

    loaded = ws.load_config(ExperimentSpec)
    assert loaded.from_config is None  # directive excluded from the snapshot


def test_load_task_cfg_resolves_entry_point_to_instance():
    _register_dummy("DummyThunder-Load-v0")
    cfg = Experiment._load_task_cfg("DummyThunder-Load-v0")
    assert isinstance(cfg, _DummyThunderCfg)
    assert cfg.seed == 3


def test_load_task_cfg_injects_task_id_so_one_cfg_serves_many_tasks():
    _register_dummy("DummyThunder-Inject-v0")
    cfg = Experiment._load_task_cfg("DummyThunder-Inject-v0")
    # the gym id is authoritative for which env to build
    assert cfg.env.task == "DummyThunder-Inject-v0"


def test_load_task_cfg_returns_none_without_registered_cfg():
    # unknown task, and a task without a thunder entry point, both -> None
    assert Experiment._load_task_cfg("Totally-Unregistered-v0") is None
    if "PlainEnv-v0" not in gym.registry:
        gym.register("PlainEnv-v0", entry_point="dummy:NotImported")
    assert Experiment._load_task_cfg("PlainEnv-v0") is None


def test_parse_strips_our_flags_and_keeps_hydra_overrides(monkeypatch):
    import sys

    monkeypatch.setattr(
        sys, "argv", ["train.py", "--seed", "5", "env.episode_length=10"]
    )
    spec = Experiment.parse()
    assert spec.seed == 5
    # our flag is consumed; the Hydra-style override is left for IsaacLab
    assert sys.argv == ["train.py", "env.episode_length=10"]


def test_parse_reports_unknown_dashed_flag_with_our_parser(monkeypatch):
    import sys

    import pytest

    monkeypatch.setattr(sys, "argv", ["train.py", "--totally-unknown", "x"])
    # our ArgParser owns the error (SystemExit from argparse), not IsaacLab/Hydra
    with pytest.raises(SystemExit):
        Experiment.parse()


def test_parse_does_not_resolve_task_cfg_preapp():
    # Resolution is deferred to apply_task_cfg (post-app); parse stays USD-free.
    spec = Experiment.parse(argv=["--env.task", "DummyThunder-NoResolve-v0", "--seed", "7"])
    assert type(spec) is ExperimentSpec  # not a thunder-cfg subclass
    assert spec.env.task == "DummyThunder-NoResolve-v0"
    assert spec.seed == 7


def test_apply_task_cfg_merges_cfg_and_returns_base_spec():
    _register_dummy("DummyThunder-Apply-v0")
    spec = ExperimentSpec()
    spec.env.task = "DummyThunder-Apply-v0"

    merged = Experiment.apply_task_cfg(spec, argv=[])
    # plain ExperimentSpec (so config.py stays import-light / reproducible pre-app)
    assert type(merged) is ExperimentSpec
    assert merged.env.task == "DummyThunder-Apply-v0"  # env preserved
    assert merged.seed == 3  # tuning from the registered cfg applied


def test_apply_task_cfg_noop_when_reproducing():
    spec = ExperimentSpec(seed=9)
    spec.from_config = "/x/config.py"
    assert Experiment.apply_task_cfg(spec, argv=[]) is spec


def test_apply_task_cfg_noop_without_registered_cfg():
    spec = ExperimentSpec()
    spec.env.task = "Totally-Unregistered-Apply-v0"
    assert Experiment.apply_task_cfg(spec, argv=[]) is spec


def test_subclass_cfg_roundtrips_through_config_py(tmp_path):
    # A thunder_cfg-style subclass must serialize + reload losslessly, so a
    # `--task` run can be reproduced from its saved config.py.
    spec = _DummyThunderCfg(seed=11)
    ws = Workspace.create(str(tmp_path), "proj", run_name="r1", timestamp=False)
    ws.save_config(spec)

    loaded = ws.load_config(ExperimentSpec)  # base-class load accepts the subclass
    assert isinstance(loaded, _DummyThunderCfg)
    assert loaded.seed == 11


def test_play_help_lists_every_configurable_field():
    from thunder.utils import ArgParser
    from thunder.utils.experiment import PlaySpec

    help_text = ArgParser(PlaySpec).parser.format_help()
    # play-only knobs and the full experiment surface share one parser
    assert "--run" in help_text
    assert "--checkpoint" in help_text
    assert "--env.visualizer" in help_text
    assert "--env.num-envs" in help_text or "--env.num_envs" in help_text
    assert "--agent.lr" in help_text


def test_play_cli_overlays_saved_spec():
    from thunder.utils import ArgParser

    saved = ExperimentSpec(seed=7)
    saved.env.task = "DexLab-Rotate-Cube-V12Max-v0"
    saved.env.num_envs = 4096
    argv = [
        "--checkpoint", "best",  # play-only flag: tolerated, not consumed
        "--env.num-envs", "16",
        "--env.visualizer", "kit",
    ]
    spec = ArgParser(ExperimentSpec, default_instance=saved, overlay=True).parse(argv)
    assert spec.env.num_envs == 16
    assert spec.env.visualizer == ["kit"]
    # fields not on the CLI keep the recording's values
    assert spec.env.task == "DexLab-Rotate-Cube-V12Max-v0"
    assert spec.seed == 7
