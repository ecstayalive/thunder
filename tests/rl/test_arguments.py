from dataclasses import dataclass, field
import enum
from typing import List, Literal, Tuple
from thunder.utils.arguments import ArgParser

@dataclass
class Config:
    use_cuda: bool = True
    use_compile: bool = False
    opt_level: int = 3

def test_boolean_action_default_true():
    # Test BooleanAction when default is True
    # 1. No CLI args (should keep default True)
    parser = ArgParser(Config)
    cfg = parser.parse([])
    assert cfg.use_cuda is True

    # 2. Toggle to False using --no-use-cuda
    parser = ArgParser(Config)
    cfg = parser.parse(["--no-use-cuda"])
    assert cfg.use_cuda is False

    # 3. Explicitly set to True using --use-cuda
    parser = ArgParser(Config)
    cfg = parser.parse(["--use-cuda"])
    assert cfg.use_cuda is True

def test_boolean_action_default_false():
    # Test BooleanAction when default is False
    # 1. No CLI args (should keep default False)
    parser = ArgParser(Config)
    cfg = parser.parse([])
    assert cfg.use_compile is False

    # 2. Toggle to True using --use-compile
    parser = ArgParser(Config)
    cfg = parser.parse(["--use-compile"])
    assert cfg.use_compile is True

    # 3. Explicitly set to False using --no-use-compile
    parser = ArgParser(Config)
    cfg = parser.parse(["--no-use-compile"])
    assert cfg.use_compile is False

@dataclass
class TupleConfig:
    coords: Tuple[int, float] = (0, 0.0)
    sizes: Tuple[int, ...] = (10, 20)
    info: Tuple[str, bool, float] = ("default", False, 1.0)

def test_tuple_action_parsing():
    # 1. No CLI args (should keep defaults)
    parser = ArgParser(TupleConfig)
    cfg = parser.parse([])
    assert cfg.coords == (0, 0.0)
    assert cfg.sizes == (10, 20)
    assert cfg.info == ("default", False, 1.0)

    # 2. Parse fixed-size heterogeneous tuple coords
    parser = ArgParser(TupleConfig)
    cfg = parser.parse(["--coords", "1", "2.5"])
    assert cfg.coords == (1, 2.5)
    assert isinstance(cfg.coords[0], int)
    assert isinstance(cfg.coords[1], float)

    # 3. Parse arbitrary-size homogeneous tuple sizes
    parser = ArgParser(TupleConfig)
    cfg = parser.parse(["--sizes", "1", "2", "3"])
    assert cfg.sizes == (1, 2, 3)

    # 4. Parse three-element heterogeneous tuple info
    parser = ArgParser(TupleConfig)
    cfg = parser.parse(["--info", "hello", "True", "2.5"])
    assert cfg.info == ("hello", True, 2.5)
    assert isinstance(cfg.info[1], bool)


class Mode(enum.Enum):
    fast = "fast"
    slow = "slow"


@dataclass
class NestedConfig:
    size: int = 1
    enabled: bool = False


@dataclass
class AdvancedConfig:
    flags: List[bool] = None
    visualizer: List[Literal["rerun", "newton", "none"]] = None
    precision: Literal["fp32", "fp16", "bf16"] = "fp32"
    mode: Mode = Mode.fast
    nested: NestedConfig = field(default_factory=NestedConfig)


def test_ast_parser_handles_container_leaf_types():
    parser = ArgParser(AdvancedConfig)
    cfg = parser.parse(
        [
            "--flags",
            "true",
            "false",
            "1",
            "--visualizer",
            "rerun,newton",
            "--precision",
            "bf16",
            "--mode",
            "slow",
            "--nested.size",
            "7",
            "--nested.enabled",
        ]
    )

    assert cfg.flags == [True, False, True]
    assert cfg.visualizer == ["rerun", "newton"]
    assert cfg.precision == "bf16"
    assert cfg.mode is Mode.slow
    assert cfg.nested.size == 7
    assert cfg.nested.enabled is True


@dataclass
class SourceConfig:
    framework: str = "base"
    task: str = "task"
    num_envs: int = 1


@dataclass
class TargetConfig(SourceConfig):
    headless: bool = False
    device: str = "cuda"


def test_transform_preserves_source_fields_and_binds_target_fields():
    source = SourceConfig(framework="gym", task="CartPole-v1", num_envs=8)

    cfg = ArgParser.transform(
        source,
        TargetConfig,
        ["--headless", "--device", "cpu"],
    )

    assert cfg.framework == "gym"
    assert cfg.task == "CartPole-v1"
    assert cfg.num_envs == 8
    assert cfg.headless is True
    assert cfg.device == "cpu"


@dataclass
class _OverlayCfg:
    seed: int = 0
    iteration: int = 10


def test_overlay_uses_instance_defaults_but_allows_cli_override():
    base = _OverlayCfg(seed=5, iteration=123)

    # No CLI flags -> every field falls back to the base instance.
    got = ArgParser(_OverlayCfg, default_instance=base, overlay=True).parse([])
    assert (got.seed, got.iteration) == (5, 123)

    # CLI overrides one field; the other is still inherited from the base.
    got = ArgParser(_OverlayCfg, default_instance=base, overlay=True).parse(["--seed", "9"])
    assert (got.seed, got.iteration) == (9, 123)
