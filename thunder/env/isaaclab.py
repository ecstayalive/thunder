import argparse
import atexit
import enum
import sys
from typing import Any, Dict, List, Literal, Tuple

import gymnasium

from thunder.utils import ArgsOpt

from .interface import EnvSpec, EnvWrapper
from .loader import register_loader


class IsaacLabEnvSpec(EnvSpec):
    """ """

    framework: str = "isaaclab"
    task: str = ArgsOpt(default="Isaac-Cartpole-Direct-v0")
    num_envs: int = 1024
    headless: bool = False
    device: str = ArgsOpt(default="cuda:0")
    livestream: int = -1
    enable_cameras: bool = False
    xr: bool = False
    verbose: bool = False
    info: bool = False
    experience: str = ArgsOpt(default="")
    rendering_mode: Literal["performance", "balanced", "quality"] = "balanced"
    kit_args: str = ArgsOpt(default="")
    disable_fabric: bool = False
    distributed: bool = False
    cpu: bool = False
    anim_recording_enabled: bool = False
    anim_recording_start_time: float = 0.0
    anim_recording_stop_time: float = 10.0
    visualizer: List[str] = None


class IsaacLabAdapter(EnvWrapper):
    """ """

    def __init__(self, env: gymnasium.Env):
        self.env = env

    def reset(self) -> Tuple[Any, Dict]:
        return self.env.reset()

    def step(self, action: Any) -> Tuple[Any, Any, Any, Any, Dict]:
        return self.env.step(action)

    @property
    def unwrapped(self) -> Any:
        return self.env.unwrapped


@register_loader("isaaclab")
def load_isaaclab(spec: EnvSpec | IsaacLabEnvSpec) -> EnvWrapper:
    """ """
    from isaaclab.app import AppLauncher

    parser = IsaacLabEnvSpec.parser()
    parser.set_defaults(**spec.to_dict(recurse=False))
    spec = IsaacLabEnvSpec.parse(spec._unknown_args, parser=parser, final=True)
    app_launcher = AppLauncher(spec.to_namespace())
    import gymnasium
    import isaaclab_tasks
    from isaaclab.utils.timer import Timer
    from isaaclab_tasks.utils import parse_env_cfg

    Timer.enable = False
    Timer.enable_display_output = False

    cfg = parse_env_cfg(
        spec.task,
        device=spec.device,
        num_envs=spec.num_envs,
        use_fabric=not spec.disable_fabric,
    )
    if spec.distributed:
        cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    env = gymnasium.make(spec.task, cfg=cfg)
    # return IsaacLabAdapter(env)
    return env
