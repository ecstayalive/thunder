from typing import Any, Dict, List, Literal, Tuple

import gymnasium as gym

from thunder.utils import ArgsOpt

from .loader import EnvLoaderSpec, ThunderEnvWrapper, register_loader


class IsaacLabLoaderSpec(EnvLoaderSpec):
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


@register_loader("isaaclab")
def load_isaaclab(spec: EnvLoaderSpec | IsaacLabLoaderSpec) -> ThunderEnvWrapper:
    """ """
    from isaaclab.app import AppLauncher

    spec = spec.to(IsaacLabLoaderSpec, final=True)
    app_launcher = AppLauncher(spec.to_namespace())
    import gymnasium
    import isaaclab_tasks
    import isaaclab_tasks_experimental
    from isaaclab.utils.timer import Timer
    from isaaclab_tasks.utils import parse_env_cfg

    Timer.enable = False
    Timer.enable_display_output = False

    cfg = parse_env_cfg(
        spec.task, device=spec.device, num_envs=spec.num_envs, use_fabric=not spec.disable_fabric
    )
    if spec.distributed:
        cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    env = gymnasium.make(spec.task, cfg=cfg)
    return ThunderEnvWrapper(env)
