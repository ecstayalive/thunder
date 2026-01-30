from dataclasses import dataclass
from typing import List, Literal

from thunder.utils import ArgParser

from .loader import EnvLoaderSpec, ThunderEnvWrapper, register_loader


@dataclass(kw_only=True)
class IsaacLabLoaderSpec(EnvLoaderSpec):
    """ """

    framework: str = "isaaclab"
    task: str = "Isaac-Cartpole-Direct-v0"
    num_envs: int = 1024
    headless: bool = False
    device: str = "cuda:0"
    livestream: int = -1
    enable_cameras: bool = False
    xr: bool = False
    verbose: bool = False
    info: bool = False
    experience: str = ""
    rendering_mode: Literal["performance", "balanced", "quality"] = "balanced"
    kit_args: str = ""
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

    spec = ArgParser.transform(spec, IsaacLabLoaderSpec)
    app_launcher = AppLauncher(ArgParser.as_dict(spec))
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
