from .isaaclab import IsaacLabEnvSpec
from .loader import EnvSpec, ThunderWrapper, register_loader


class RayLabSpec(IsaacLabEnvSpec):
    framework: str = "ray_lab"


@register_loader("ray_lab")
def load_ray_lab(spec: EnvSpec | RayLabSpec) -> ThunderWrapper:
    from isaaclab.app import AppLauncher

    parser = IsaacLabEnvSpec.parser()
    parser.set_defaults(**spec.to_dict(recurse=False))
    spec = IsaacLabEnvSpec.parse(spec._unknown_args, parser=parser, final=True)
    app_launcher = AppLauncher(spec.to_namespace())
    import gymnasium
    import isaaclab_tasks
    import isaaclab_tasks_experimental
    from isaaclab.utils.timer import Timer
    from isaaclab_tasks.utils import parse_env_cfg

    # from ray_lab.envs import *

    Timer.enable = False
    Timer.enable_display_output = False

    cfg = parse_env_cfg(
        spec.task, device=spec.device, num_envs=spec.num_envs, use_fabric=not spec.disable_fabric
    )
    if spec.distributed:
        cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    env = gymnasium.make(spec.task, cfg=cfg)
    return env
