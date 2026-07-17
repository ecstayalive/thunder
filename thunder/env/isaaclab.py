from dataclasses import asdict, dataclass
from typing import List, Literal

from isaaclab.utils.modifiers import ModifierCfg
from isaaclab.utils.string import string_to_callable

from thunder.utils import ArgParser

from .env import EnvLoaderSpec, ThunderEnv, register_loader


def _resolve_modifier_funcs(env_cfg):
    # FIXME: isaaclab v3.0.0-beta has a parsing error for custom Modifier.
    # FIXME: This is a temporary fix and will be removed later
    if not hasattr(env_cfg, "observations"):
        return
    for group_cfg in env_cfg.observations.__dict__.values():
        if group_cfg is None or not hasattr(group_cfg, "__dict__"):
            continue
        for term_cfg in group_cfg.__dict__.values():
            modifiers = getattr(term_cfg, "modifiers", None)
            if modifiers is None:
                continue
            for mod_cfg in modifiers:
                if isinstance(mod_cfg, ModifierCfg) and isinstance(mod_cfg.func, str):
                    mod_cfg.func = string_to_callable(str(mod_cfg.func))


@dataclass(kw_only=True)
class IsaacLabLoaderSpec(EnvLoaderSpec):
    """Specification for loading an IsaacLab environment into Thunder.

    This dataclass defines the configuration required to launch an IsaacLab
    task and wrap it as a :class:`ThunderEnv`. It controls task selection,
    simulation device, rendering, distributed execution, animation recording
    and visualizer integration.

    Attributes:
        framework: Framework identifier
        task: Registered IsaacLab task / gym environment.
        num_envs: Number of parallel environments to spawn.
        headless: If ``True``, run the simulator without a GUI.
        device: Compute device string (e.g. ``"cuda:0"`` or ``"cpu"``).
        livestream: Livestream mode code. ``-1`` disables livestreaming;
            other values enable the corresponding streaming backend.
        enable_cameras: If ``True``, enable camera / tiled-render
            extensions.
        xr: If ``True``, enable XR / VR extensions.
        verbose: If ``True``, enable verbose Kit / Omniverse logging.
        info: If ``True``, show Kit / extension info-level output.
        experience: Path or name of a custom Kit ``.kit`` experience to
            load.
        rendering_mode: Rendering quality preset passed to the simulator.
        kit_args: Extra command-line arguments forwarded to Kit.
        disable_fabric: If ``True``, disable Fabric and fall back to direct
            USD composition.
        distributed: If ``True``, the process is part of a distributed run
            and the device is derived from the local rank.
        cpu: If ``True``, force CPU simulation regardless of ``device``.
        anim_recording_enabled: If ``True``, record the simulation as an
            animation during the configured time window.
        anim_recording_start_time: Simulation time (in seconds) at which
            to start the animation recording.
        anim_recording_stop_time: Simulation time (in seconds) at which
            to stop the animation recording.
        visualizer: List of visualizer backends to attach to the env
            (e.g. ``"rerun"``, ``"viser"``).
    """

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
    visualizer: List[Literal["rerun", "newton", "viser", "kit", "none"]] = None

    def parse_env_cfg(self):
        from isaaclab.envs import DirectRLEnvCfg, ManagerBasedEnvCfg
        from isaaclab_tasks.utils import resolve_task_config

        cfg, _ = resolve_task_config(self.task, agent_cfg_entry_point="")
        cfg: DirectRLEnvCfg | ManagerBasedEnvCfg
        cfg.sim.device = self.device
        cfg.sim.use_fabric = not self.disable_fabric
        if self.num_envs is not None:
            cfg.scene.num_envs = self.num_envs
        return cfg


@register_loader("isaaclab")
def load_isaaclab(spec: EnvLoaderSpec | IsaacLabLoaderSpec) -> ThunderEnv:
    """ """
    from isaaclab.app import AppLauncher

    spec = ArgParser.transform(spec, IsaacLabLoaderSpec)
    app_launcher = AppLauncher(**asdict(spec))
    import omni.kit.app

    ext_manager = omni.kit.app.get_app().get_extension_manager()
    ext_manager.set_extension_enabled_immediate("isaacsim.asset.importer.mjcf", True)

    import gymnasium
    import isaaclab_tasks
    from isaaclab.utils.timer import Timer

    Timer.enable = False
    Timer.enable_display_output = False

    cfg = spec.parse_env_cfg()
    if spec.distributed:
        cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    _resolve_modifier_funcs(cfg)
    env = gymnasium.make(spec.task, cfg=cfg)
    return ThunderEnv(env)
