from dataclasses import asdict, dataclass, field

from isaaclab.utils.modifiers import ModifierCfg
from isaaclab.utils.string import string_to_callable

from thunder.utils import ArgParser

from .env import EnvLoaderSpec, ThunderEnv, register_loader
from .isaaclab import IsaacLabLoaderSpec


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
class DexLabLoaderSpec(IsaacLabLoaderSpec):
    """ """

    framework: str = "dexlab"
    task: str = "DexLab-Repose-Cube-V12-v0"
    num_envs: int = 4096


@register_loader("dexlab")
def load_dexlab(spec: EnvLoaderSpec | DexLabLoaderSpec) -> ThunderEnv:
    """ """
    from isaaclab.app import AppLauncher

    spec = ArgParser.transform(spec, DexLabLoaderSpec)
    app_launcher = AppLauncher(**asdict(spec))

    import omni.kit.app

    ext_manager = omni.kit.app.get_app().get_extension_manager()
    ext_manager.set_extension_enabled_immediate("isaacsim.asset.importer.mjcf", True)

    import dexlab.tasks
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
