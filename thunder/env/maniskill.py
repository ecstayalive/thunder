from dataclasses import dataclass

from thunder.utils import ArgParser

from .loader import EnvLoaderSpec, ThunderEnvWrapper, register_loader


@dataclass(kw_only=True)
class ManiSkillLoaderSpec(EnvLoaderSpec):
    framework: str = "maniskill"
    task: str = "PickCube-v1"
    num_envs: int = 1
    obs_mode: str = "state"
    control_mode: str = "pd_ee_delta_pose"
    render_mode: str = "human"


@register_loader("maniskill")
def load_maniskill(spec: EnvLoaderSpec | ManiSkillLoaderSpec) -> ThunderEnvWrapper:
    import gymnasium as gym
    import mani_skill.envs

    spec = ArgParser.transform(spec, ManiSkillLoaderSpec)
    return gym.make(spec.task, num_envs=spec.num_envs, obs_mode=spec.obs_mode)
