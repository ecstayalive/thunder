from .interface import EnvSpec, EnvWrapper
from .loader import register_loader


class ManiSkillSpec(EnvSpec):
    framework: str = "maniskill"
    task: str = "default"
    num_envs: int = 1


@register_loader("maniskill")
def load_maniskill(spec: EnvSpec | ManiSkillSpec) -> EnvWrapper: ...
