from .interface import EnvSpec, EnvWrapper
from .loader import register_loader


class ManiSkillSpec(EnvSpec): ...


@register_loader("maniskill")
def load_mjlab(spec: EnvSpec | ManiSkillSpec) -> EnvWrapper: ...
