from typing import Any, Dict, Tuple

import gymnasium

from .interface import EnvSpec, EnvWrapper
from .loader import register_loader


class GymnasiumSpec(EnvSpec): ...


@register_loader("gym")
def load_gym(spec: EnvSpec | GymnasiumSpec) -> EnvWrapper: ...
