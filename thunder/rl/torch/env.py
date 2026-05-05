from thunder.env import ObservationWrapper, ThunderEnv, ObservationType, WrapperObsType
from thunder.nn.torch import DictRunningNorm1d


class ObsNormalizationWrapper(ObservationWrapper):
    """Normalizes observations using running mean and variance."""

    def __init__(self, env: ThunderEnv, normalizer: DictRunningNorm1d, update: bool = True):
        super().__init__(env)
        self.normalizer = normalizer
        self.update = update

    def observation(self, observation: ObservationType) -> WrapperObsType:
        if self.update:
            self.normalizer.update(observation)
        return self.normalizer(observation)

    def train(self, mode: bool = True):
        self.update = mode
        return self

    def eval(self):
        return self.train(False)
