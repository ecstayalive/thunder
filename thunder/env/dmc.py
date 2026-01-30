from dataclasses import dataclass

from thunder.utils import ArgParser

from .loader import EnvLoaderSpec, ObservationWrapper, register_loader


@dataclass
class DmcLoaderSpec:
    """ """

    framework: str = "dmc"
    task: str = ...
    num_envs: int = 1
    headless: bool = False


class DmcAdapter(ObservationWrapper):
    """ """

    def observation(self, observation):
        return {"policy": observation}


@register_loader("dmc")
def load_dmc(spec: DmcLoaderSpec | EnvLoaderSpec) -> DmcAdapter:
    """ """
    spec = ArgParser.transform(spec, DmcLoaderSpec)
    from dm_control import suite

    domain, task = spec.task.split("_", 1)
    env = suite.load(domain_name=domain, task_name=task)
    return DmcAdapter(env)
