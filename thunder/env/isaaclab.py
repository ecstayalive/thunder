from .interface import EnvWrapper


class IsaacLabEnvWrapper(EnvWrapper):
    def __init__(self, env): ...


def isaaclab_loader(task: str, **kwargs):
    """Load an IsaacLab environment based on the specified task.

    Args:
        task (str): The specific IsaacLab task or environment name.

    Returns:
        An instance of the created IsaacLab environment.
    """
    ...
