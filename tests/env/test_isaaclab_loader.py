import torch

from thunder.env.isaaclab import IsaacLabEnvSpec
from thunder.env.loader import make_env

spec = IsaacLabEnvSpec()
env = make_env(spec.parse())
obs = env.reset()
while True:
    with torch.inference_mode():
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        env.step(actions)
