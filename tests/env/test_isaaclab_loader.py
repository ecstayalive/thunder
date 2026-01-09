import torch

from thunder.env.interface import EnvSpec
from thunder.env.loader import make_env

spec = EnvSpec()
env = make_env(spec=spec.parse())
obs = env.reset()
while True:
    with torch.inference_mode():
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        obs, reward, done, timeout, info = env.step(actions)
        print(obs)
