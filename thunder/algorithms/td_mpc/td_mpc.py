import itertools

import torch
import torch.optim as optim
from torch import nn

from thunder.nn import freeze
from thunder.rl.actor import DecActor
from thunder.rl.critic import MultiQNet
from thunder.rl.utils import gaussian_kl_divergence, unpad_trajectory

from .buffer import TdMpcBuffer

GIGA = 2**30


class TdMpc:
    def __init__(
        self,
        actor: DecActor,
        model: nn.Module,
        reward_model: nn.Module,
        q_func: MultiQNet,
        num_envs,
        num_collects,
        num_learning_epochs,
    ):
        """ """
        ...

    def init_storage(self):
        """"""
        ...

    @torch.inference_mode()
    def act(self, actor_obs, critic_obs):
        """
        Args:
            actor_obs: Incomplete observations with noise and delays
            critic_obs: Delay-free and noise-free, representing as much information as possible.
        """
        ...

    @torch.inference_mode()
    def step(self, next_obs, rewards, dones, timeouts, next_critic_obs=None):
        """"""
        ...

    def clear_hidden_states(self, dones):
        """"""
        ...

    def _warmup_step(self):
        """"""
        ...

    def _train_step(self, epoch=0) -> dict:
        """"""
        ...

    def regularization_anneal(self):
        """"""
        ...
