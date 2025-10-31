from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch

__all__ = ["Buffer"]


class Buffer:
    """A class to store and manage transitions and batches of data.
    This class is the most basic `buffer` class, which manages
    scalar `reward` or vector `reward`, `obs` and other variables.
    """

    @dataclass(slots=True)
    class Transition:
        """The Transition class represents a transition in a reinforcement
        learning buffer.
        Args:
            actor_obs: Observation for the actor. :shape: `[num_envs, obs_features...]`
            next_obs: Observation after agent act in environment, the
                value is for the actor. :shape: `[num_envs, obs_features...]`
            actions: The output of the actor. :shape: `[num_envs, action_dimension]`
            reward: The reward obtained by the agent. :shape: `[num_envs, 1]` for
                scalar reward or `[num_envs, reward_dimension]` for multi-object reward.
            dones: Whether the environment is done. :shape: `[num_envs,]`
            timeouts: Whether the agent has been running in the environment for
                longer than the specified time. :shape: `[num_envs,]`
            critic_obs: Observation for the critic, default is `None`.
                :shape:`[num_envs, critic_obs_features...]`
            next_critic_obs: Observation after agent act in environment, the value
                is for critic. :shape:`[num_envs, critic_obs_features...]`
        """

        actor_obs: Any = None
        next_obs: Any = None
        actions: Any = None
        rewards: Any = None
        dones: Any = None
        timeouts: Any = None
        _critic_obs: Any = None
        _next_critic_obs: Any = None

        @property
        def critic_obs(self):
            return self._critic_obs if self._critic_obs is not None else self.actor_obs

        @critic_obs.setter
        def critic_obs(self, value):
            self._critic_obs = value

        @property
        def next_critic_obs(self):
            return self._next_critic_obs if self._next_critic_obs is not None else self.next_obs

        @next_critic_obs.setter
        def next_critic_obs(self, value):
            self._next_critic_obs = value

    @dataclass(slots=True)
    class Batch:
        actor_obs: Any = None
        next_obs: Any = None
        actions: Any = None
        rewards: Any = None
        dones: Any = None
        critic_obs: Any = None
        next_critic_obs: Any = None

    def __init__(self, length: int, device: torch.device = None):
        """Initializes the Buffer object with a specified length and device.
        Args:
            length: The maximum number of transitions to store.
            device: The device to place the tensors on.
        """
        self.device = device
        self.length = length
        self.num_envs: Optional[int] = None
        self._initialized: bool = False

        # Core
        self.actor_obs: Optional[torch.Tensor] = None
        self.next_obs: Optional[torch.Tensor] = None
        self.rewards: Optional[torch.Tensor] = None
        self.actions: Optional[torch.Tensor] = None
        self.dones: Optional[torch.Tensor] = None
        self.timeouts: Optional[torch.Tensor] = None
        self._critic_obs: Optional[torch.Tensor] = None
        self._next_critic_obs: Optional[torch.Tensor] = None

        self.step = 0

    @property
    def critic_obs(self):
        return self._critic_obs if self._critic_obs is not None else self.actor_obs

    @critic_obs.setter
    def critic_obs(self, value):
        self._critic_obs = value

    @property
    def next_critic_obs(self):
        return self._next_critic_obs if self._next_critic_obs is not None else self.next_obs

    @next_critic_obs.setter
    def next_critic_obs(self, value):
        self._next_critic_obs = value

    def zeros_th(self, *shape, **kwargs):
        return torch.zeros(shape, **kwargs, device=self.device)

    def as_th(self, data):
        return torch.as_tensor(data, device=self.device)

    def add_transition(self, t: Transition):
        if self.step >= self.length:
            raise RuntimeError("Buffer overflow")
        if not self._initialized:
            self._lazy_init(t)
            self._initialized = True
        self._add(t)
        self.step += 1

    def _lazy_init(self, t: Transition):
        self.num_envs = t.actor_obs.shape[0]
        self.actor_obs = self.zeros_th(self.length, *t.actor_obs.shape)
        self.next_obs = torch.zeros_like(self.actor_obs)
        self.rewards = self.zeros_th(self.length, *t.rewards.shape)
        self.actions = self.zeros_th(self.length, *t.actions.shape)
        self.dones = self.zeros_th(self.length, self.num_envs, 1, dtype=torch.bool)
        self.timeouts = self.zeros_th(self.length, self.num_envs, 1, dtype=torch.bool)

        if t._critic_obs is not None:
            self._critic_obs = self.zeros_th(self.length, *t._critic_obs.shape)
            self._next_critic_obs = torch.zeros_like(self._critic_obs)

    def _add(self, t: Transition):
        self.actor_obs[self.step] = self.as_th(t.actor_obs)
        self.next_obs[self.step] = self.as_th(t.next_obs)
        self.actions[self.step] = self.as_th(t.actions)
        self.rewards[self.step] = self.as_th(t.rewards)
        self.dones[self.step] = self.as_th(t.dones).unsqueeze_(-1)
        self.timeouts[self.step] = self.as_th(t.timeouts).unsqueeze_(-1)

        if t._critic_obs is not None:
            self._critic_obs[self.step] = self.as_th(t._critic_obs)
            self._next_critic_obs[self.step] = self.as_th(t._next_critic_obs)

    def save_hidden_states(
        self,
        hidden: Optional[torch.Tensor | Tuple[torch.Tensor, torch.Tensor]],
        buf_name: str,
    ):
        """Saves hidden states in the buffer.
        Args:
            hidden: The hidden states to save.
            buf_name: The name of the buffer to save the hidden states in.
        """
        if hidden is not None:
            if isinstance(hidden, torch.Tensor):
                hidden = [hidden]
            buffer = getattr(self, buf_name)
            # lstm, hidden = (h, c)
            if buffer is None:
                buffer = [self.zeros_th(self.length, *h.shape) for h in hidden]
                setattr(self, buf_name, buffer)
            for i in range(len(hidden)):
                buffer[i][self.step] = hidden[i]
