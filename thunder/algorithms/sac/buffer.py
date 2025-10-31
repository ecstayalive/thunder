from typing import Optional

import torch

from thunder.rl import Buffer


class ReplayBuffer(Buffer):
    def __init__(self, buffer_size, device):
        super().__init__(buffer_size, device)

        self.views: Optional[Buffer.Transition] = None
        self.is_full = False

    def _lazy_init(self, t: Buffer.Transition):
        # actual length
        self.length = int(self.length / t.actor_obs.shape[0])
        super()._lazy_init(t)
        self.views = self.Transition(
            self.actor_obs.flatten(0, 1),
            self.next_obs.flatten(0, 1),
            self.actions.flatten(0, 1),
            self.rewards.flatten(0, 1),
            self.dones.flatten(0, 1),
            self.timeouts.flatten(0, 1),
            self.critic_obs_.flatten(0, 1),
            self.next_critic_obs_.flatten(0, 1),
        )

    def add_transition(self, t: Buffer.Transition):
        super().add_transition(t)
        if self.step == self.length:
            self.step -= self.length
            self.is_full = True

    def sampler(self, batch_size, num_batches):
        num_samples = self.num_samples
        for i in range(num_batches):
            indices = torch.randint(num_samples, (batch_size,), device=self.device)
            batch = self.Batch(
                self.views.actor_obs[indices],
                self.views.next_obs[indices],
                self.views.actions[indices],
                self.views.rewards[indices],
                self.views.dones[indices].logical_and(~self.views.timeouts[indices]),
            )
            if self.critic_obs is not None:
                batch.critic_obs = self.views.critic_obs[indices]
                batch.next_critic_obs = self.views.next_critic_obs[indices]
            else:
                batch.critic_obs = batch.actor_obs
                batch.next_critic_obs = batch.next_obs
            yield batch

    @property
    def num_samples(self):
        if self.is_full:
            return self.length * self.num_envs
        if self.step == 0:
            return 0
        return self.step * self.num_envs
