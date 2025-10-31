import itertools

import torch
import torch.nn as nn

from thunder.rl.utils import gaussian_kl_divergence, unpad_trajectory

from ..ppo.ppo import PPO
from .ggf_buffer import GgfRolloutBuffer

__all__ = ["GgfPPO"]


class GgfPPO(PPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init_storage(self):
        self.storage = GgfRolloutBuffer(self.num_collects, self.device)
        self.transition = self.storage.Transition()
        self.ggf_weight = torch.tensor([0.64, 0.36], device=self.device)

    def _train_step(self, epoch=0) -> dict:
        mean_value_loss = torch.zeros(1, device=self.device)
        mean_surrogate_loss = torch.zeros(1, device=self.device)
        mean_ratio = torch.zeros(1, device=self.device)
        mean_exploration = torch.zeros(1, device=self.device)
        torso_mean_exploration = torch.zeros(1, device=self.device)
        arm_mean_exploration = torch.zeros(1, device=self.device)
        mean_extra_losses = None
        for batch in self.storage.sampler(self.num_mini_batches, self.num_learning_epochs):
            (batch.curr_mu, batch.curr_sigma), *_ = self.actor.explore(
                batch.actor_obs, batch.actor_hidden, sample=False
            )
            if batch.curr_mu.shape[:-1] != batch.actions.shape[:-1]:
                batch.curr_mu = unpad_trajectory(
                    batch.curr_mu, batch.trajectory_masks, self.num_collects
                )
                batch.curr_sigma = unpad_trajectory(
                    batch.curr_sigma, batch.trajectory_masks, self.num_collects
                )
            curr_actions_log_prob, curr_entropy = self.actor.calc_log_prob_entropy(
                batch.curr_mu, batch.curr_sigma, batch.actions
            )

            batch.curr_values, *_ = self.critic(batch.critic_obs, batch.critic_hidden)
            if batch.curr_values.shape[:-1] != batch.actions.shape[:-1]:
                batch.curr_values = unpad_trajectory(
                    batch.curr_values, batch.trajectory_masks, self.num_collects
                )

            # Adjusting the learning rate using KL divergence
            if self.desired_kl is not None and self.lr_schedule == "adaptive":
                with torch.no_grad():
                    kl_mean = gaussian_kl_divergence(
                        batch.mu, batch.sigma, batch.curr_mu, batch.curr_sigma
                    )

                if kl_mean > self.desired_kl * 2.0:
                    self.learning_rate = max(1e-5, self.learning_rate / 1.2)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = max(1e-5, param_group["lr"] / 1.2)
                elif 0.0 < kl_mean < self.desired_kl / 2.0:
                    self.learning_rate = min(1e-2, self.learning_rate * 1.2)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = min(1e-2, param_group["lr"] * 1.2)

            # Surrogate loss
            ratio = torch.exp(curr_actions_log_prob - batch.actions_log_prob)
            # surrogate_loss = -torch.min(
            #     batch.advantages * ratio,
            #     batch.advantages
            #     * ratio.clamp(1.0 - self.clip_ratio, 1.0 + self.clip_ratio),
            # ).mean()
            surrogate_target = torch.min(
                batch.advantages * ratio,
                batch.advantages * ratio.clamp(1.0 - self.clip_ratio, 1.0 + self.clip_ratio),
            )
            surrogate_target, _ = torch.sort(surrogate_target, dim=-1)
            surrogate_loss = -(surrogate_target * self.ggf_weight).sum(dim=-1).mean()

            # Value function loss
            if self.clip_value_loss:
                value_diff = batch.curr_values - batch.values
                value_clipped = batch.values + value_diff.clamp(-self.clip_ratio, self.clip_ratio)
                value_losses = (batch.curr_values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - batch.curr_values).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss
            if self.enforce_avg_std is None:
                loss -= self.entropy_coef * curr_entropy.mean()
            else:
                loss += abs(batch.curr_sigma.mean() - self.enforce_avg_std)

            extra_losses = self._calc_extra_losses(batch)
            if extra_losses:
                if mean_extra_losses is None:
                    mean_extra_losses = torch.zeros(len(extra_losses), device=self.device)
                for i, extra_loss in enumerate(extra_losses):
                    loss += extra_loss
                    mean_extra_losses[i] += extra_loss.detach()

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                itertools.chain(
                    self.actor.parameters(),
                    self.critic.parameters(),
                ),
                self.max_grad_norm,
            )
            self.optimizer.step()
            self.actor.resample_distribution()

            with torch.inference_mode():
                mean_value_loss += value_loss
                mean_surrogate_loss += surrogate_loss
                mean_ratio += torch.abs(ratio - 1.0).mean()
                mean_exploration += batch.curr_sigma.mean()
                if batch.curr_sigma.shape[-1] > 12:
                    # meaning that the robot has an arm
                    torso_mean_exploration += batch.curr_sigma[..., :12].mean()
                    arm_mean_exploration += batch.curr_sigma[..., 12:].mean()

        if self.min_std is not None or self.max_std is not None:
            self.actor.clamp_exploration_std(min=self.min_std, max=self.max_std)

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_ratio /= num_updates
        mean_exploration /= num_updates
        torso_mean_exploration /= num_updates
        arm_mean_exploration /= num_updates

        data = {
            "PPO/value_function": mean_value_loss.item(),
            "PPO/surrogate": mean_surrogate_loss.item(),
            "PPO/ratio": mean_ratio.item(),
            "PPO/exploration": mean_exploration.item(),
        }
        if batch.curr_sigma.shape[-1] > 12:
            data.update(
                {
                    "PPO/torso_exploration": torso_mean_exploration.item(),
                    "PPO/arm_mean_exploration": arm_mean_exploration.item(),
                }
            )
        if mean_extra_losses is not None:
            mean_extra_losses = (mean_extra_losses / num_updates).cpu().numpy()
            data.update(self._get_extra_loss_info(mean_extra_losses))

        return data
