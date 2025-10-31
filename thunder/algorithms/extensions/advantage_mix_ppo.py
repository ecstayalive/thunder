import itertools

import torch
from torch import nn

from thunder.rl.utils import gaussian_kl_divergence, unpad_trajectory

from ..ppo import PPO
from .advantage_mix_buffer import AdvantageMixBuffer

GIGA = 2**30


class AdvantageMixPPO(PPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init_storage(self):
        self.storage = AdvantageMixBuffer(self.num_collects, self.device)
        self.transition = self.storage.Transition()

    @torch.inference_mode()
    def act(self, actor_obs, critic_obs=None):
        t = self.transition
        t.actor_obs = self.as_th(actor_obs)
        if critic_obs is not None:
            t.critic_obs = self.as_th(critic_obs)
        t.actor_hidden = self.last_actor_hidden
        t.critic_hidden = self.last_critic_hidden

        actions = self.actor.explore(t.actor_obs, self.last_actor_hidden, return_joint_prob=False)
        (
            (t.mu, t.sigma),
            (t.actions, t.actions_log_prob),
            self.last_actor_hidden,
        ) = actions
        t.values, self.last_critic_hidden = self.critic(t.critic_obs, self.last_critic_hidden)
        return t.actions.cpu().numpy()

    def update(self, epoch=0, warmup=False):
        # Learning step
        self.storage.compute_returns(self.gamma, self.lamda)
        summary = self._warmup_step() if warmup else self._train_step(epoch)
        if torch.cuda.memory_reserved() > 4 * GIGA:
            # TODO: FIGURE OUT WHY TORCH IS CONSTANTLY
            #  RESERVING MEMORY DURING RNN TRAINING
            torch.cuda.empty_cache()
        self.storage.clear()
        summary.update(
            {
                "PPO/learning_rate": self.learning_rate,
                "PPO/entropy": self.entropy_coef,
            }
        )
        return summary

    def _train_step(self, epoch=0) -> dict:
        mean_value_loss = torch.zeros(1, device=self.device)
        mean_surrogate_loss = torch.zeros(1, device=self.device)
        mean_ratio = torch.zeros(1, device=self.device)
        mean_action_std = torch.zeros(1, device=self.device)
        mean_torso_action_std = torch.zeros(1, device=self.device)
        mean_arm_action_std = torch.zeros(1, device=self.device)
        mean_extra_losses = None
        for batch in self.storage.sampler(self.num_mini_batches, self.num_learning_epochs):
            (batch.curr_mu, batch.curr_sigma), *_ = self.actor.explore(
                batch.actor_obs,
                hx=batch.actor_hidden,
                sample=False,
                return_joint_prob=False,
            )
            if batch.curr_mu.shape[:-1] != batch.actions.shape[:-1]:
                batch.curr_mu = unpad_trajectory(
                    batch.curr_mu, batch.trajectory_masks, self.num_collects
                )
                batch.curr_sigma = unpad_trajectory(
                    batch.curr_sigma, batch.trajectory_masks, self.num_collects
                )
            curr_actions_log_prob, curr_entropy = self.actor.calc_log_prob_entropy(
                batch.curr_mu, batch.curr_sigma, batch.actions, return_joint_prob=False
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
            prob_ratio = curr_actions_log_prob - batch.actions_log_prob
            torso_ratio = torch.exp(torch.sum(prob_ratio[..., :12], dim=-1, keepdim=True))
            arm_ratio = torch.exp(torch.sum(prob_ratio[..., -6:], dim=-1, keepdim=True))
            # torso_ratio = torch.exp(
            #     prob_ratio[..., :12].sum(dim=-1, keepdim=True)
            #     + prob_ratio[..., -6:].detach().sum(dim=-1, keepdim=True)
            # )
            # arm_ratio = torch.exp(
            #     prob_ratio[..., :12].detach().sum(dim=-1, keepdim=True)
            #     + prob_ratio[..., -6:].sum(dim=-1, keepdim=True)
            # )
            ratio = torch.cat((torso_ratio, arm_ratio), dim=-1)
            surrogate_loss = (
                -torch.min(
                    batch.advantages * ratio,
                    batch.advantages * ratio.clamp(1.0 - self.clip_ratio, 1.0 + self.clip_ratio),
                )
                .sum(dim=-1)
                .mean()
            )

            # Value function loss
            if self.clip_value_loss:
                value_diff = batch.curr_values - batch.values
                value_clipped = batch.values + value_diff.clamp(-self.clip_ratio, self.clip_ratio)
                value_losses = (batch.curr_values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).sum(dim=-1).mean()
            else:
                value_loss = (batch.returns - batch.curr_values).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss
            if self.enforce_avg_std is None:
                loss -= self.entropy_coef * curr_entropy.sum(dim=-1).mean()
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
                mean_action_std += batch.curr_sigma.mean()
                if batch.curr_sigma.shape[-1] > 12:
                    # meaning that the robot has an arm
                    mean_torso_action_std += batch.curr_sigma[..., :12].mean()
                    mean_arm_action_std += batch.curr_sigma[..., 12:].mean()

        if self.min_std is not None or self.max_std is not None:
            self.actor.clamp_exploration_std(min=self.min_std, max=self.max_std)

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_ratio /= num_updates
        mean_action_std /= num_updates
        mean_torso_action_std /= num_updates
        mean_arm_action_std /= num_updates

        data = {
            "PPO/value_loss": mean_value_loss.item(),
            "PPO/surrogate": mean_surrogate_loss.item(),
            "PPO/ratio": mean_ratio.item(),
            "PPO/action_std": mean_action_std.item(),
            "PPO/torso_action_std": mean_torso_action_std.item(),
            "PPO/arm_action_std": mean_arm_action_std.item(),
        }
        if mean_extra_losses is not None:
            mean_extra_losses = (mean_extra_losses / num_updates).cpu().numpy()
            data.update(self._get_extra_loss_info(mean_extra_losses))

        return data
