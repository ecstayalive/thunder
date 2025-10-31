import itertools
from typing import Optional

import numpy as np
import torch
import torch.optim as optim
from torch import nn

from thunder.nn import freeze
from thunder.rl.actor import RoaActor
from thunder.rl.utils import gaussian_kl_divergence, unpad_trajectory

from ..ppo import PPO
from .roa_ppo_buffer import MixRoaBuffer, RoaBuffer

GIGA = 2**30


class RoaPPO(PPO):

    def __init__(self, actor: RoaActor, *args, **kwargs):
        super().__init__(actor, *args, **kwargs)
        self.actor: RoaActor = actor.to(self.device)
        self.last_obs_enc_hx: torch.Tensor = None
        self.last_state_enc_hx: torch.Tensor = None
        # optimizers
        self.actor_obs_enc_optim = optim.Adam(self.actor.encoder.parameters(), self.learning_rate)

    def init_storage(self):
        """ """
        self.storage: RoaBuffer = RoaBuffer(self.num_collects, self.device)
        self.transition: RoaBuffer.Transition = self.storage.Transition()

    @torch.inference_mode()
    def act(
        self, actor_obs: np.ndarray, critic_obs: np.ndarray, adaption: bool = False
    ) -> np.ndarray:
        """
        Args:
            actor_obs: Incomplete observations with noise and delays
            critic_obs: Delay-free and noise-free, representing as much information as possible.
            adaption: if True, use adaption module to transform observation to latent variables,
                same as deploy the network in real robot.
        """
        t = self.transition
        t.actor_obs = self.as_th(actor_obs)
        if critic_obs is not None:
            t.critic_obs = self.as_th(critic_obs)
        t.obs_enc_hidden = self.last_obs_enc_hx
        t.state_enc_hidden = self.last_state_enc_hx
        t.critic_hidden = self.last_critic_hidden
        t.obs_latent, self.last_obs_enc_hx = self.actor.encode_obs(
            t.actor_obs, self.last_obs_enc_hx
        )
        t.state_latent, self.last_state_enc_hx = self.actor.encode_state(
            t.critic_obs, self.last_state_enc_hx
        )
        if adaption:
            (t.mu, t.sigma), (t.actions, t.actions_log_prob) = self.actor.decode_action(
                t.obs_latent
            )
        else:
            (t.mu, t.sigma), (t.actions, t.actions_log_prob) = self.actor.decode_action(
                t.state_latent
            )
        t.values, self.last_critic_hidden = self.critic(t.critic_obs, self.last_critic_hidden)
        return t.actions.cpu().numpy()

    def clear_hidden_states(self, dones):
        super().clear_hidden_states(dones)
        if self.last_obs_enc_hx is not None:
            if isinstance(self.last_obs_enc_hx, torch.Tensor):  # gru
                self.last_obs_enc_hx[..., dones, :] = 0.0
            else:
                for hidden in self.last_obs_enc_hx:  # lstm
                    hidden[..., dones, :] = 0.0

        if self.last_state_enc_hx is not None:
            if isinstance(self.last_state_enc_hx, torch.Tensor):  # gru
                self.last_state_enc_hx[..., dones, :] = 0.0
            else:
                for hidden in self.last_state_enc_hx:  # lstm
                    hidden[..., dones, :] = 0.0

    def update(self, warmup=False, adaption: bool = False, regular_coeff: float = 0.0):
        # Learning step
        self.storage.compute_returns(self.gamma, self.lamda)
        if warmup:
            summary = self._warmup_step
        elif adaption:
            summary = self._adaption_step()
        else:
            summary = self._train_step(regular_coeff)
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

    def _train_step(self, regular_coeff: float = 0.0) -> dict:
        mean_value_loss = torch.zeros(1, device=self.device)
        mean_value = torch.zeros(1, device=self.device)
        mean_surrogate_loss = torch.zeros(1, device=self.device)
        mean_ratio = torch.zeros(1, device=self.device)
        action_std = torch.zeros(1, device=self.device)
        mean_torso_action_std = torch.zeros(1, device=self.device)
        mean_arm_action_std = torch.zeros(1, device=self.device)
        mean_extra_losses = None
        mean_regular_loss: torch.Tensor = torch.zeros(1, device=self.device)
        for batch in self.storage._recurrent_sampler(
            self.num_mini_batches, self.num_learning_epochs
        ):
            batch: RoaBuffer.Batch
            # critic
            batch.curr_values, *_ = self.critic(batch.critic_obs, batch.critic_hidden)
            if batch.curr_values.shape[:-1] != batch.actions.shape[:-1]:
                batch.curr_values = unpad_trajectory(
                    batch.curr_values, batch.trajectory_masks, self.num_collects
                )
            # update actor with regularization online adaption (roa)
            with torch.inference_mode():
                curr_latent_label, _ = self.actor.encode_obs(batch.actor_obs, batch.obs_enc_hidden)
            curr_latent, _ = self.actor.encode_state(batch.critic_obs, batch.state_enc_hidden)
            (batch.curr_mu, batch.curr_sigma) = self.actor.decode_action(curr_latent, sample=False)
            if batch.curr_mu.shape[:-1] != batch.actions.shape[:-1]:
                batch.curr_mu = unpad_trajectory(
                    batch.curr_mu, batch.trajectory_masks, self.num_collects
                )
                batch.curr_sigma = unpad_trajectory(
                    batch.curr_sigma, batch.trajectory_masks, self.num_collects
                )
                curr_latent = unpad_trajectory(
                    curr_latent, batch.trajectory_masks, self.num_collects
                )
                curr_latent_label = unpad_trajectory(
                    curr_latent_label, batch.trajectory_masks, self.num_collects
                )

            curr_actions_log_prob, curr_entropy = self.actor.calc_log_prob_entropy(
                batch.curr_mu, batch.curr_sigma, batch.actions
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
            surrogate_loss = -torch.min(
                batch.advantages * ratio,
                batch.advantages * ratio.clamp(1.0 - self.clip_ratio, 1.0 + self.clip_ratio),
            ).mean()

            # Value function loss
            if self.clip_value_loss:
                value_diff = batch.curr_values - batch.values
                value_clipped = batch.values + value_diff.clamp(-self.clip_ratio, self.clip_ratio)
                value_losses = (batch.curr_values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - batch.curr_values).pow(2).mean()
            regular_loss = (
                torch.linalg.norm(curr_latent - curr_latent_label, ord=2, dim=-1).pow(2).mean()
            )
            loss = surrogate_loss + self.value_loss_coef * value_loss + regular_coeff * regular_loss
            # Other loss
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
            with freeze(self.actor.encoder):
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
                mean_regular_loss += regular_loss.detach()
                mean_value_loss += value_loss
                mean_value += batch.curr_values.mean()
                mean_surrogate_loss += surrogate_loss
                mean_ratio += torch.abs(ratio - 1.0).mean()
                action_std += batch.curr_sigma.mean()
                if batch.curr_sigma.shape[-1] > 12:
                    # Meaning that the robot has an manipulator
                    mean_torso_action_std += batch.curr_sigma[..., :12].mean()
                    mean_arm_action_std += batch.curr_sigma[..., 12:].mean()

        if self.min_std is not None or self.max_std is not None:
            self.actor.clamp_exploration_std(min=self.min_std, max=self.max_std)

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_value /= num_updates
        mean_surrogate_loss /= num_updates
        mean_ratio /= num_updates
        action_std /= num_updates
        mean_torso_action_std /= num_updates
        mean_arm_action_std /= num_updates
        mean_regular_loss /= num_updates

        # record
        data = {
            "PPO/action_std": action_std.item(),
            "PPO/torso_action_std": mean_torso_action_std.item(),
            "PPO/arm_action_std": mean_arm_action_std.item(),
            "PPO/value": mean_value.item(),
            "PPO/value_loss": mean_value_loss.item(),
            "PPO/surrogate": mean_surrogate_loss.item(),
            "PPO/ratio": mean_ratio.item(),
            "PPO/regular_loss": mean_regular_loss.item(),
        }
        if mean_extra_losses is not None:
            mean_extra_losses = (mean_extra_losses / num_updates).cpu().numpy()
            data.update(self._get_extra_loss_info(mean_extra_losses))

        return data

    def _adaption_step(self):
        mean_adaption_loss: torch.Tensor = torch.zeros(1, device=self.device)
        for batch in self.storage._recurrent_sampler(
            self.num_mini_batches, self.num_learning_epochs
        ):
            batch: RoaBuffer.Batch
            curr_latent, _ = self.actor.encode_obs(batch.actor_obs, batch.obs_enc_hidden)
            with torch.inference_mode():
                curr_latent_label, _ = self.actor.encode_state(
                    batch.critic_obs, batch.state_enc_hidden
                )
            (batch.curr_mu, batch.curr_sigma) = self.actor.decode_action(curr_latent, sample=False)
            if batch.curr_mu.shape[:-1] != batch.actions.shape[:-1]:
                # batch.curr_mu = unpad_trajectory(
                #     batch.curr_mu, batch.trajectory_masks, self.num_collects
                # )
                # batch.curr_sigma = unpad_trajectory(
                #     batch.curr_sigma, batch.trajectory_masks, self.num_collects
                # )
                curr_latent = unpad_trajectory(
                    curr_latent, batch.trajectory_masks, self.num_collects
                )
                curr_latent_label = unpad_trajectory(
                    curr_latent_label, batch.trajectory_masks, self.num_collects
                )
            adaption_loss = (
                torch.linalg.norm(curr_latent - curr_latent_label, ord=2, dim=-1).pow(2).mean()
            )
            self.actor_obs_enc_optim.zero_grad()
            adaption_loss.backward()
            nn.utils.clip_grad_norm_(
                self.actor.encoder.parameters(),
                self.max_grad_norm,
            )
            self.actor_obs_enc_optim.step()
            with torch.inference_mode():
                mean_adaption_loss += adaption_loss.detach()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_adaption_loss /= num_updates

        # record
        return {
            "PPO/adaption_loss": mean_adaption_loss.item(),
        }


class MixRoaPPO(RoaPPO):

    def __init__(self, actor: RoaActor, *args, **kwargs):
        super().__init__(actor, *args, **kwargs)

    def init_storage(self):
        """ """
        self.storage: MixRoaBuffer = MixRoaBuffer(self.num_collects, self.device)
        self.transition: MixRoaBuffer.Transition = self.storage.Transition()

    @torch.inference_mode()
    def act(self, actor_obs, critic_obs, deploy_mode: bool = False):
        """
        Args:
            actor_obs: Incomplete observations with noise and delays
            critic_obs: Delay-free and noise-free, representing as much information as possible.
            deploy_mode: if True, use adaption module to transform observation to latent variables,
                same as deploy the network in real robot.
        """
        t = self.transition
        t.actor_obs = self.as_th(actor_obs)
        if critic_obs is not None:
            t.critic_obs = self.as_th(critic_obs)
        t.obs_enc_hidden = self.last_obs_enc_hx
        t.state_enc_hidden = self.last_state_enc_hx
        t.critic_hidden = self.last_critic_hidden
        t.obs_latent, self.last_obs_enc_hx = self.actor.encode_obs(
            t.actor_obs, self.last_obs_enc_hx
        )
        t.state_latent, self.last_state_enc_hx = self.actor.encode_state(
            t.critic_obs, self.last_state_enc_hx
        )
        if deploy_mode:
            (t.mu, t.sigma), (t.actions, t.actions_log_prob) = self.actor.decode_action(
                t.obs_latent, return_joint_prob=False
            )
        else:
            (t.mu, t.sigma), (t.actions, t.actions_log_prob) = self.actor.decode_action(
                t.state_latent, return_joint_prob=False
            )
        t.values, self.last_critic_hidden = self.critic(t.critic_obs, self.last_critic_hidden)
        return t.actions.cpu().numpy()

    def _train_step(self, regular_coeff: float = 0.0) -> dict:
        mean_value_loss = torch.zeros(1, device=self.device)
        mean_value = torch.zeros(1, device=self.device)
        mean_surrogate_loss = torch.zeros(1, device=self.device)
        mean_ratio = torch.zeros(1, device=self.device)
        action_std = torch.zeros(1, device=self.device)
        mean_torso_action_std = torch.zeros(1, device=self.device)
        mean_arm_action_std = torch.zeros(1, device=self.device)
        mean_extra_losses = None
        mean_regular_loss: float = 0.0
        for batch in self.storage._recurrent_sampler(
            self.num_mini_batches, self.num_learning_epochs
        ):
            # critic
            batch.curr_values, *_ = self.critic(batch.critic_obs, batch.critic_hidden)
            if batch.curr_values.shape[:-1] != batch.actions.shape[:-1]:
                batch.curr_values = unpad_trajectory(
                    batch.curr_values, batch.trajectory_masks, self.num_collects
                )
            # update actor with regularization online adaption (roa)
            curr_latent, _ = self.actor.encode_state(batch.critic_obs, batch.state_enc_hidden)
            (batch.curr_mu, batch.curr_sigma) = self.actor.decode_action(curr_latent, sample=False)
            if batch.curr_mu.shape[:-1] != batch.actions.shape[:-1]:
                batch.curr_mu = unpad_trajectory(
                    batch.curr_mu, batch.trajectory_masks, self.num_collects
                )
                batch.curr_sigma = unpad_trajectory(
                    batch.curr_sigma, batch.trajectory_masks, self.num_collects
                )
                curr_latent = unpad_trajectory(
                    curr_latent, batch.trajectory_masks, self.num_collects
                )

            curr_actions_log_prob, curr_entropy = self.actor.calc_log_prob_entropy(
                batch.curr_mu, batch.curr_sigma, batch.actions, return_joint_prob=False
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
            ################################################
            # Surrogate loss
            ################################################
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
            ################################################
            # Value function loss
            ################################################
            if self.clip_value_loss:
                value_diff = batch.curr_values - batch.values
                value_clipped = batch.values + value_diff.clamp(-self.clip_ratio, self.clip_ratio)
                value_losses = (batch.curr_values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).sum(-1).mean()
            else:
                value_loss = (batch.returns - batch.curr_values).pow(2).mean()
            #################################################
            # Loss
            #################################################
            if self.enforce_avg_std is None:
                surrogate_loss -= self.entropy_coef * curr_entropy.sum(dim=-1).mean()
            else:
                surrogate_loss += abs(batch.curr_sigma.mean() - self.enforce_avg_std)
            extra_losses = self._calc_extra_losses(batch)
            if extra_losses:
                if mean_extra_losses is None:
                    mean_extra_losses = torch.zeros(len(extra_losses), device=self.device)
                for i, extra_loss in enumerate(extra_losses):
                    surrogate_loss += extra_loss
                    mean_extra_losses[i] += extra_loss.detach()
            #################################################
            # Gradient step
            #################################################
            regular_loss = torch.pow(curr_latent - batch.obs_latent, 2).mean()
            loss = surrogate_loss + self.value_loss_coef * value_loss + regular_coeff * regular_loss
            self.actor_state_enc_optim.zero_grad()
            self.actor_action_dec_optim.zero_grad()
            self.critic_optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                itertools.chain(
                    self.actor.state_enc.parameters(),
                    self.actor.action_dec.parameters(),
                    self.critic.parameters(),
                ),
                self.max_grad_norm,
            )
            self.actor_state_enc_optim.step()
            self.actor_action_dec_optim.step()
            self.critic_optim.step()
            mean_regular_loss += regular_loss.detach().item()

            self.actor.resample_distribution()
            with torch.inference_mode():
                mean_value_loss += value_loss
                mean_value += batch.curr_values.mean()
                mean_surrogate_loss += surrogate_loss
                mean_ratio += torch.abs(ratio - 1.0).mean()
                action_std += batch.curr_sigma.mean()
                if batch.curr_sigma.shape[-1] > 12:
                    # meaning that the robot has an arm
                    mean_torso_action_std += batch.curr_sigma[..., :12].mean()
                    mean_arm_action_std += batch.curr_sigma[..., 12:].mean()

        if self.min_std is not None or self.max_std is not None:
            self.actor.clamp_exploration_std(min=self.min_std, max=self.max_std)

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_value /= num_updates
        mean_surrogate_loss /= num_updates
        mean_ratio /= num_updates
        action_std /= num_updates
        mean_torso_action_std /= num_updates
        mean_arm_action_std /= num_updates
        mean_regular_loss /= num_updates

        # record
        data = {
            "PPO/action_std": action_std.item(),
            "PPO/torso_action_std": mean_torso_action_std.item(),
            "PPO/arm_action_std": mean_arm_action_std.item(),
            "PPO/value": mean_value.item(),
            "PPO/value_loss": mean_value_loss.item(),
            "PPO/surrogate": mean_surrogate_loss.item(),
            "PPO/ratio": mean_ratio.item(),
            "PPO/regular_loss": mean_regular_loss,
        }
        if mean_extra_losses is not None:
            mean_extra_losses = (mean_extra_losses / num_updates).cpu().numpy()
            data.update(self._get_extra_loss_info(mean_extra_losses))

        return data
