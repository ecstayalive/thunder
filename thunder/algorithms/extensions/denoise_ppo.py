import itertools

import torch
import torch.optim as optim
from torch import nn

from thunder.nn import freeze
from thunder.rl.actor import DecActor
from thunder.rl.utils import gaussian_kl_divergence, unpad_trajectory

from ..ppo import PPO
from .denoise_ppo_buffer import DenoiseBuffer

GIGA = 2**30


class DenoisePPO(PPO):
    def __init__(self, actor: DecActor, *args, **kwargs):
        super().__init__(actor, *args, **kwargs)
        self.actor: DecActor = actor.to(self.device)
        self.state_enc_optim = optim.Adam(self.actor.state_encoder.parameters(), self.learning_rate)
        self.state_dec_optim = optim.Adam(self.actor.decoder.parameters(), self.learning_rate)
        self.regularization_period = 20
        self.regularization_coef = 0.42
        self.reconstruction_reward = 0.0
        self.reconstruction_reward_coef = 0.1

    def init_storage(self):
        """ """
        self.storage: DenoiseBuffer = DenoiseBuffer(self.num_collects, self.device)
        self.transition: DenoiseBuffer.Transition = self.storage.Transition()

    @torch.inference_mode()
    def act(self, actor_obs, critic_obs):
        """
        Args:
            actor_obs: Incomplete observations with noise and delays
            critic_obs: Delay-free and noise-free, representing as much information as possible.
        """
        t = self.transition
        t.actor_obs = self.as_th(actor_obs)
        t.critic_obs = self.as_th(critic_obs)
        t.actor_hidden = self.last_actor_hidden
        t.critic_hidden = self.last_critic_hidden
        t.latent_features = self.actor.encode_obs(t.actor_obs)
        (t.mu, t.sigma), (t.actions, t.actions_log_prob), self.last_actor_hidden = (
            self.actor.decode_action(t.latent_features, self.last_actor_hidden)
        )
        reconstruction_state = self.actor.decode_obs(t.latent_features)
        self.reconstruction_reward = torch.mean(torch.abs(reconstruction_state - t.critic_obs))
        t.values, self.last_critic_hidden = self.critic(t.critic_obs, self.last_critic_hidden)
        return t.actions.cpu().numpy()

    @torch.inference_mode()
    def step(self, next_obs, rewards, dones, timeouts, next_critic_obs=None):
        self.transition.next_obs = self.as_th(next_obs)
        self.transition.next_latent_features = self.actor.encode_obs(self.transition.next_obs)
        if next_critic_obs is not None:
            self.transition.next_critic_obs = self.as_th(next_critic_obs)
        # self.transition.rewards = (
        #     self.as_th(rewards)
        #     + self.reconstruction_reward * self.rfeconstruction_reward_coef
        # )
        self.transition.rewards = rewards
        self.transition.dones = dones
        self.transition.timeouts = timeouts
        # duplicated inference for convenience and performance
        self.transition.next_values = self.critic.evaluate(
            self.transition.next_critic_obs, self.last_critic_hidden  # updated
        )
        self.storage.add_transition(self.transition)
        self.transition.__init__()
        self.clear_hidden_states(dones)

    def clear_hidden_states(self, dones):
        if self.last_actor_hidden is not None:
            if isinstance(self.last_actor_hidden, torch.Tensor):  # gru
                self.last_actor_hidden[..., dones, :] = 0.0
            else:
                for hidden in self.last_actor_hidden:  # lstm
                    hidden[..., dones, :] = 0.0

        if self.last_critic_hidden is not None:
            if isinstance(self.last_critic_hidden, torch.Tensor):  # gru
                self.last_critic_hidden[..., dones, :] = 0.0
            else:
                for hidden in self.last_critic_hidden:  # lstm
                    hidden[..., dones, :] = 0.0

    def _warmup_step(self):
        mean_value_loss = torch.zeros(1, device=self.device)
        mean_reconstruction_loss = torch.zeros(1, device=self.device)
        for batch in self.storage._recurrent_sampler(
            self.num_mini_batches, self.num_learning_epochs
        ):
            latent_features = self.actor.encode_obs(batch.actor_obs)
            batch.curr_reconstruction_state = self.actor.decode_obs(latent_features)
            ground_truth_state = batch.critic_obs
            curr_value, _ = self.critic(batch.critic_obs, batch.critic_hidden)
            if curr_value.shape[:-1] != batch.actions.shape[:-1]:
                curr_value = unpad_trajectory(curr_value, batch.trajectory_masks, self.num_collects)
                batch.curr_reconstruction_state = unpad_trajectory(
                    batch.curr_reconstruction_state,
                    batch.trajectory_masks,
                    self.num_collects,
                )
                ground_truth_state = unpad_trajectory(
                    ground_truth_state, batch.trajectory_masks, self.num_collects
                )

            # Value function loss
            if self.clip_value_loss:
                value_diff = curr_value - batch.values
                value_clipped = batch.values + value_diff.clamp(-self.clip_ratio, self.clip_ratio)
                value_losses = (curr_value - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - curr_value).pow(2).mean()
            # de-noising auto-encoder loss
            reconstruction_loss = torch.mean(
                torch.abs(batch.curr_reconstruction_state - ground_truth_state)
            )
            loss = (
                self.value_loss_coef * value_loss + self.regularization_coef * reconstruction_loss
            )
            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.detach()
            mean_reconstruction_loss += reconstruction_loss.detach()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_reconstruction_loss /= num_updates
        return {
            "PPO/value_function": mean_value_loss.item(),
            "PPO/surrogate": 0.0,
            "PPO/ratio": 0.0,
            "PPO/reconstruction_loss": mean_reconstruction_loss.item(),
        }

    def _train_step(self, epoch=0) -> dict:
        mean_value_loss = torch.zeros(1, device=self.device)
        mean_surrogate_loss = torch.zeros(1, device=self.device)
        mean_reconstruction_loss = torch.zeros(1, device=self.device)
        mean_ratio = torch.zeros(1, device=self.device)
        mean_exploration = torch.zeros(1, device=self.device)
        torso_mean_exploration = torch.zeros(1, device=self.device)
        arm_mean_exploration = torch.zeros(1, device=self.device)
        mean_extra_losses = None
        for batch in self.storage._recurrent_sampler(
            self.num_mini_batches, self.num_learning_epochs
        ):
            latent_features = self.actor.encode_obs(batch.actor_obs)
            (batch.curr_mu, batch.curr_sigma), _ = self.actor.decode_action(
                latent_features, batch.actor_hidden, sample=False
            )
            if epoch % self.regularization_period == 0:
                batch.curr_reconstruction_state = self.actor.decode_obs(latent_features)
            else:
                batch.curr_reconstruction_state = self.actor.decode_obs(latent_features.detach())
            ground_truth_state = batch.critic_obs
            if batch.curr_mu.shape[:-1] != batch.actions.shape[:-1]:
                batch.curr_mu = unpad_trajectory(
                    batch.curr_mu, batch.trajectory_masks, self.num_collects
                )
                batch.curr_sigma = unpad_trajectory(
                    batch.curr_sigma, batch.trajectory_masks, self.num_collects
                )
                batch.curr_reconstruction_state = unpad_trajectory(
                    batch.curr_reconstruction_state,
                    batch.trajectory_masks,
                    self.num_collects,
                )
                ground_truth_state = unpad_trajectory(
                    ground_truth_state, batch.trajectory_masks, self.num_collects
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
            ################################################
            # Surrogate loss
            ################################################
            ratio = torch.exp(curr_actions_log_prob - batch.actions_log_prob)
            surrogate_loss = -torch.min(
                batch.advantages * ratio,
                batch.advantages * ratio.clamp(1.0 - self.clip_ratio, 1.0 + self.clip_ratio),
            ).mean()
            ################################################
            # Denoise auto-encoder loss
            ################################################
            reconstruction_loss = torch.mean(
                torch.abs(batch.curr_reconstruction_state - ground_truth_state)
            )
            ################################################
            # Value function loss
            ################################################
            if self.clip_value_loss:
                value_diff = batch.curr_values - batch.values
                value_clipped = batch.values + value_diff.clamp(-self.clip_ratio, self.clip_ratio)
                value_losses = (batch.curr_values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - batch.curr_values).pow(2).mean()
            #################################################
            # Loss
            #################################################
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
            #################################################
            # Gradient step
            #################################################
            if epoch % self.regularization_period == 0:
                self.state_enc_optim.zero_grad()
                # self.state_dec_optim.zero_grad()
                reconstruction_loss.backward()
                nn.utils.clip_grad_norm_(
                    itertools.chain(
                        self.actor.state_encoder.parameters(),
                        self.actor.decoder.parameters(),
                    ),
                    self.max_grad_norm,
                )
                # self.state_dec_optim.step()
                self.state_enc_optim.step()
            else:
                loss += self.regularization_coef * reconstruction_loss
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    itertools.chain(self.actor.parameters(), self.critic.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

            self.actor.resample_distribution()

            with torch.inference_mode():
                mean_value_loss += value_loss
                mean_surrogate_loss += surrogate_loss
                mean_reconstruction_loss += reconstruction_loss
                mean_ratio += torch.abs(ratio - 1.0).mean()
                mean_exploration += batch.curr_sigma.mean()
                if batch.curr_sigma.shape[-1] > 12:
                    # meaning that the robot has an arm
                    torso_mean_exploration += batch.curr_sigma[..., :12].mean()
                    arm_mean_exploration += batch.curr_sigma[..., 12:].mean()
        # if epoch % self.regularization_period == 0:
        #     self.regularization_anneal()
        if self.min_std is not None or self.max_std is not None:
            self.actor.clamp_exploration_std(min=self.min_std, max=self.max_std)

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_reconstruction_loss /= num_updates
        mean_ratio /= num_updates
        mean_exploration /= num_updates
        torso_mean_exploration /= num_updates
        arm_mean_exploration /= num_updates

        data = {
            "PPO/exploration": mean_exploration.item(),
            "PPO/torso_exploration": torso_mean_exploration.item(),
            "PPO/arm_mean_exploration": arm_mean_exploration.item(),
            "PPO/value_function": mean_value_loss.item(),
            "PPO/surrogate": mean_surrogate_loss.item(),
            "PPO/reconstruction_loss": mean_reconstruction_loss.item(),
            "PPO/ratio": mean_ratio.item(),
        }
        if mean_extra_losses is not None:
            mean_extra_losses = (mean_extra_losses / num_updates).cpu().numpy()
            data.update(self._get_extra_loss_info(mean_extra_losses))

        return data

    def regularization_anneal(self):
        self.regularization_coef *= 0.996
        self.regularization_coef = max(self.regularization_coef, 0.01)
