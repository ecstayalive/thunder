import itertools

import torch
from torch import nn

from thunder.rl.actor import DecActor
from thunder.rl.utils import gaussian_kl_divergence, unpad_trajectory

from ..ppo import PPO
from .denoise_ppo_buffer import DenoiseBuffer

GIGA = 2**30


class PpoX(PPO):
    def __init__(self, actor: DecActor, *args, **kwargs):
        super().__init__(actor, *args, **kwargs)
        self.actor: DecActor = actor.to(self.device)
        self.last_state_encoder_hidden = None
        # self.last_action_decoder_hidden = None

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
        t.state_encoder_hidden = self.last_state_encoder_hidden
        # t.action_decoder_hidden = self.last_action_decoder_hidden
        t.critic_hidden = self.last_critic_hidden

        actions = self.actor.explore(
            t.actor_obs,
            self.last_state_encoder_hidden,
            enable_state_decoder=False,
        )
        (
            (t.mu, t.sigma),
            (t.actions, t.actions_log_prob),
            self.last_state_encoder_hidden,
            _,
        ) = actions
        t.values, self.last_critic_hidden = self.critic(t.critic_obs, self.last_critic_hidden)
        return t.actions.cpu().numpy()

    def clear_hidden_states(self, dones):
        if self.last_state_encoder_hidden is not None:
            if isinstance(self.last_state_encoder_hidden, torch.Tensor):  # gru
                self.last_state_encoder_hidden[..., dones, :] = 0.0
            else:
                for hidden in self.last_state_encoder_hidden:  # lstm
                    hidden[..., dones, :] = 0.0

        # if self.last_action_decoder_hidden is not None:
        #     if isinstance(self.last_action_decoder_hidden, torch.Tensor):  # gru
        #         self.last_action_decoder_hidden[..., dones, :] = 0.0
        #     else:
        #         for hidden in self.last_action_decoder_hidden:  # lstm
        #             hidden[..., dones, :] = 0.0

        if self.last_critic_hidden is not None:
            if isinstance(self.last_critic_hidden, torch.Tensor):  # gru
                self.last_critic_hidden[..., dones, :] = 0.0
            else:
                for hidden in self.last_critic_hidden:  # lstm
                    hidden[..., dones, :] = 0.0

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
            (batch.curr_mu, batch.curr_sigma), _, batch.curr_reconstruction_state = (
                self.actor.explore(
                    batch.actor_obs,
                    batch.state_encoder_hidden,
                    sample=False,
                    enable_state_decoder=True,
                )
            )
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
            reconstruction_loss = 0.0
            if epoch % 50:
                reconstruction_loss = torch.mean(
                    torch.norm(batch.curr_reconstruction_state - ground_truth_state, 2.0, -1)
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
            loss = surrogate_loss + self.value_loss_coef * value_loss + 0.1 * reconstruction_loss
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
                mean_reconstruction_loss += reconstruction_loss
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
            "PPO/denoise_state_loss": mean_reconstruction_loss.item(),
            "PPO/ratio": mean_ratio.item(),
        }
        if mean_extra_losses is not None:
            mean_extra_losses = (mean_extra_losses / num_updates).cpu().numpy()
            data.update(self._get_extra_loss_info(mean_extra_losses))

        return data
