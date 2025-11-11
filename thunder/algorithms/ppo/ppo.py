import itertools
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from thunder.rl import GeneralActor, GeneralVNet
from thunder.rl.utils import gaussian_kl_divergence, split_trajectory, unpad_trajectory

from .buffer import RolloutBuffer

GIGA = 2**30


class PpoConf:
    actor: GeneralActor
    critic: GeneralVNet
    num_envs: int
    num_collects: int
    num_learning_epochs: int
    num_mini_batches: int
    clip_ratio: float = 0.2
    gamma: float = 0.998
    lamda: float = 0.95
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.0
    learning_rate: float = 5e-4
    critic_lr: Optional[float] = None
    max_grad_norm: float = 0.5
    lr_schedule: str = "adaptive"
    desired_kl: float = 0.01
    clip_value_loss = True
    init_std: Optional[float | np.ndarray] = 0.5
    min_std: Optional[float] = None
    max_std: Optional[float] = None
    enforce_avg_std: Optional[bool] = None
    smooth_coef: Optional[float] = None
    device = "cpu"


class PPO:
    def __init__(
        self,
        actor: GeneralActor,
        critic: GeneralVNet,
        num_envs,
        num_collects,
        num_learning_epochs,
        num_mini_batches,
        clip_ratio: float = 0.2,
        gamma: float = 0.998,
        lamda: float = 0.95,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.0,
        learning_rate: float = 5e-4,
        critic_lr: Optional[float] = None,
        max_grad_norm: float = 0.5,
        lr_schedule: str = "adaptive",
        desired_kl: float = 0.01,
        clip_value_loss=True,
        init_std: Optional[float | np.ndarray] = 0.5,
        min_std: Optional[float] = None,
        max_std: Optional[float] = None,
        enforce_avg_std: Optional[bool] = None,
        smooth_coef: Optional[float] = None,
        device="cpu",
        **kwargs,
    ):
        if kwargs:
            print("PPO: Ignored kwargs: ", *[f"  - {k}: {v}" for k, v in kwargs.items()], sep="\n")

        # PPO components
        self.device = torch.device(device)
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        self.num_envs = num_envs
        self.num_collects = num_collects

        # PPO parameters
        self.clip_ratio = clip_ratio
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lamda = lamda
        self.max_grad_norm = max_grad_norm
        self.clip_value_loss = clip_value_loss
        if init_std is not None:
            self.actor.set_exploration_std(
                init_std if isinstance(init_std, float) else self.as_th(init_std)
            )
        self.min_std = min_std
        self.max_std = max_std
        self.enforce_avg_std = enforce_avg_std
        self.last_actor_hidden: Optional[torch.Tensor | Tuple[torch.Tensor, torch.Tensor]] = None
        self.last_critic_hidden: Optional[torch.Tensor | Tuple[torch.Tensor, torch.Tensor]] = None

        # Optimizer
        self.learning_rate = learning_rate
        self.critic_lr = learning_rate if critic_lr is None else critic_lr
        self.desired_kl = desired_kl
        self.lr_schedule = lr_schedule
        self.optimizer = optim.Adam(
            [
                {"params": self.actor.parameters(), "lr": self.learning_rate},
                {"params": self.critic.parameters(), "lr": self.critic_lr},
            ],
            lr=self.learning_rate,
        )

        # Storage
        self.init_storage()

        # Smooth loss
        self.smooth_coef: Optional[float] = smooth_coef
        self.smooth_conv = self.as_th([[[-1.0, 2.0, -1.0]]])
        self.torso_smooth_weight = 1 / torch.tensor([0.3, 0.5, 0.3] * 4, device=self.device)
        self.arm_smooth_weight = 1 / torch.tensor(
            [0.4, 0.3, 0.3, 0.4, 0.4, 0.4], device=self.device
        )
        self.smooth_weight = torch.cat((self.torso_smooth_weight, self.arm_smooth_weight))

    def init_storage(self):
        self.storage: RolloutBuffer = RolloutBuffer(self.num_collects, self.device)
        self.transition: RolloutBuffer.Transition = self.storage.Transition()

    @torch.inference_mode()
    def act(self, actor_obs: np.ndarray, critic_obs: Optional[np.ndarray] = None) -> np.ndarray:
        t = self.transition
        t.actor_obs = self.as_th(actor_obs)
        if critic_obs is not None:
            t.critic_obs = self.as_th(critic_obs)
        t.actor_hidden = self.last_actor_hidden
        t.critic_hidden = self.last_critic_hidden

        actions = self.actor.explore(t.actor_obs, self.last_actor_hidden)
        ((t.mu, t.sigma), (t.actions, t.actions_log_prob), self.last_actor_hidden) = actions
        t.values, self.last_critic_hidden = self.critic(t.critic_obs, self.last_critic_hidden)
        return t.actions.cpu().numpy()

    @torch.inference_mode()
    def step(
        self,
        next_obs: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        timeouts: np.ndarray,
        next_critic_obs: Optional[np.ndarray] = None,
    ):
        self.transition.next_obs = self.as_th(next_obs)
        if next_critic_obs is not None:
            self.transition.next_critic_obs = self.as_th(next_critic_obs)
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

    def as_th(self, tensor, **kwargs):
        return torch.as_tensor(tensor, **kwargs, device=self.device)

    def update(self, epoch: int = 0, warmup: bool = False) -> Dict[str, float]:
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

    def _warmup_step(self) -> Dict[str, float]:
        mean_value_loss = torch.zeros(1, device=self.device)
        for batch in self.storage.sampler(self.num_mini_batches, self.num_learning_epochs):
            curr_value, _ = self.critic(batch.critic_obs, batch.critic_hidden)
            if curr_value.shape[:-1] != batch.actions.shape[:-1]:
                curr_value = unpad_trajectory(curr_value, batch.trajectory_masks, self.num_collects)

            # Value function loss
            if self.clip_value_loss:
                value_diff = curr_value - batch.values
                value_clipped = batch.values + value_diff.clamp(-self.clip_ratio, self.clip_ratio)
                value_losses = (curr_value - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - curr_value).pow(2).mean()

            loss = self.value_loss_coef * value_loss
            # TODO: ADD EXTRA LOSS TO WARMUP
            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.detach()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        return {
            "PPO/value_loss": mean_value_loss.item(),
            "PPO/surrogate": 0.0,
            "PPO/ratio": 0.0,
        }

    def _train_step(self, epoch: int = 0) -> Dict[str, float]:
        mean_value_loss = torch.zeros(1, device=self.device)
        mean_value = torch.zeros(1, device=self.device)
        mean_surrogate_loss = torch.zeros(1, device=self.device)
        mean_ratio = torch.zeros(1, device=self.device)
        action_mean_std = torch.zeros(1, device=self.device)
        torso_action_mean_std = torch.zeros(1, device=self.device)
        arm_action_mean_std = torch.zeros(1, device=self.device)
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
                mean_value += batch.curr_values.mean()
                mean_surrogate_loss += surrogate_loss
                mean_ratio += torch.abs(ratio - 1.0).mean()
                action_mean_std += batch.curr_sigma.mean()
                if batch.curr_sigma.shape[-1] > 12:
                    # meaning that the robot has an arm
                    torso_action_mean_std += batch.curr_sigma[..., :12].mean()
                    arm_action_mean_std += batch.curr_sigma[..., 12:].mean()

        if self.min_std is not None or self.max_std is not None:
            self.actor.clamp_exploration_std(min=self.min_std, max=self.max_std)

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_value /= num_updates
        mean_surrogate_loss /= num_updates
        mean_ratio /= num_updates
        action_mean_std /= num_updates
        torso_action_mean_std /= num_updates
        arm_action_mean_std /= num_updates

        data = {
            "PPO/value": mean_value.item(),
            "PPO/value_loss": mean_value_loss.item(),
            "PPO/surrogate": mean_surrogate_loss.item(),
            "PPO/ratio": mean_ratio.item(),
            "PPO/action_std": action_mean_std.item(),
        }
        if batch.curr_sigma.shape[-1] > 12:
            data.update(
                {
                    "PPO/torso_action_std": torso_action_mean_std.item(),
                    "PPO/arm_action_std": arm_action_mean_std.item(),
                }
            )
        if mean_extra_losses is not None:
            mean_extra_losses = (mean_extra_losses / num_updates).cpu().numpy()
            data.update(self._get_extra_loss_info(mean_extra_losses))

        return data

    def _calc_extra_losses(self, batch: RolloutBuffer.Batch):
        if self.smooth_coef is not None:
            if batch.trajectory_masks is not None:
                action = split_trajectory(batch.curr_mu, batch.trajectory_lengths)
                smoothness = (
                    nn.functional.conv1d(  # convolution in time dimension
                        action.permute(1, 2, 0).flatten(0, 1).unsqueeze_(1),
                        self.smooth_conv,
                    )
                    .reshape(*action.shape[1:], -1)
                    .permute(2, 0, 1)
                )
                # smooth_loss = (
                #     self.smooth_coef * smoothness[batch.trajectory_masks[2:]].abs()
                # )
                # return ((smooth_loss * self.smooth_weight).mean(),)
                smooth_loss = self.smooth_coef * smoothness[batch.trajectory_masks[1:-1]].abs()
                return ((smooth_loss * self.smooth_weight).mean(),)
            else:
                raise NotImplementedError("Smooth loss is not implemented for non RNNs!")
        return ()

    def _get_extra_loss_info(self, mean_extra_losses):
        if self.smooth_coef is not None:
            return {"PPO/smoothness": mean_extra_losses[0].item()}
        return {}

    # def _eval_batch(self, batch: RolloutBuffer.Batch): ...

    # def _calc_surrogate_loss(self, batch: RolloutBuffer.Batch) -> torch.Tensor: ...

    def load_state_dict(self, state_dict):
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])
        self.optimizer.load_state_dict(state_dict["optimizer"])

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        """Override ``state_dict`` method in ``torch.nn.Module``

        .. warning::
            Currently ``state_dict()`` also accepts positional arguments for
            ``destination``, ``prefix`` and ``keep_vars`` in order. However,
            this is being deprecated and keyword arguments will be enforced in
            future releases.
            For details: https://pytorch.org/docs/master/generated/torch.nn.Module.html#torch.nn.Module.state_dict
        """
        return {
            "actor": self.actor.state_dict(
                destination=destination, prefix=prefix, keep_vars=keep_vars
            ),
            "critic": self.critic.state_dict(
                destination=destination, prefix=prefix, keep_vars=keep_vars
            ),
            "optimizer": self.optimizer.state_dict(),
        }

    def reset_optimizer(self):
        self.optimizer = optim.Adam(
            [
                {"params": self.actor.parameters(), "lr": self.learning_rate},
                {"params": self.critic.parameters(), "lr": self.critic_lr},
            ],
            lr=self.learning_rate,
        )
