from typing import Optional, Union

import numpy as np
import torch
import torch as th
from torch import nn

from thunder.nn import clone_net
from thunder.rl import GeneralActor, MultiQNet, soft_update

from .buffer import ReplayBuffer


class SAC:
    """
    Soft Actor-Critic (SAC)
    Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor,
    Args:
        learning_rate: learning rate for adam optimizer,
            the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        buffer_size: size of the replay buffer
        learning_starts: how many steps of the model to collect transitions for before learning starts
        batch_size: Minibatch size for each gradient update
        tau: the soft update coefficient ("Polyak update", between 0 and 1)
        gamma: the discount factor
        gradient_steps: How many gradient steps for each rollout
        entropy_coef: Entropy regularization coefficient. (Equivalent to
            inverse of reward scale in the original SAC paper.)  Controlling exploration/exploitation trade-off.
            Set it to 'auto' to learn it automatically (and 'auto_0.1' for using 0.1 as initial value)
        target_update_interval: update the target network every ``target_network_update_freq``
            gradient steps.
        target_entropy: target entropy when learning ``ent_coef`` (``ent_coef = 'auto'``)
        device: Device (cpu, cuda, ...) on which the code should be run.
    """

    Critic = MultiQNet

    def __init__(
        self,
        actor: GeneralActor,
        critic: MultiQNet,
        num_envs: int,
        learning_rate: float = 3e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        gradient_steps: int = 1,
        entropy_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        device: Union[th.device, str] = "cuda",
    ):
        self.device = th.device(device)
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.critic_target = clone_net(critic, False)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)

        self.learning_rate = learning_rate

        # Create and wrap the env if needed
        self.num_envs = num_envs

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.tau = tau
        self.gamma = gamma
        self.gradient_steps = gradient_steps

        self.target_entropy = target_entropy
        self.log_ent_coef = None  # type: Optional[th.Tensor]
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef = entropy_coef
        self.target_update_interval = target_update_interval

        self.replay_buffer = ReplayBuffer(self.buffer_size, self.device)
        self.transition = self.replay_buffer.Transition()

        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy == "auto":
            # automatically set target entropy if needed
            self.target_entropy = -float(self.actor.action_shape[-1])  # type: ignore
        else:
            # Force conversion
            # this will also throw an error for unexpected string
            self.target_entropy = float(self.target_entropy)

        # The entropy coefficient or entropy can be learned automatically
        # see Automating Entropy Adjustment for Maximum Entropy RL section
        # of https://arxiv.org/abs/1812.05905
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            # Default initial value of ent_coef when learned
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            self.log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(
                True
            )
            self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=self.learning_rate)
        else:
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
            self.ent_coef_tensor = th.tensor(float(self.ent_coef), device=self.device)

        self._obs = None
        self._action = None

    def update(self, warmup=False):
        if self.replay_buffer.num_samples < self.learning_starts:
            return {}
        # Switch to train mode (this affects batch norm / dropout)
        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for step, batch in enumerate(
            self.replay_buffer.sampler(self.batch_size, self.gradient_steps)
        ):
            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.act_log_prob(batch.actor_obs)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(
                    self.log_ent_coef * (log_prob + self.target_entropy).detach()
                ).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.act_log_prob(batch.next_obs)
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(
                    self.critic_target.evaluate(batch.next_critic_obs, next_actions),
                    dim=1,
                )
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = batch.rewards + ~batch.dones * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic.evaluate(batch.critic_obs, batch.actions)

            # Compute critic loss
            critic_loss = 0.5 * sum(
                nn.functional.mse_loss(current_q, target_q_values) for current_q in current_q_values
            )
            critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

            # Optimize the critic
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Min over all critic networks
            q_values_pi = th.cat(self.critic.evaluate(batch.critic_obs, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            # Update target networks
            if step % self.target_update_interval == 0:
                soft_update(self.critic_target, self.critic, self.tau)

        result = {
            "SAC/ent_coef": np.mean(ent_coefs),
            "SAC/actor_loss": np.mean(actor_losses),
            "SAC/critic_loss": np.mean(critic_losses),
        }
        if len(ent_coef_losses) > 0:
            result["SAC/ent_coef_loss"] = np.mean(ent_coef_losses)
        return result

    @torch.inference_mode()
    def act(
        self,
        observation,
    ) -> np.ndarray:
        """
        Sample an action according to the exploration policy.
        This is either done by sampling the probability distribution of the policy,
        or sampling a random action (from a uniform distribution over the action space)
        or by adding noise to the deterministic output.

        :param observation:
        :return: action to take in the environment
            and scaled action that will be stored in the replay buffer.
            The two differs when the action space is not normalized (bounds are not [-1, 1]).
        """
        # Select action randomly or according to policy
        if self.replay_buffer.num_samples < self.learning_starts:
            # Warmup phase
            actions = (
                np.random.rand(self.num_envs, self.actor.action_shape[-1]).astype(np.float32) * 2.0
                - 1.0
            )
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            obs = torch.as_tensor(observation, device=self.device)
            actions, _ = self.actor.act_stochastic(obs)
            actions = actions.cpu().numpy()

        self.transition.actor_obs = observation
        self.transition.actions = actions
        return actions

    @torch.inference_mode()
    def step(self, next_obs, rewards, dones, timeouts):
        self.transition.next_obs = next_obs
        self.transition.rewards = rewards
        self.transition.dones = dones
        self.transition.timeouts = timeouts
        self.replay_buffer.add_transition(self.transition)

    def state_dict(self):
        state_dict = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optim": self.actor_optim.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
        }
        if self.log_ent_coef is not None:
            state_dict["log_ent_coef"] = self.log_ent_coef
            state_dict["ent_coef_optimizer"] = self.ent_coef_optimizer.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])
        self.critic_target.load_state_dict(state_dict["critic1_target"])
        self.actor_optim.load_state_dict(state_dict["actor_optim"])
        self.critic_optim.load_state_dict(state_dict["critic1_optim"])
        if self.log_ent_coef is not None:
            self.log_ent_coef = state_dict["log_ent_coef"]
            self.ent_coef_optimizer.load_state_dict(state_dict["ent_coef_optimizer"])
        elif "log_ent_coef" in state_dict:
            raise ValueError("Loading a model with different entropy setting")
            raise ValueError("Loading a model with different entropy setting")
