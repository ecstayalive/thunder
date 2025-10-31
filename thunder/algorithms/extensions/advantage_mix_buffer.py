import torch

from thunder.rl.utils.rnn import (
    get_trajectory_lengths,
    get_trajectory_mask,
    split_trajectory,
    unpad_trajectory,
)

from ..ppo.buffer import RolloutBuffer


class AdvantageMixBuffer(RolloutBuffer):
    """This buffer is a custom buffer, and
    the default reward is a vector. One of the problems
    is modeled as an infinite mdp problem, and the other
    is modeled as a limited mdp problem.
    """

    def __init__(self, length: int, device: torch.device) -> None:
        super().__init__(length, device)

    def _lazy_init(self, t: RolloutBuffer.Transition):
        super()._lazy_init(t)
        self._norm_advantages_dims = tuple(i for i in range(self.advantages.dim() - 1))
        self.actions_log_prob = torch.zeros_like(self.actions)
        self._default_mixing_index = (
            torch.arange(1, self.length + 1, dtype=torch.float32, device=self.device)
            .view([-1, 1, 1])
            .expand([-1, *self.dones.shape[1:]])
        )
        self._last_mixing_index = self.zeros_th(*self.dones.shape[1:])
        self._last_dones = self.zeros_th(*self.dones.shape[1:])
        self._last_timeout = self.zeros_th(*self.dones.shape[1:])

    def compute_returns(self, gamma: float, lamda: float, normalize_adv: bool = True) -> None:
        """Calculates GAE
        Calculates the mixture GAE. It is assumed that the
        reward and value of the system are composed of two
        parts.
        Args:
            gamma: A parameter for reducing the summary of the reward.
                Determine the field of view length of the agent.
            lamda: Parameters used to estimate advantage in GAE
            normalize_adv: Whether to perform advantage regularization
        Property:
            beta: In mixed gradient advantage, the parameter
                used to mix the advantages of two tasks, the
                value is between 0 and 1
        """
        advantage = 0

        with torch.inference_mode():
            for step in reversed(range(self.length)):
                mask = self.dones[step].logical_not().logical_or(self.timeouts[step]) * gamma
                delta = self.rewards[step] + mask * self.next_values[step] - self.values[step]
                advantage = delta + mask * lamda * advantage
                self.returns[step] = advantage + self.values[step]
        separate_advantages = self.returns - self.values
        beta = self.calculate_beta()
        self.advantages = separate_advantages + beta * separate_advantages.flip((-1,))
        # Compute and normalize the advantages
        if normalize_adv:
            # std, mean = torch.std_mean(self.advantages)
            std, mean = torch.std_mean(self.advantages, dim=self._norm_advantages_dims)
            self.advantages = (self.advantages - mean) / (std + 1e-8)

    def calculate_beta(self, mixing_time_steps: int = 2000) -> torch.Tensor:
        """Calculate the coefficient of the mixing advantage"""
        reset_flag = torch.zeros_like(self.dones)
        reset_flag[1:] = self.dones[:-1]
        mixing_index = (
            self._last_mixing_index
            * (self._last_dones.logical_not().logical_or(self._last_timeout))
            + self._default_mixing_index
        )
        traj_len = get_trajectory_lengths(self.dones)
        traj_mask = get_trajectory_mask(traj_len)
        reset_flag_traj = split_trajectory(reset_flag, traj_len)
        mixing_index_traj = split_trajectory(mixing_index, traj_len)
        mixing_index_traj[:, reset_flag_traj.any(dim=0)] -= (
            mixing_index_traj[0, reset_flag_traj.any(dim=0)] - 1
        )
        mixing_index = unpad_trajectory(mixing_index_traj, traj_mask, self.length)
        self._last_mixing_index = mixing_index[-1]
        self._last_dones = self.dones[-1]
        self._last_timeout = self.timeouts[-1]
        beta = torch.clamp_max(mixing_index / mixing_time_steps, 1.0)
        return beta
