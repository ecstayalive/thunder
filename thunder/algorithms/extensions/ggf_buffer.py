import torch

from ..ppo.buffer import RolloutBuffer


class GgfRolloutBuffer(RolloutBuffer):
    """This buffer is a custom buffer, and
    the default reward is a vector. One of the problems
    is modeled as an infinite mdp problem, and the other
    is modeled as a limited mdp problem.
    """

    def __init__(self, length: int, device: torch.device) -> None:
        super().__init__(length, device)
        self.ggf_weight = torch.tensor([0.64, 0.36], device=self.device)
        self._eps = torch.tensor([1e-8], device=self.device)

    def _lazy_init(self, t: RolloutBuffer.Transition) -> None:
        super()._lazy_init(t)
        self._norm_advantages_dims = tuple(i for i in range(self.advantages.dim() - 1))
        self.actions_log_prob = self.zeros_th(*self.rewards.shape[:-1], 1)
        self.advantages = self.zeros_th(*self.rewards.shape[:-1], 1)

    def compute_returns(
        self, gamma: float, lamda: float, normalize_adv: bool = True
    ) -> torch.Tensor:
        """Calculates GAE
        Calculates the mixture GAE. It is assumed that the
        reward and value of the system are composed of two
        parts.
        """
        advantage = 0

        with torch.inference_mode():
            for step in reversed(range(self.length)):
                mask = (
                    self.dones[step].logical_not().logical_or(self.timeouts[step])
                    * gamma
                )
                delta = (
                    self.rewards[step]
                    + mask * self.next_values[step]
                    - self.values[step]
                )
                advantage = delta + mask * lamda * advantage
                self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages
        self.advantages = self.returns - self.values
        # self.advantages, _ = torch.sort(self.advantages, dim=-1)
        # self.advantages = torch.sum(
        #     self.advantages * self.ggf_weight, dim=-1, keepdim=True
        # )
        if normalize_adv:
            std, mean = torch.std_mean(self.advantages)
            self.advantages = (self.advantages - mean) / torch.max(std, self._eps)
