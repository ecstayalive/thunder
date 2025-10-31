import math
from typing import Iterable, Optional, Tuple, overload

import torch
from torch import nn

from thunder.nn import LinearBlock

LOG_SQRT_2PI = math.log(math.sqrt(2 * math.pi))
ENTROPY_BIAS = 0.5 + 0.5 * math.log(2 * math.pi)

__all__ = [
    "Distribution",
    "Gaussian",
    "ConsistentGaussian",
    "StateDependentGaussian",
    "GeneralizedSDE",
    "SDE",
    "make_distribution",
]


class Distribution(nn.Module):
    """For deep reinforcement learning, the output of the last
    layer is often a probability distribution. Therefore, the
    significance of this class is to use a network to output
    relevant information about the action distribution after
    obtaining the input embedding encoding or action dimensions.
    This class is the base class for all such probability distribution
    transformation networks and is responsible for the most basic
    information processing.

    Args:
        action_dim: Dimension of the action

    """

    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.action_dim = action_dim

    @property
    def action_shape(self):
        return [
            self.action_dim,
        ]

    @torch.jit.export
    def determine(self, latent: torch.Tensor) -> torch.Tensor:
        """After the embedded representation of the observation
        is known, use this representation to generate a deterministic
        action.

        Returns:
            Deterministic action.
        """
        raise NotImplementedError

    def forward(self, latent: torch.Tensor) -> torch.Tensor | Tuple[torch.Tensor, ...]:
        """After the embedded representation of the observation
        is known, it is converted into information related to
        the action distribution, and a multi-head kernel network
        is used to convert it into information related to the
        action distribution.

        Returns:
            Related parameters of action distribution.
        """
        raise NotImplementedError

    def sample_log_prob(
        self, latent: torch.Tensor, return_joint_log_prob: bool = True
    ) -> Tuple[torch.Tensor, ...]:
        """After the embedded representation of the observation
        is known, this function will convert the embedding
        representation into relevant parameters of the action
        distribution, and perform action parameter sampling in
        the action distribution.
        Args:
            latent: The embedded representation of the observation
            return_overall_log_prob: Whether to return the overall probability information
                of the action. For example, when the action has 12 dimensions, each dimension
                conforms to a Gaussian distribution. When this value is set to True, the
                overall probability of the action is returned. Otherwise, probability of
                each dimension of the action is returned.
        Returns:
            Parameters of action distribution, the sample action and the probability of this action.
        """
        raise NotImplementedError

    @overload
    def calc_log_prob(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        sample: torch.Tensor,
        return_joint_log_prob: bool = True,
    ) -> torch.Tensor:
        """"""
        ...

    def calc_log_prob(self, *args, **kwargs) -> torch.Tensor:
        """
        After given the parameters of the action distribution
        and the sampling point, calculate the logarithmic value
        of the probability of the sampling point. This function
        needs to be implemented for each probability distribution
        class that inherits `Distribution`. Since the representation
        parameters of each distribution cannot be determined, this
        method is not implemented in the base class.
        """
        raise NotImplementedError

    @overload
    def calc_entropy(self, std: torch.Tensor, return_joint_entropy: bool = True) -> torch.Tensor:
        """ """
        ...

    def calc_entropy(self, *args, **kwargs) -> torch.Tensor:
        """After given the parameters of the action distribution
        and the sampling point, calculate the entropy of the action
        distribution. This function needs to be implemented for each
        probability distribution class that inherits `Distribution`.
        Since the representation parameters of each distribution
        cannot be determined, this method is not implemented in the
        base class.
        """
        raise NotImplementedError


class GaussianBase(Distribution):
    """This class is the base class for all action distributions
    based on Gaussian distributions. It defines the general
    behavior when the class is a Gaussian action distribution.
    """

    def __init__(self, action_dim, squash=False):
        super().__init__(action_dim)
        self.squash = squash

    def calc_log_prob(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        sample: torch.Tensor,
        return_joint_log_prob: bool = True,
    ) -> torch.Tensor:
        if self.squash:
            raise NotImplementedError
        return self._g_log_prob(mean, std, sample, return_joint_log_prob)

    def calc_entropy(self, std: torch.Tensor, return_joint_entropy: bool = True) -> torch.Tensor:
        if self.squash:
            raise NotImplementedError
        return self._g_entropy(std, return_joint_entropy)

    @staticmethod
    def _g_log_prob(
        mean: torch.Tensor,
        std: torch.Tensor,
        sample: torch.Tensor,
        return_joint_log_prob: bool = True,
    ) -> torch.Tensor:
        """Given the mean and sampling point of a multivariate
        independent Gaussian distribution, calculate the probability
        of that point.
        Args:
            mean: The mean of the multivariate independent Gaussian distribution
            std: The standard variance of the multivariate independent Gaussian distribution
            sample: The sample point
            return_overall_log_prob: Whether to return the overall information,
                for example, the parameter of the multivariate independent
                Gaussian distribution is 3. When the value is true, a value
                is returned representing the product of three independent
                probabilities.
        """
        log_prob = -((sample - mean) ** 2) / (2 * std**2) - std.log() - LOG_SQRT_2PI
        return torch.sum(log_prob, dim=-1, keepdim=True) if return_joint_log_prob else log_prob

    @staticmethod
    def _g_entropy(std: torch.Tensor, return_joint_entropy: bool = True) -> torch.Tensor:
        entropy = std.log() + ENTROPY_BIAS
        return torch.sum(entropy, dim=-1, keepdim=True) if return_joint_entropy else entropy

    def _g_sample_log_prob(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        return_joint_log_prob: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples from the given Gaussian action distribution and
        returns the logarithm of the sampling probability
        """
        eps = torch.normal(
            torch.zeros_like(mean, dtype=mean.dtype, device=mean.device),
            torch.ones_like(std, dtype=std.dtype, device=std.device),
        )
        sample = mean + eps * std
        log_prob = self._g_log_prob(mean, std, sample, return_joint_log_prob)
        if self.squash:
            sample = sample.tanh()
            log_prob = log_prob - torch.log((1 - sample.pow(2)) + 1e-8).sum(dim=-1, keepdim=True)
        return sample, log_prob


class Gaussian(GaussianBase):
    """The most basic Gaussian distribution, where its
    standard deviation is a learnable parameter.
    """

    def __init__(
        self,
        action_dim: int,
        squash: bool = False,
        init_std: float = 1.0,
        dtype=None,
        device=None,
    ):
        super().__init__(action_dim, squash=squash)
        factory_kwargs = {"dtype": dtype, "device": device}
        self.std = nn.Parameter(init_std * torch.ones(action_dim, **factory_kwargs))

    @torch.jit.ignore
    def forward(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return latent, self.std.repeat(*latent.shape[:-1], 1)

    @torch.jit.export
    def determine(self, latent: torch.Tensor) -> torch.Tensor:
        return latent

    def sample_log_prob(
        self, latent: torch.Tensor, return_joint_log_prob: bool = True
    ) -> Tuple[torch.Tensor, ...]:
        mean, std = self(latent)
        return mean, std, *self._g_sample_log_prob(mean, std, return_joint_log_prob)

    def set_std(self, std: float | torch.Tensor) -> None:
        if isinstance(std, float):
            std = std * torch.ones(self.action_dim, dtype=self.std.dtype, device=self.std.device)
        assert std.shape == self.std.shape
        with torch.no_grad():
            self.std.data.copy_(std)

    def clamp_std(self, min=None, max=None, indices: slice = None):
        self.std.data[indices].clamp_(min=min, max=max)


class ConsistentGaussian(Gaussian):
    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        shape: Iterable[int] = None,
        activation: str = "softsign",
        squash: bool = False,
        init_std: float = 1.0,
        dtype=None,
        device=None,
    ) -> None:
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__(action_dim=action_dim, squash=squash, init_std=init_std, **factory_kwargs)
        self.mean_head = LinearBlock(
            latent_dim,
            action_dim,
            shape,
            activation,
            **factory_kwargs,
        )

    @torch.jit.export
    def determine(self, latent: torch.Tensor) -> torch.Tensor:
        return self.mean_head(latent)

    @torch.jit.ignore
    def forward(self, latent: torch.Tensor):
        return super().forward(self.mean_head(latent))


class StateDependentGaussian(GaussianBase):
    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        shape: Tuple,
        activation: str = "softsign",
        squash: bool = False,
        dtype=None,
        device=None,
    ):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__(action_dim, squash=squash)
        self.mean_head = LinearBlock(latent_dim, action_dim, shape, activation, **factory_kwargs)
        self.log_std_head = LinearBlock(latent_dim, action_dim, shape, activation, **factory_kwargs)
        self.min_log_std, self.max_log_std = -20.0, 0.5

    @torch.jit.export
    def determine(self, latent: torch.Tensor) -> torch.Tensor:
        return self.mean_head(latent)

    def forward(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        log_std = self.log_std_head(latent).clamp(self.min_log_std, self.max_log_std)
        return self.mean_head(latent), log_std.exp()


class SDEBase(GaussianBase):
    def __init__(
        self,
        latent_dim,
        action_dim,
        weighted=True,
        init_weight=1.0,
        biased=False,
        init_bias=1.0,
    ):
        super().__init__(action_dim, False)
        assert weighted or biased
        self.latent_dim, self.action_dim = latent_dim, action_dim
        self.weighted, self.biased = weighted, biased
        if self.weighted:
            self.log_weight = nn.Parameter(
                torch.ones(self.latent_dim, self.action_dim) * math.log(init_weight)
            )
        if self.biased:
            self.log_bias = nn.Parameter(torch.ones(action_dim) * math.log(init_bias))
        self.exploration_weights = self.exploration_bias = None
        self.min_log_std, self.max_log_std = -20.0, 0.0
        self._sde_count = 0
        self.eps = 1e-6

    def _sde(self, latent):
        self._sde_count += 1
        exploration = 0
        if self.weighted:
            exploration += latent.unsqueeze(1).bmm(self.exploration_weights).squeeze(1)
        if self.biased:
            exploration += self.exploration_bias
        return exploration

    def _sde_std(self, latent):
        var = 0
        if self.weighted:
            var = latent**2 @ self.log_weight.exp() ** 2
        if self.biased:
            var += (self.log_bias.exp() ** 2).repeat(*latent.shape[:-1], 1)
        return torch.sqrt(var + self.eps)

    def _sde_resample(self, num_parallels: int):
        self._sde_count = 0
        if self.weighted:
            self.log_weight.data.clamp_(self.min_log_std, self.max_log_std)
            self.exploration_weights = torch.distributions.Normal(
                torch.zeros_like(self.log_weight), self.log_weight.exp()
            ).rsample(torch.Size((num_parallels,)))
        if self.biased:
            self.log_bias.data.clamp_(self.min_log_std, self.max_log_std)
            self.exploration_bias = torch.distributions.Normal(
                torch.zeros_like(self.log_bias), self.log_bias.exp()
            ).rsample(torch.Size((num_parallels,)))


class GeneralizedSDE(SDEBase):
    def __init__(
        self,
        latent_dim,
        action_dim,
        backward=False,
        sample_intv=1,
        weighted=True,
        init_weight=1.0,
        biased=False,
        init_bias=1.0,
    ):
        super().__init__(latent_dim, action_dim, weighted, init_weight, biased, init_bias)
        self.activation = nn.Tanh()
        self.backward = backward
        self.num_parallels = None
        self.mean_head = nn.Linear(latent_dim, action_dim)
        self._sample_intv = sample_intv

    def forward(self, latent):
        latent = self.activation(latent)
        return self.mean_head(latent), self._sde_std(latent)

    @torch.jit.export
    def determine(self, latent):
        return self.mean_head(self.activation(latent))

    def sample_log_prob(self, latent):
        if self.exploration_weights is None or self._sde_count >= self._sample_intv:
            self.num_parallels = latent.shape[0]
            self.resample_weights()
        latent = self.activation(latent)
        mean = self.mean_head(latent)

        if not self.backward:
            latent = latent.detach()
        action = mean + self._sde(latent)
        std = self._sde_std(latent)
        log_prob = self._g_log_prob(mean, std, action)
        return mean, std, action, log_prob

    def resample_weights(self):
        self._sde_resample(self.num_parallels)


class SDE(SDEBase):
    def __init__(
        self,
        latent_dim,
        obs_dim,
        action_dim,
        activation_fn,
        sample_intv=1,
        weighted=True,
        init_weight=1.0,
        biased=False,
        init_bias=1.0,
    ):
        super().__init__(obs_dim, action_dim, weighted, init_weight, biased, init_bias)
        self.activation = activation_fn()
        self.num_parallels = None
        self.mean_head = nn.Linear(latent_dim, action_dim)
        self._sample_intv = sample_intv

    def forward(self, latent):
        latent, obs = latent
        action_mean = self.mean_head(self.activation(latent))
        return action_mean, self._sde_std(obs.tanh())

    @torch.jit.export
    def determine(self, latent):
        return self.mean_head(self.activation(latent))

    def sample_log_prob(self, latent):
        latent, obs = latent
        if self.exploration_weights is None or self._sde_count >= self._sample_intv:
            self.num_parallels = latent.shape[0]
            self.resample_weights()
        mean = self.mean_head(self.activation(latent))

        obs = obs.tanh()
        action = mean + self._sde(obs)
        std = self._sde_std(obs)
        log_prob = self._g_log_prob(mean, std, action)
        return mean, std, action, log_prob

    def resample_weights(self):
        self._sde_resample(self.num_parallels)


class GmmBase(Distribution):
    def __init__(
        self,
        num_components: int,
        dtype=None,
        device=None,
    ):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()

    def determine(self, latent): ...

    def forward(self, latent) -> Tuple[torch.Tensor, ...]: ...

    def sample_log_prob(self, latent) -> Tuple[torch.Tensor, ...]: ...

    def resample_weights(self):
        """ """
        ...

    def calc_log_prob(self, *args, **kwargs) -> torch.Tensor:
        """ """
        raise NotImplementedError

    def calc_entropy(self, *args, **kwargs) -> torch.Tensor:
        """ """
        raise NotImplementedError


def make_distribution(
    action_dim: int,
    latent_dim: int = None,
    obs_dim: int = None,
    shape: Optional[Tuple] = None,
    activation="relu",
    type="gaussian",
    **kwargs,
):
    if shape is None:
        shape = []
    type = type.lower()
    if type == "sde":
        return SDE(latent_dim, obs_dim, action_dim, activation, **kwargs)
    if type == "gsde":
        return GeneralizedSDE(latent_dim, action_dim, activation, **kwargs)
    if type == "sdv":
        return StateDependentGaussian(latent_dim, action_dim, shape, activation, **kwargs)
    if type != "gaussian":
        raise ValueError(f"Unknown dist type {type}")
    if latent_dim is None:
        return Gaussian(action_dim, **kwargs)
    return ConsistentGaussian(latent_dim, action_dim, shape, activation, **kwargs)
