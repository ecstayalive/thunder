import numpy as np
import torch
import torch.nn as nn


class Normalization(nn.Module):
    def __init__(
        self,
        mean: torch.Tensor | np.ndarray,
        var: torch.Tensor | np.ndarray,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean))
        self.register_buffer("var", torch.as_tensor(var))
        self.mean: torch.Tensor
        self.var: torch.Tensor
        self.eps = torch.tensor([eps], dtype=self.mean.dtype, device=self.mean.device)
        self.std = torch.sqrt(torch.max(self.var, self.eps))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return (input - self.mean) / self.std

    @torch.jit.export
    def denormalize(self, input: torch.Tensor) -> torch.Tensor:
        return input * self.std + self.mean

    def extra_repr(self):
        features = self.mean.numel()
        return f"features={features}, eps={self.eps}"


class RunningNorm1d(nn.Module):
    """Applies Normalization over a 2D or 3D input.
    Method described in:
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = False,
        track_running_stats: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = nn.Parameter(torch.empty(num_features, **factory_kwargs))
            self.bias = nn.Parameter(torch.empty(num_features, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.register_buffer("running_mean", torch.zeros(num_features, **factory_kwargs))
        self.register_buffer("running_var", torch.ones(num_features, **factory_kwargs))
        self.register_buffer("running_std", torch.ones(num_features, **factory_kwargs))
        self.running_mean: torch.Tensor
        self.running_var: torch.Tensor
        self.running_std: torch.Tensor
        self.register_buffer(
            "num_data_tracked",
            torch.tensor(
                0,
                dtype=torch.long,
                **{k: v for k, v in factory_kwargs.items() if k != "dtype"},
            ),
        )
        self.num_data_tracked: torch.Tensor
        self.reset_parameters()

    def reset_running_stats(self) -> None:
        self.running_mean.zero_()  # type: ignore[union-attr]
        self.running_var.fill_(1)  # type: ignore[union-attr]
        self.running_std.fill_(1)  # type: ignore[union-attr]
        self.num_data_tracked.zero_()  # type: ignore[union-attr,operator]

    def reset_parameters(self) -> None:
        self.reset_running_stats()
        if self.affine:
            nn.init.ones_(tensor=self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.track_running_stats:
            self._update(input)
        if self.affine:
            return (input - self.running_mean) / self.running_std * self.weight + self.bias
        return (input - self.running_mean) / self.running_std

    def _update(self, input: torch.Tensor):
        batch_mean = torch.mean(input, dim=input.dim_order()[:-1])
        batch_var = torch.var(input, dim=input.dim_order()[:-1])
        batch_size = input.size()[:-1].numel()
        delta = torch.square(self.running_mean - batch_mean)
        total_data_size = self.num_data_tracked + batch_size
        w1, w2 = self.num_data_tracked / total_data_size, batch_size / total_data_size
        with torch.inference_mode(mode=False):
            self.running_mean = self.running_mean * w1 + batch_mean * w2
            self.running_var = self.running_var * w1 + batch_var * w2 + delta * w1 * w2
            # std is not unbiased estimation
            self.running_std = torch.sqrt(self.running_var + self.eps)
            # self.std = np.sqrt(self.var * tot_count / (tot_count - 1) + self.eps)
            self.num_data_tracked = total_data_size

    def extra_repr(self):
        return (
            "{num_features}, eps={eps}, affine={affine}, "
            "track_running_stats={track_running_stats}".format(**self.__dict__)
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.track_running_stats = not mode

    def tracking(self, mode: bool = True):
        self.track_running_stats = mode
