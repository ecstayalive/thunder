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
        self, num_features: int, eps: float = 1e-5, affine: bool = False, device=None, dtype=None
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
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

    def forward(
        self, input: torch.Tensor, update_stats: bool = False, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if update_stats:
            self.update(input, mask=mask)
        return self.normalize(input)

    def normalize(self, input: torch.Tensor) -> torch.Tensor:
        if self.affine:
            return (input - self.running_mean) / self.running_std * self.weight + self.bias
        return (input - self.running_mean) / self.running_std

    @torch.no_grad()
    def update(self, input: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        if input.shape[-1] != self.num_features:
            raise ValueError(
                f"expected input with {self.num_features} features, got {input.shape[-1]}"
            )

        flat_input = input.reshape(-1, self.num_features)
        if mask is not None:
            mask = mask.bool()
            if mask.shape == (*input.shape[:-1], 1):
                mask = mask.squeeze(-1)
            if mask.shape != input.shape[:-1]:
                raise ValueError(
                    f"expected mask shape {input.shape[:-1]} or {(*input.shape[:-1], 1)}, "
                    f"got {mask.shape}"
                )
            flat_input = flat_input[mask.reshape(-1)]

        batch_size = flat_input.shape[0]
        if batch_size == 0:
            return

        flat_input = flat_input.to(dtype=self.running_mean.dtype)
        batch_mean = flat_input.mean(dim=0)
        batch_var = flat_input.var(dim=0, unbiased=False)
        batch_count = torch.as_tensor(
            batch_size, device=self.num_data_tracked.device, dtype=torch.long
        )
        total_count = self.num_data_tracked + batch_count

        old_count = self.num_data_tracked.to(dtype=self.running_mean.dtype)
        new_count = batch_count.to(dtype=self.running_mean.dtype)
        total_count_f = total_count.to(dtype=self.running_mean.dtype)
        w1 = old_count / total_count_f
        w2 = new_count / total_count_f

        delta = self.running_mean - batch_mean
        new_mean = self.running_mean * w1 + batch_mean * w2
        new_var = self.running_var * w1 + batch_var * w2 + delta.square() * w1 * w2

        self.running_mean.copy_(new_mean)
        self.running_var.copy_(new_var)
        self.running_std.copy_(torch.sqrt(new_var + self.eps))
        self.num_data_tracked.copy_(total_count)

    def extra_repr(self):
        return f"{self.num_features}, eps={self.eps}, affine={self.affine}"


class DictRunningNorm1d(nn.Module):
    """Applies RunningNorm1d to a dictionary of tensors."""

    def __init__(self, specs: dict[str, int], eps=1e-5):
        super().__init__()
        self.keys = tuple(specs.keys())
        self.norms = nn.ModuleDict({key: RunningNorm1d(dim, eps=eps) for key, dim in specs.items()})

    @torch.no_grad()
    def update(self, obs):
        for key in self.keys:
            self.norms[key].update(obs.get(key, obs[key]))

    def forward(self, obs):
        obs = dict(obs)
        for key in self.keys:
            obs[key] = self.norms[key](obs.get(key, obs[key]))
        return obs
