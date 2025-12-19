import math
import warnings
from typing import Optional, Tuple, TypeVar

import torch
import torch.nn as nn
from thunder.nn.mapping import ACTIVATION_CLS_NAME

__all__ = ["_ConvNdBlock", "Conv1dBlock", "Conv2dBlock", "ResBasicBlock", "ResBottleneckBlock"]


T = TypeVar("T")
_scalar_or_tuple_any_t = Tuple[T, ...] | T
_scalar_or_tuple_1_t = Tuple[T] | T
_scalar_or_tuple_2_t = Tuple[T, T] | T
_size_any_t = _scalar_or_tuple_any_t[int]
_size_1_t = _scalar_or_tuple_1_t[int]
_size_2_t = _scalar_or_tuple_2_t[int]


class _ConvNdBlock(nn.Module):

    def __init__(
        self,
        in_shape: Tuple[int, ...],
        channels: Tuple[int, ...],
        kernel_sizes: Tuple[_size_any_t, ...],
        strides: Optional[Tuple[_size_any_t, ...]] = None,
        paddings: Optional[Tuple[_size_any_t, ...]] = None,
        dilations: Optional[Tuple[_size_any_t, ...]] = None,
        pool_kernel: Optional[Tuple[_size_any_t, ...]] = None,
        activation: str = "mish",
        gap: bool = False,
        activate_output: bool = True,
        dtype=None,
        device=None,
    ):
        super().__init__()
        self.factory_kwargs = {"device": device, "dtype": dtype}
        self.in_shape = in_shape
        self.in_channels: int = in_shape[0]
        self.in_features: int = math.prod(in_shape)
        # Convolutional layers
        self.channels = channels
        self.conv_nums: int = len(channels)
        self.kernel_sizes = kernel_sizes
        self.strides = tuple(1 for _ in range(self.conv_nums)) if strides is None else strides
        self.paddings = tuple(0 for _ in range(self.conv_nums)) if paddings is None else paddings
        self.dilations = tuple(1 for _ in range(self.conv_nums)) if dilations is None else dilations
        # Pooling
        self.pool_kernel = pool_kernel
        self.pool_nums: int = 0 if pool_kernel is None else len(pool_kernel)
        assert self.pool_nums < self.conv_nums, "Too many pooling layers."
        self.pool_interval: int = (
            self.conv_nums if pool_kernel is None else self.conv_nums // (self.pool_nums + 1)
        )
        # Output shape
        if gap:
            self.out_shape = (self.channels[-1], *(1 for _ in range(len(in_shape) - 1)))
            self.out_channels: int = self.channels[-1]
            self.out_features: int = self.out_channels
        else:
            self.out_shape = self._calc_out_shape(in_shape)
            self.out_channels: int = self.channels[-1]
            self.out_features: int = math.prod(self.out_shape)
        # Activation
        activation_name = ACTIVATION_CLS_NAME[activation.lower()]
        self.activation_cls = getattr(nn, activation_name)
        self.activate_output = activate_output

    @staticmethod
    def calc_conv_out_shape(
        in_shape: Tuple[int, ...],
        kernel_size: _size_any_t,
        stride: _size_any_t = 1,
        padding: _size_any_t = 0,
        dilation: _size_any_t = 1,
    ) -> Tuple[int, ...]:
        def __ensure_tuple(param, ndim):
            if isinstance(param, int):
                return (param,) * ndim
            if len(param) == 1:
                return param * ndim
            return param

        kernel_size = __ensure_tuple(kernel_size, len(in_shape))
        stride = __ensure_tuple(stride, len(in_shape))
        padding = __ensure_tuple(padding, len(in_shape))
        dilation = __ensure_tuple(dilation, len(in_shape))
        fn = lambda i, k, s, p, d: int((i + 2 * p - d * (k - 1) - 1) // s + 1)
        return tuple(map(fn, in_shape, kernel_size, stride, padding, dilation))

    def _calc_out_shape(self, in_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        out_shape = in_shape[1:]
        for i in range(self.conv_nums - 1):
            out_shape = self.calc_conv_out_shape(
                out_shape,
                self.kernel_sizes[i],
                self.strides[i],
                self.paddings[i],
                self.dilations[i],
            )
            if (
                (i + 1) % self.pool_interval == 0
                and i // self.pool_interval < self.pool_nums
                and self.pool_kernel is not None
            ):
                out_shape = self.calc_conv_out_shape(
                    out_shape,
                    self.pool_kernel[i // self.pool_interval],
                    stride=self.pool_kernel[i // self.pool_interval],
                )
        out_shape = self.calc_conv_out_shape(
            out_shape,
            self.kernel_sizes[-1],
            self.strides[-1],
            self.paddings[-1],
            self.dilations[-1],
        )
        return (self.channels[-1], *out_shape)


class Conv1dBlock(_ConvNdBlock):
    """Convolutional block with customizable layers and activation functions
    Args:
    """

    def __init__(
        self,
        in_shape: Tuple[int, int],
        channels: Tuple[int, ...] = (16, 8),
        kernel_sizes: Tuple[_size_1_t, ...] = (3, 5),
        strides: Optional[Tuple[_size_1_t, ...]] = None,
        paddings: Optional[Tuple[_size_1_t, ...]] = None,
        dilations: Optional[Tuple[_size_1_t, ...]] = None,
        pool_kernel: Optional[Tuple[_size_1_t, ...]] = None,
        activation: str = "mish",
        gap: bool = False,
        activate_output: bool = True,
        dtype=None,
        device=None,
    ):
        super().__init__(
            in_shape,
            channels,
            kernel_sizes,
            strides,
            paddings,
            dilations,
            pool_kernel,
            activation,
            gap,
            activate_output,
            dtype,
            device,
        )
        # Initialize network
        layers = []
        for i in range(self.conv_nums - 1):
            layers.extend(
                [
                    nn.Conv1d(
                        self.in_channels if i == 0 else self.channels[i - 1],
                        self.channels[i],
                        self.kernel_sizes[i],
                        self.strides[i],
                        self.paddings[i],
                        self.dilations[i],
                        **self.factory_kwargs,
                    ),
                    self.activation_cls(),
                ]
            )
            if (
                (i + 1) % self.pool_interval == 0
                and i // self.pool_interval < self.pool_nums
                and self.pool_kernel is not None
            ):
                layers.append(nn.MaxPool1d(kernel_size=pool_kernel[i // self.pool_interval]))
        layers.append(
            nn.Conv1d(
                self.in_channels if self.conv_nums == 1 else self.channels[self.conv_nums - 2],
                self.channels[self.conv_nums - 1],
                self.kernel_sizes[self.conv_nums - 1],
                self.strides[self.conv_nums - 1],
                self.paddings[self.conv_nums - 1],
                self.dilations[self.conv_nums - 1],
                **self.factory_kwargs,
            )
        )
        if gap:
            layers.append(nn.AdaptiveAvgPool1d((1)))
        if activate_output:
            layers.append(self.activation_cls())
        self.conv_block = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self, gain: float = 2.0) -> None:
        for layer in self.conv_block:
            if isinstance(layer, nn.Conv1d):
                nn.init.orthogonal_(layer.weight, math.sqrt(gain))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.conv_block(input)


class Conv2dBlock(_ConvNdBlock):
    """Convolutional block with customizable layers and activation functions
    Args:
    """

    def __init__(
        self,
        in_shape: Tuple[int, int, int],
        channels: Tuple[int, ...] = (16, 8),
        kernel_sizes: Tuple[_size_2_t, ...] = (3, 5),
        strides: Optional[Tuple[_size_2_t, ...]] = None,
        paddings: Optional[Tuple[_size_2_t, ...]] = None,
        dilations: Optional[Tuple[_size_2_t, ...]] = None,
        pool_kernel: Optional[Tuple[_size_2_t, ...]] = None,
        activation: str = "mish",
        gap: bool = False,
        activate_output: bool = True,
        dtype=None,
        device=None,
    ):
        super().__init__(
            in_shape,
            channels,
            kernel_sizes,
            strides,
            paddings,
            dilations,
            pool_kernel,
            activation,
            gap,
            activate_output,
            dtype,
            device,
        )
        # Initialize network
        layers = []
        for i in range(self.conv_nums - 1):
            layers.extend(
                [
                    nn.Conv2d(
                        self.in_channels if i == 0 else self.channels[i - 1],
                        self.channels[i],
                        self.kernel_sizes[i],
                        self.strides[i],
                        self.paddings[i],
                        self.dilations[i],
                        **self.factory_kwargs,
                    ),
                    self.activation_cls(),
                ]
            )
            if (
                (i + 1) % self.pool_interval == 0
                and i // self.pool_interval < self.pool_nums
                and self.pool_kernel is not None
            ):
                layers.append(nn.MaxPool2d(kernel_size=pool_kernel[i // self.pool_interval]))
        layers.append(
            nn.Conv2d(
                self.in_channels if self.conv_nums == 1 else self.channels[self.conv_nums - 2],
                self.channels[self.conv_nums - 1],
                self.kernel_sizes[self.conv_nums - 1],
                self.strides[self.conv_nums - 1],
                self.paddings[self.conv_nums - 1],
                self.dilations[self.conv_nums - 1],
                **self.factory_kwargs,
            )
        )
        if gap:
            layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        if activate_output:
            layers.append(self.activation_cls())
        self.conv_block = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self, gain: float = 2.0) -> None:
        for layer in self.conv_block:
            if isinstance(layer, nn.Conv2d):
                nn.init.orthogonal_(layer.weight, math.sqrt(gain))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.conv_block(input)


class ResBasicBlock(nn.Module):
    """Basic block in residual network.
    The basic residual convolution block, :math:`y = x + F(x)`.
    NB: For details: https://arxiv.org/abs/1512.03385v1
    NB: BatchNorm shouldn't used in RL
            http://www.deeprlhub.com/d/469-batch-norm
    """

    def __init__(
        self,
        in_channels: int,
        activation: str = "mish",
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        activation_name = ACTIVATION_CLS_NAME[activation.lower()]
        activation_instance = getattr(nn, activation_name)
        out_channels = in_channels
        straight_pass_layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                **factory_kwargs,
            ),
            activation_instance(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                **factory_kwargs,
            ),
        ]
        self.straight_pass = nn.Sequential(*straight_pass_layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        straight_output = self.straight_pass(input)
        output = straight_output + input

        return output


class ResBottleneckBlock(nn.Module):
    """Bottleneck block in residual network.
    Another residual convolution block, :math:`y = H(x) + F(x)`,
    where the :math:`H(x)` means use 1x1 kernel to process the image.
    NB: For details: https://arxiv.org/abs/1512.03385v1
    NB: BatchNorm and MaxPool shouldn't used in RL
            http://www.deeprlhub.com/d/469-batch-norm
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str = "mish",
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        activation_name = ACTIVATION_CLS_NAME[activation.lower()]
        activation_instance = getattr(nn, activation_name)
        if in_channels == out_channels:
            warnings.warn("The input channel should be different with the output channel.")
        straight_pass_layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                **factory_kwargs,
            ),
            activation_instance(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                **factory_kwargs,
            ),
        ]
        self.straight_pass = nn.Sequential(*straight_pass_layers)
        self.short_cut_pass = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            **factory_kwargs,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        straight_output = self.straight_pass(input)
        short_cut_output = self.short_cut_pass(input)
        return straight_output + short_cut_output
