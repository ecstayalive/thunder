from typing import Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
from thunder.nn.functional import position_embedding_2d
from thunder.nn.modules import LinearBlock, _ConvNdBlock
from thunder.nn.modules.attention import *


class Perception(nn.Module):
    """
    Args:
        in_features: The number of expected features in the input `x`
        embed_features: The number of features of the residual connection,
            by default these features are in the tail of `x`.
        out_features: The number of expected features in the output `y`
        rnn_hidden_size: The number of features of the multi-layer
            long short-term (LSTM) RNN's hidden states `h`
        mlp_shape: The expected shape of the multi-layer perception block,
            for example, one tuple (256, 126, 10) stands that there are
            three hidden layer in MLP, which sizes are 256, 126, 10.
        rnn_num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
            would mean stacking two LSTMs together to form a `stacked LSTM`,
            with the second LSTM taking in outputs of the first LSTM and
            computing the final results. Default: 1
        rnn_batch_first: If ``True``, then the input and output tensors are provided
            as `(batch, seq, feature)` instead of `(seq, batch, feature)`.
            Note that this does not apply to hidden or cell states. See the
            Inputs/Outputs sections below for details.  Default: ``False``
        rnn_dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
            LSTM layer except the last layer, with dropout probability equal to
            :attr:`dropout`. Default: 0
        rnn_bidirectional: If ``True``, becomes a bidirectional LSTM. Default: ``False``
        rnn_proj_size: If ``> 0``, will use LSTM with projections of corresponding size. Default: 0
        activation: The type of activation function used in mlp and rnn
        activation_output: Whether the output needs to be activated
    Note:
        See https://datascience.stackexchange.com/questions/66594/activation-function-between-lstm-layers
        for detail about the activation between lstm and mlp. The conclusion is that
        there is no need to add a activation layer between lstm and mlp.
    Inputs: inputs, (h_0, c_0)
    Outputs: output, (lstm_h_n, lstm_c_n)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rnn_hidden_size: int,
        conv_head: _ConvNdBlock,
        mlp_shape: Iterator[int] = None,
        rnn_num_layers: int = 1,
        rnn_dropout: float = 0.0,
        rnn_proj_size: int = 0,
        activation: str = "mish",
        activate_output: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.conv_head = conv_head
        self.rnn = nn.LSTM(
            in_features - self.conv_head.in_features + self.conv_head.out_features,
            rnn_hidden_size,
            rnn_num_layers,
            dropout=rnn_dropout,
            proj_size=rnn_proj_size,
            **factory_kwargs,
        )
        mlp_in_features = rnn_hidden_size
        if rnn_proj_size > 0:
            mlp_in_features = rnn_proj_size
        self.mlp = LinearBlock(
            mlp_in_features, out_features, mlp_shape, activation, activate_output, **factory_kwargs
        )
        self.project = nn.Linear(self.conv_head.out_features, rnn_hidden_size)

    def forward(self, input: torch.Tensor, hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        input_shape = input.shape
        input1: torch.Tensor = input[..., : -self.conv_head.in_features]
        conv_input: torch.Tensor = (
            input[..., -self.conv_head.in_features :]
            .unflatten(-1, self.conv_head.in_shape)
            .flatten(end_dim=1)
        )
        conv_output: torch.Tensor = (
            self.conv_head(conv_input)
            .flatten(start_dim=-3, end_dim=-1)
            .unflatten(0, input_shape[:-1])
        )
        rnn_output, rnn_hidden = self.rnn(torch.cat([input1, conv_output], dim=-1), hx)
        output = self.mlp(rnn_output + self.project(conv_output))
        return output, rnn_hidden


class LinearMhaPerception(nn.Module):
    """_summary_
    Args:
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rnn_hidden_size: int,
        conv_head: _ConvNdBlock,
        mlp_shape: Iterator[int] = None,
        embed_dim: int = 32,
        num_heads: int = 2,
        rnn_num_layers: int = 1,
        rnn_dropout: float = 0.0,
        activation: str = "mish",
        activate_output: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.conv_head = conv_head
        c_out, h_out, w_out = self.conv_head.out_shape
        self.q_dim = in_features - self.conv_head.in_features
        self.embed_dim = embed_dim
        self.mha = MultiHeadLinearCrossAttention(
            embed_dim, num_heads, q_dim=self.q_dim, kv_dim=c_out
        )
        self.register_buffer("pos_embedding", position_embedding_2d(c_out, h_out, w_out))
        self.pos_embedding: torch.Tensor
        self.rnn = nn.LSTM(
            self.q_dim + self.embed_dim,
            rnn_hidden_size,
            rnn_num_layers,
            dropout=rnn_dropout,
            **factory_kwargs,
        )
        self.mlp = LinearBlock(
            rnn_hidden_size,
            out_features,
            mlp_shape,
            activation,
            activate_output,
            **factory_kwargs,
        )
        self.v_norm = nn.LayerNorm(c_out)
        self.project = nn.Linear(self.embed_dim, rnn_hidden_size)
        self.reset_parameters()

    def reset_parameters(self, gain: float = 2.0) -> None:
        nn.init.orthogonal_(self.project.weight, math.sqrt(gain))

    def forward(self, input: torch.Tensor, hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        L, B, _ = input.shape
        query: torch.Tensor = input[..., : self.q_dim]
        conv_input: torch.Tensor = (
            input[..., self.q_dim :].unflatten(-1, self.conv_head.in_shape).flatten(end_dim=1)
        )
        # [T*B, C, L, W] => [T*B, L*W, C]
        conv_output: torch.Tensor = self.conv_head(conv_input)
        conv_output = conv_output + self.pos_embedding
        kv = self.v_norm(conv_output.flatten(2).transpose(1, 2))
        values = self.mha(query.view(-1, 1, self.q_dim), kv).view(L, B, self.embed_dim)
        rnn_output, rnn_hidden = self.rnn(torch.cat([query, values], dim=-1), hx)
        output = self.mlp(rnn_output + self.project(values))
        return output, rnn_hidden


class BeliefPerception(nn.Module):
    """
    Args:
        in_features: The number of expected features in the input `x`
        embed_features: The number of features of the residual connection,
            by default these features are in the tail of `x`.
        out_features: The number of expected features in the output `y`
        rnn_hidden_size: The number of features of the multi-layer
            long short-term (LSTM) RNN's hidden states `h`
        mlp_shape: The expected shape of the multi-layer perception block,
            for example, one tuple (256, 126, 10) stands that there are
            three hidden layer in MLP, which sizes are 256, 126, 10.
        rnn_num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
            would mean stacking two LSTMs together to form a `stacked LSTM`,
            with the second LSTM taking in outputs of the first LSTM and
            computing the final results. Default: 1
        rnn_batch_first: If ``True``, then the input and output tensors are provided
            as `(batch, seq, feature)` instead of `(seq, batch, feature)`.
            Note that this does not apply to hidden or cell states. See the
            Inputs/Outputs sections below for details.  Default: ``False``
        rnn_dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
            LSTM layer except the last layer, with dropout probability equal to
            :attr:`dropout`. Default: 0
        rnn_bidirectional: If ``True``, becomes a bidirectional LSTM. Default: ``False``
        rnn_proj_size: If ``> 0``, will use LSTM with projections of corresponding size. Default: 0
        activation: The type of activation function used in mlp and rnn
        activation_output: Whether the output needs to be activated
    Note:
        See https://datascience.stackexchange.com/questions/66594/activation-function-between-lstm-layers
        for detail about the activation between lstm and mlp. The conclusion is that
        there is no need to add a activation layer between lstm and mlp.
    Inputs: inputs, (h_0, c_0)
    Outputs: output, (lstm_h_n, lstm_c_n)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rnn_hidden_size: int,
        conv_head: _ConvNdBlock,
        mlp_shape: Iterator[int] = None,
        gate_hidden_ratio: float = 0.2,
        rnn_num_layers: int = 1,
        rnn_dropout: float = 0.0,
        activation: str = "softsign",
        activate_output: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.conv_head = conv_head
        self.rnn = nn.LSTM(
            in_features - self.conv_head.in_features + self.conv_head.out_features,
            rnn_hidden_size,
            rnn_num_layers,
            dropout=rnn_dropout,
            **factory_kwargs,
        )
        self.gate_dim = int(gate_hidden_ratio * rnn_hidden_size)
        mlp_dim = rnn_hidden_size - self.gate_dim
        self.mlp = LinearBlock(
            mlp_dim, out_features, mlp_shape, activation, activate_output, **factory_kwargs
        )
        self.gate_project = nn.Sequential(
            nn.Linear(self.gate_dim, self.conv_head.out_features), nn.Sigmoid()
        )
        self.project = nn.Linear(self.conv_head.out_features, mlp_dim)

    def forward(self, input: torch.Tensor, hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        input_shape = input.shape
        input1: torch.Tensor = input[..., : -self.conv_head.in_features]
        conv_input: torch.Tensor = (
            input[..., -self.conv_head.in_features :]
            .flatten(end_dim=-2)
            .unflatten(-1, self.conv_head.in_shape)
        )
        conv_output: torch.Tensor = (
            self.conv_head(conv_input)
            .flatten(start_dim=-3, end_dim=-1)
            .unflatten(0, input_shape[:-1])
        )
        rnn_output, rnn_hidden = self.rnn(torch.cat([input1, conv_output], dim=-1), hx)
        gated = self.gate_project(rnn_output[..., -self.gate_dim :])
        output = self.mlp(rnn_output[..., : -self.gate_dim] + self.project(conv_output * gated))
        return output, rnn_hidden


class MhaBelief(nn.Module):
    """_summary_
    Args:
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rnn_hidden_size: int,
        conv_head: _ConvNdBlock,
        mlp_shape: Iterator[int] = None,
        embed_dim: int = 32,
        num_heads: int = 4,
        gate_hidden_ratio: float = 0.2,
        rnn_num_layers: int = 1,
        rnn_dropout: float = 0.0,
        activation: str = "softsign",
        activate_output: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.conv_head = conv_head
        c_out, h_out, w_out = self.conv_head.out_shape
        self.q_dim = in_features - self.conv_head.in_features
        self.embed_dim = embed_dim
        self.mha = MultiHeadLinearCrossAttention(
            embed_dim, num_heads, q_dim=self.q_dim, kv_dim=c_out
        )
        self.register_buffer("pos_embedding", position_embedding_2d(c_out, h_out, w_out))
        self.pos_embedding: torch.Tensor
        self.rnn = nn.LSTM(
            self.q_dim + self.embed_dim,
            rnn_hidden_size,
            rnn_num_layers,
            dropout=rnn_dropout,
            **factory_kwargs,
        )
        self.gate_dim = int(gate_hidden_ratio * rnn_hidden_size)
        self.mlp_dim = rnn_hidden_size - self.gate_dim
        self.mlp = LinearBlock(
            self.mlp_dim,
            out_features,
            mlp_shape,
            activation,
            activate_output,
            **factory_kwargs,
        )
        self.v_norm = nn.LayerNorm(c_out)
        self.gate_project = nn.Sequential(nn.Linear(self.gate_dim, self.embed_dim), nn.Sigmoid())
        self.project = nn.Linear(self.embed_dim, self.mlp_dim)
        self.reset_parameters()

    def reset_parameters(self, gain: float = 2.0) -> None:
        for module in self.gate_project.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, math.sqrt(gain))
        nn.init.orthogonal_(self.project.weight, math.sqrt(gain))

    def forward(self, input: torch.Tensor, hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        input_shape = input.shape
        query: torch.Tensor = input[..., : self.q_dim]
        conv_input: torch.Tensor = (
            input[..., self.q_dim :].flatten(end_dim=-2).unflatten(-1, self.conv_head.in_shape)
        )
        # [T*B, C, L, W] => [T*B, L*W, C]
        conv_output: torch.Tensor = self.conv_head(conv_input)
        conv_output = conv_output + self.pos_embedding
        kv = self.v_norm(conv_output.flatten(2).transpose(1, 2))
        values = self.mha(query.view(-1, 1, self.q_dim), kv).view(*input_shape[:-1], self.embed_dim)
        rnn_output, rnn_hidden = self.rnn(torch.cat([query, values], dim=-1), hx)
        gated = self.gate_project(rnn_output[..., -self.gate_dim :])
        output = self.mlp(rnn_output[..., : -self.gate_dim] + self.project(values * gated))
        return output, rnn_hidden


class SpatialBelief(nn.Module):
    """_summary_
    Args:
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rnn_hidden_size: int,
        conv_head: _ConvNdBlock,
        mlp_shape: Iterator[int] = None,
        gate_hidden_ratio: float = 0.2,
        rnn_num_layers: int = 1,
        rnn_dropout: float = 0.0,
        activation: str = "softsign",
        activate_output: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.conv_head = conv_head
        c_out, h_out, w_out = self.conv_head.out_shape
        self.rnn = nn.LSTM(
            in_features - self.conv_head.in_features + 4 * c_out,
            rnn_hidden_size,
            rnn_num_layers,
            dropout=rnn_dropout,
            **factory_kwargs,
        )
        self.gate_dim = int(gate_hidden_ratio * rnn_hidden_size)
        self.mlp_dim = rnn_hidden_size - self.gate_dim
        self.mlp = LinearBlock(
            self.mlp_dim,
            out_features,
            mlp_shape,
            activation,
            activate_output,
            **factory_kwargs,
        )
        self.spatial_attention = SpatialArgSoftmaxUncertainty(h_out, w_out)
        self.gate_project = nn.Sequential(nn.Linear(self.gate_dim, 2 * c_out), nn.Sigmoid())
        self.project = nn.Linear(2 * c_out, self.mlp_dim)

    def forward(self, input: torch.Tensor, hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        L, B, _ = input.shape
        conv_input: torch.Tensor = (
            input[..., -self.conv_head.in_features :]
            .unflatten(-1, self.conv_head.in_shape)
            .flatten(end_dim=1)
        )
        conv_output: torch.Tensor = self.conv_head(conv_input)  # [T*B, C, H, W]
        key_points, var_xy, peak = self.spatial_attention(conv_output)  # [T*B, C, 2]
        key_points = key_points.view(L, B, -1)  # [T, B, 2C]
        var_xy = var_xy.view(L, B, -1)  # [T, B, 2C]
        peak = peak.view(L, B, -1)  # [T, B, C]
        rnn_input1: torch.Tensor = input[..., : -self.conv_head.in_features]  # [T, B, C]
        rnn_output, rnn_hidden = self.rnn(torch.cat([rnn_input1, key_points, var_xy], dim=-1), hx)
        gated = self.gate_project(rnn_output[..., -self.gate_dim :])
        output = self.mlp(rnn_output[..., : -self.gate_dim] + self.project(key_points * gated))
        return output, rnn_hidden
