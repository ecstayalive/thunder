from typing import Iterator, Optional, Tuple

import torch
import torch.nn as nn
from thunder.nn.modules import LinearBlock, MultiHeadAttention, _ConvNdBlock


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
        rnn_num_layers: int = 1,
        rnn_batch_first: bool = False,
        rnn_dropout: float = 0.0,
        rnn_bidirectional: bool = False,
        rnn_proj_size: int = 0,
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
            True,
            rnn_batch_first,
            rnn_dropout,
            rnn_bidirectional,
            rnn_proj_size,
            **factory_kwargs,
        )
        mlp_in_features = rnn_hidden_size + self.conv_head.out_features
        if rnn_proj_size > 0:
            mlp_in_features = rnn_proj_size + self.conv_head.out_features
        if rnn_bidirectional:
            mlp_in_features = 2 * rnn_hidden_size + self.conv_head.out_features
        self.mlp = LinearBlock(
            mlp_in_features,
            out_features,
            mlp_shape,
            activation,
            activate_output,
            **factory_kwargs,
        )
        self.gate = LinearBlock(
            rnn_hidden_size,
            self.conv_head.out_features,
            mlp_shape,
            activation,
            activate_output,
            **factory_kwargs,
        )

    def forward(self, input: torch.Tensor, hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        seq_len, batch_size, _ = input.shape
        input1: torch.Tensor = input[..., : -self.conv_head.in_features]
        conv_input: torch.Tensor = (
            input[..., -self.conv_head.in_features :]
            .unflatten(-1, self.conv_head.in_shape)
            .flatten(end_dim=1)
        )
        conv_output: torch.Tensor = (
            self.conv_head(conv_input)
            .flatten(start_dim=-3, end_dim=-1)
            .unflatten(0, (seq_len, batch_size))
        )
        rnn_output, rnn_hidden = self.rnn(torch.cat([input1, conv_output], dim=-1), hx)
        gated_features = nn.functional.sigmoid(self.gate(rnn_output))
        output = self.mlp(torch.cat((rnn_output, conv_output * gated_features), dim=-1))
        return output, rnn_hidden


class AttentionPerception(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rnn_hidden_size: int,
        conv_head: _ConvNdBlock,
        mlp_shape: Iterator[int] = None,
        embed_dim: int = 32,
        num_heads: int = 4,
        rnn_num_layers: int = 1,
        # rnn_batch_first: bool = False,
        rnn_dropout: float = 0.0,
        rnn_bidirectional: bool = False,
        rnn_proj_size: int = 0,
        activation: str = "softsign",
        activate_output: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.conv_head = conv_head
        self.q_dim = in_features - self.conv_head.in_features
        self.embed_dim = embed_dim
        self.mha = MultiHeadAttention(
            embed_dim,
            num_heads,
            q_dim=self.q_dim,
            k_dim=self.conv_head.out_channels,
            v_dim=self.conv_head.out_channels,
            is_causal=False,
        )

        self.rnn = nn.LSTM(
            self.q_dim + self.embed_dim,
            rnn_hidden_size,
            rnn_num_layers,
            True,
            # rnn_batch_first,
            rnn_dropout,
            rnn_bidirectional,
            rnn_proj_size,
            **factory_kwargs,
        )
        mlp_in_features = rnn_hidden_size + self.embed_dim
        if rnn_proj_size > 0:
            mlp_in_features = rnn_proj_size + self.embed_dim
        if rnn_bidirectional:
            mlp_in_features = 2 * rnn_hidden_size + self.embed_dim
        self.mlp = LinearBlock(
            mlp_in_features,
            out_features,
            mlp_shape,
            activation,
            activate_output,
            **factory_kwargs,
        )
        self.gate = LinearBlock(
            rnn_hidden_size,
            self.embed_dim,
            mlp_shape,
            activation,
            activate_output,
            **factory_kwargs,
        )

    def forward(self, input: torch.Tensor, hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        seq_len, batch_size, _ = input.shape
        query: torch.Tensor = input[..., : -self.conv_head.in_features]
        conv_input: torch.Tensor = (
            input[..., -self.conv_head.in_features :]
            .unflatten(-1, self.conv_head.in_shape)
            .flatten(end_dim=1)
        )
        # [T*B, C, L, W] => [T*B, L*W, C]
        conv_output: torch.Tensor = self.conv_head(conv_input)
        _, C, L, W = conv_output.shape
        conv_output = conv_output.permute(0, 2, 3, 1).view(-1, L * W, C)
        values: torch.Tensor = self.mha(query.view(-1, 1, self.q_dim), conv_output, conv_output)
        values = values.view(seq_len, batch_size, self.embed_dim)
        rnn_output, rnn_hidden = self.rnn(torch.cat([query, values], dim=-1), hx)
        gated_features = nn.functional.sigmoid(self.gate(rnn_output))
        output = self.mlp(torch.cat((rnn_output, values * gated_features), dim=-1))
        return output, rnn_hidden
