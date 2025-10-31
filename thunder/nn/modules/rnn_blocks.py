from typing import Iterator, Optional, Tuple, overload

import torch
import torch.nn as nn

from .linear_blocks import LinearBlock

__all__ = ["LstmMlp", "EmbedLstmMlp", "GruMlp", "RecurrentMlp"]


class LstmMlp(nn.Module):
    """A composite module of a multi-layer long short-term (LSTM) RNN
    and a multi-layer perception block. The document is based pytorch.
    Args:
        in_features: The number of expected features in the input `x`
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
        self.rnn = nn.LSTM(
            in_features,
            rnn_hidden_size,
            rnn_num_layers,
            True,
            rnn_batch_first,
            rnn_dropout,
            rnn_bidirectional,
            rnn_proj_size,
            **factory_kwargs,
        )
        mlp_in_features = rnn_hidden_size
        if rnn_proj_size > 0:
            mlp_in_features = rnn_proj_size
        if rnn_bidirectional:
            mlp_in_features *= 2
        self.mlp = LinearBlock(
            mlp_in_features,
            out_features,
            mlp_shape,
            activation,
            activate_output,
            **factory_kwargs,
        )

    def forward(
        self,
        input: torch.Tensor,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        rnn_output, rnn_hidden = self.rnn(input, hx)
        output = self.mlp(rnn_output)
        return output, rnn_hidden


class EmbedLstmMlp(nn.Module):
    """A composite module of a multi-layer long short-term (LSTM) RNN
    and a multi-layer perception block. The document is based pytorch.
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
        embed_type: The type of embedding used in the residual connection, "normal" | "shortcut"
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
        embed_features: int,
        out_features: int,
        rnn_hidden_size: int,
        mlp_shape: Iterator[int] = None,
        rnn_num_layers: int = 1,
        rnn_batch_first: bool = False,
        rnn_dropout: float = 0.0,
        rnn_bidirectional: bool = False,
        rnn_proj_size: int = 0,
        activation: str = "softsign",
        activate_output: bool = False,
        embed_type: str = "shortcut",
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.embed_features = embed_features
        match embed_type:
            case "normal":
                self.process = self._normal_process
                in_features = in_features - embed_features
            case "shortcut":
                self.process = self._shortcut_process

        self.rnn = nn.LSTM(
            in_features,
            rnn_hidden_size,
            rnn_num_layers,
            True,
            rnn_batch_first,
            rnn_dropout,
            rnn_bidirectional,
            rnn_proj_size,
            **factory_kwargs,
        )
        mlp_in_features = rnn_hidden_size + self.embed_features
        if rnn_proj_size > 0:
            mlp_in_features = rnn_proj_size
        if rnn_bidirectional:
            mlp_in_features *= 2
        self.mlp = LinearBlock(
            mlp_in_features,
            out_features,
            mlp_shape,
            activation,
            activate_output,
            **factory_kwargs,
        )

    def forward(self, input: torch.Tensor, hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        rnn_output, rnn_hidden = self.process(input, hx)
        output = self.mlp(torch.cat((rnn_output, input[..., -self.embed_features :]), dim=-1))
        return output, rnn_hidden

    def _normal_process(
        self, input: torch.Tensor, hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ):
        return self.rnn(input[..., : -self.embed_features], hx)

    def _shortcut_process(
        self, input: torch.Tensor, hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ):
        return self.rnn(input, hx)


class GruMlp(nn.Module):
    """A composite module of a gated recurrent unit (GRU) RNN
    and a multi-layer perception block. The document is based pytorch.
    Args:
        in_features: The number of expected features in the input `x`
        out_features: The number of expected features in the output `y`
        rnn_hidden_size: The number of features of the gated recurrent unit
            (LSTM) RNN's hidden states `h`
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
        rnn_proj_size: Enabled only when the `rnn_type=lstm`,
            if ``> 0``, will use LSTM with projections of corresponding size. Default: 0
        activation: The type of activation function used in mlp and rnn
        activation_output: Whether the output needs to be activated
    Inputs: inputs, h_0
    Outputs: output, gru_h_n
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rnn_hidden_size: int,
        mlp_shape: Iterator[int] = None,
        rnn_num_layers: int = 1,
        rnn_batch_first: bool = False,
        rnn_dropout: float = 0.0,
        rnn_bidirectional: bool = False,
        activation: str = "softsign",
        activate_output: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.rnn = nn.GRU(
            in_features,
            rnn_hidden_size,
            rnn_num_layers,
            True,
            rnn_batch_first,
            rnn_dropout,
            rnn_bidirectional,
            **factory_kwargs,
        )
        mlp_in_features = rnn_hidden_size
        if rnn_bidirectional:
            mlp_in_features *= 2
        self.mlp = LinearBlock(
            mlp_in_features,
            out_features,
            mlp_shape,
            activation,
            activate_output,
            **factory_kwargs,
        )

    def forward(
        self, input: torch.Tensor, hx: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rnn_output, rnn_hidden = self.rnn(input, hx)
        output = self.mlp(rnn_output)
        return output, rnn_hidden


class RecurrentMlp(nn.Module):
    """A composite module of recurrent neural network and a multi-layer
    perception block. The document is based pytorch.
    Args:
        rnn_type: the type of rnn, options: `lstm` or `gru`
        in_features: The number of expected features in the input `x`
        out_features: The number of expected features in the output `y`
        rnn_hidden_size: The number of features RNN's hidden states `h`
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
        activation: The type of activation function used in mlp and rnn
        activation_output: Whether the output needs to be activated
    """

    def __init__(
        self,
        rnn_type: str,
        in_features: int,
        out_features: int,
        rnn_hidden_size: int,
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
        mlp_in_features = rnn_hidden_size
        self.rnn_type = rnn_type
        match rnn_type:
            case "lstm":
                self.rnn = nn.LSTM(
                    in_features,
                    rnn_hidden_size,
                    rnn_num_layers,
                    True,
                    rnn_batch_first,
                    rnn_dropout,
                    rnn_bidirectional,
                    rnn_proj_size,
                    **factory_kwargs,
                )
                if rnn_proj_size > 0:
                    mlp_in_features = rnn_proj_size
            case "gru":
                assert (
                    rnn_proj_size == 0
                ), "The param 'rnn_proj_size' only works when the 'rnn_type' is 'lstm'."
                self.rnn = self.rnn = nn.GRU(
                    in_features,
                    rnn_hidden_size,
                    rnn_num_layers,
                    True,
                    rnn_batch_first,
                    rnn_dropout,
                    rnn_bidirectional,
                    **factory_kwargs,
                )
            case _:
                raise ValueError(f"Unsupported rnn type: {rnn_type}")

        if rnn_bidirectional:
            mlp_in_features *= 2
        self.mlp = LinearBlock(
            mlp_in_features,
            out_features,
            mlp_shape,
            activation,
            activate_output,
            **factory_kwargs,
        )

    @overload
    def forward(
        self, input: torch.Tensor, hx: Optional[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """ """
        ...

    @overload
    def forward(
        self, input: torch.Tensor, hx: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ """
        ...

    def forward(self, input: torch.Tensor, hx=None):
        rnn_output, rnn_hidden = self.rnn(input, hx)
        output = self.mlp(rnn_output)
        return output, rnn_hidden
