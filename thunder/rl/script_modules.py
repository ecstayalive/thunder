from typing import Optional, Tuple

import torch
import torch.nn as nn
from thunder.models import BeliefPerception
from thunder.nn import (
    GruMlp,
    LinearBlock,
    LstmMlp,
    Normalization,
    RecurrentMlp,
    RunningNorm1d,
)
from thunder.rl import Distribution

__all__ = ["ScriptNet", "ScriptGeneralActor", "ScriptSeqEncoderActor"]


class ScriptNet(nn.Module):
    def __init__(
        self,
        kernel: LinearBlock | LstmMlp | GruMlp,
        normalizer: Optional[nn.Module] = None,
        kernel_type: str = "mlp",
    ) -> None:
        super().__init__()
        self.normalizer = normalizer
        self.kernel = kernel
        self.kernel_type = kernel_type
        if isinstance(kernel, (LstmMlp, BeliefPerception)):
            if self.normalizer is not None:
                self.forward = self.forward_lstm_with_norm
            else:
                self.forward = self.forward_lstm
        elif isinstance(kernel, GruMlp):
            if self.normalizer is not None:
                self.forward = self.forward_gru_with_norm
            else:
                self.forward = self.forward_gru
        else:
            if self.normalizer is not None:
                self.forward = self.forward_with_norm

    def forward(self, input: torch.Tensor):
        return self.kernel(input)

    def forward_with_norm(self, input: torch.Tensor):
        return self.kernel(self.normalizer(input))

    def forward_lstm(
        self,
        input: torch.Tensor,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        return self.kernel(input, hx)

    def forward_lstm_with_norm(
        self,
        input: torch.Tensor,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        return self.kernel(self.normalizer(input), hx)

    def forward_gru(self, input: torch.Tensor, hx: Optional[torch.Tensor] = None):
        return self.kernel(input, hx)

    def forward_gru_with_norm(self, input: torch.Tensor, hx: Optional[torch.Tensor] = None):
        return self.kernel(self.normalizer(input), hx)


class ScriptGeneralActor(nn.Module):
    def __init__(
        self,
        encoder: LinearBlock | LstmMlp | GruMlp | RecurrentMlp | BeliefPerception,
        action_decoder: Distribution,
        normalizer: Normalization | RunningNorm1d,
        denormalizer: Normalization | RunningNorm1d,
        module_type: str = "mlp",
    ) -> None:
        super().__init__()
        self.normalizer = normalizer
        self.encoder = encoder
        self.action_decoder = action_decoder
        self.denormalizer = denormalizer
        self.module_type = module_type
        if isinstance(encoder, (LstmMlp, BeliefPerception)):
            self.forward = self.forward_lstm
        elif isinstance(encoder, GruMlp):
            self.forward = self.forward_gru
        elif isinstance(encoder, RecurrentMlp):
            if isinstance(encoder.rnn, nn.LSTM):
                self.forward = self.forward_lstm
            else:
                self.forward = self.forward_gru

    def forward(self, input: torch.Tensor):
        return self.denormalizer.denormalize(
            self.action_decoder.determine(self.encoder(self.normalizer(input)))
        )

    def forward_lstm(
        self,
        input: torch.Tensor,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        features, hidden = self.encoder(self.normalizer(input), hx)
        return (
            self.denormalizer.denormalize(self.action_decoder.determine(features)),
            hidden,
        )

    def forward_gru(self, input: torch.Tensor, hx: Optional[torch.Tensor] = None):
        features, hidden = self.encoder(self.normalizer(input), hx)
        return (
            self.denormalizer.denormalize(self.action_decoder.determine(features)),
            hidden,
        )


class ScriptSeqEncoderActor(nn.Module):
    def __init__(
        self,
        state_encoder: LinearBlock,
        latent_encoder: LinearBlock | LstmMlp | GruMlp,
        action_decoder: Distribution,
        normalizer: Normalization | RunningNorm1d,
        denormalizer: Normalization | RunningNorm1d,
    ):
        super().__init__()
        self.state_encoder = state_encoder
        self.latent_encoder = latent_encoder
        self.action_decoder = action_decoder
        self.normalizer = normalizer
        self.denormalizer = denormalizer
        if isinstance(latent_encoder, LstmMlp):
            self.forward = self.forward_lstm
        elif isinstance(latent_encoder, GruMlp):
            self.forward = self.forward_gru

    def forward(self, input: torch.Tensor):
        latent = self.state_encoder(self.normalizer(input))
        return self.denormalizer.denormalize(
            self.action_decoder.determine(self.latent_encoder(latent))
        )

    def forward_lstm(
        self,
        input: torch.Tensor,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        latent = self.state_encoder(self.normalizer(input))
        features, hidden = self.latent_encoder(latent, hx)
        return (
            self.denormalizer.denormalize(self.action_decoder.determine(features)),
            hidden,
        )

    def forward_gru(self, input: torch.Tensor, hx: Optional[torch.Tensor] = None):
        latent = self.state_encoder(self.normalizer(input))
        features, hidden = self.latent_encoder(latent, hx)
        return (
            self.denormalizer.denormalize(self.action_decoder.determine(features)),
            hidden,
        )
