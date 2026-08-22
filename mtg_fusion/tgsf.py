from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class AxialTextWeight(nn.Module):
    def __init__(self, text_dim: int, channels: int, max_side: int) -> None:
        super().__init__()
        self.max_side = max_side
        self.h_projection = nn.Linear(text_dim, max_side)
        self.w_projection = nn.Linear(text_dim, max_side)
        self.c_projection = nn.Linear(text_dim, channels)
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )

    def _resize_axis(self, values: Tensor, size: int) -> Tensor:
        values = values.unsqueeze(1)
        if size != self.max_side:
            values = F.interpolate(values, size=size, mode="linear", align_corners=False)
        return values.squeeze(1)

    def forward(self, text: Tensor, height: int, width: int) -> Tensor:
        h = self._resize_axis(self.h_projection(text), height)
        w = self._resize_axis(self.w_projection(text), width)
        c = self.c_projection(text).unsqueeze(-1).unsqueeze(-1)
        raw = c + h.unsqueeze(1).unsqueeze(-1) + w.unsqueeze(1).unsqueeze(-2)
        return self.refine(torch.tanh(raw))


class TGSF(nn.Module):
    """Text-guided fusion of the two modality-specific feature maps."""

    def __init__(
        self, channels: int, text_dim: int, max_side: int = 64
    ) -> None:
        super().__init__()
        self.weight_a = AxialTextWeight(text_dim, channels, max_side)
        self.weight_b = AxialTextWeight(text_dim, channels, max_side)

    def forward(
        self, sms_a: Tensor, sms_b: Tensor, t_as: Tensor, t_bs: Tensor
    ) -> tuple[Tensor, Tensor]:
        height, width = sms_a.shape[-2:]
        logit_a = self.weight_a(t_as, height, width)
        logit_b = self.weight_b(t_bs, height, width)
        weights = torch.softmax(torch.stack((logit_a, logit_b), dim=1), dim=1)
        fused = weights[:, 0] * sms_a + weights[:, 1] * sms_b
        return fused, weights


class VisualSMSFusion(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.logits = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, 2, 1),
        )

    def forward(self, a: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
        scalar_weights = torch.softmax(self.logits(torch.cat((a, b), 1)), dim=1)
        weights = scalar_weights.unsqueeze(2)
        fused = scalar_weights[:, :1] * a + scalar_weights[:, 1:] * b
        return fused, weights

