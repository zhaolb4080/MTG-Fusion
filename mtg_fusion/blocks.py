from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _groups(channels: int) -> int:
    for value in (8, 4, 2, 1):
        if channels % value == 0:
            return value
    return 1


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        activation: bool = True,
    ) -> None:
        padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(_groups(out_channels), out_channels),
        ]
        if activation:
            layers.append(nn.GELU())
        super().__init__(*layers)


class DenseSubNet(nn.Sequential):
    def __init__(self, channels: int) -> None:
        hidden = max(channels, 32)
        super().__init__(
            ConvNormAct(channels, hidden, 3),
            nn.Conv2d(hidden, channels, 3, padding=1),
        )


class AffineCoupling(nn.Module):
    """Compact invertible affine coupling block used by the two source encoders."""

    def __init__(self, channels: int, clamp: float = 1.5) -> None:
        super().__init__()
        if channels % 2:
            raise ValueError("AffineCoupling requires an even channel count")
        half = channels // 2
        self.scale = DenseSubNet(half)
        self.shift = DenseSubNet(half)
        self.clamp = clamp

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=1)
        log_scale = self.clamp * torch.tanh(self.scale(x1))
        y2 = x2 * torch.exp(log_scale) + self.shift(x1)
        return torch.cat((x1, y2), dim=1)


class LinearRetention2d(nn.Module):
    """Linear-complexity multi-head retention with a local depthwise branch."""

    def __init__(self, channels: int, heads: int, head_dim: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        inner = heads * head_dim
        self.qkv = nn.Conv2d(channels, inner * 3, 1, bias=False)
        self.local = nn.Conv2d(
            channels, channels, 5, padding=2, groups=channels, bias=False
        )
        self.project = nn.Conv2d(inner + channels, channels, 1, bias=False)
        self.norm = nn.GroupNorm(_groups(channels), channels)

    def forward(self, x: Tensor) -> Tensor:
        batch, _, height, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=1)

        def reshape(tensor: Tensor) -> Tensor:
            return tensor.reshape(
                batch, self.heads, self.head_dim, height * width
            ).transpose(-1, -2)

        q = F.elu(reshape(q)) + 1.0
        k = F.elu(reshape(k)) + 1.0
        v = reshape(v)
        scale = float(height * width)
        kv = torch.matmul(k.transpose(-1, -2), v) / scale
        normalizer = torch.matmul(q, k.mean(dim=-2).unsqueeze(-1)).clamp_min(1e-6)
        retained = torch.matmul(q, kv) / normalizer
        retained = retained.transpose(-1, -2).reshape(
            batch, self.heads * self.head_dim, height, width
        )
        mixed = torch.cat((retained, self.local(x)), dim=1)
        return self.norm(x + self.project(mixed))


class RetentionBlock(nn.Module):
    def __init__(self, channels: int, heads: int, head_dim: int) -> None:
        super().__init__()
        self.retention = LinearRetention2d(channels, heads, head_dim)
        self.ffn = nn.Sequential(
            ConvNormAct(channels, channels * 2, 1),
            nn.Conv2d(channels * 2, channels, 1),
        )
        self.norm = nn.GroupNorm(_groups(channels), channels)

    def forward(self, x: Tensor) -> Tensor:
        x = self.retention(x)
        return self.norm(x + self.ffn(x))


class SourceEncoder(nn.Module):
    def __init__(self, in_channels: int, base: int, channels: int) -> None:
        super().__init__()
        self.stem = ConvNormAct(in_channels, base, 3)
        self.down1 = ConvNormAct(base, channels, 3, stride=2)
        self.down2 = ConvNormAct(channels, channels, 3, stride=2)
        self.invertible = nn.Sequential(
            AffineCoupling(channels), AffineCoupling(channels)
        )

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor]:
        shallow = self.stem(image)
        deep = self.invertible(self.down2(self.down1(shallow)))
        return shallow, deep


class SharedEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base: int,
        channels: int,
        blocks: int,
        heads: int,
        head_dim: int,
    ) -> None:
        super().__init__()
        self.stem = ConvNormAct(in_channels, base, 3)
        self.down = nn.Sequential(
            ConvNormAct(base, channels, 3, stride=2),
            ConvNormAct(channels, channels, 3, stride=2),
        )
        self.blocks = nn.Sequential(
            *[RetentionBlock(channels, heads, head_dim) for _ in range(blocks)]
        )

    def forward(self, image_pair: Tensor) -> Tensor:
        return self.blocks(self.down(self.stem(image_pair)))


class LocalCrossAlign(nn.Module):
    """Efficient bidirectional local alignment for modality-specific features."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.a_to_b = nn.Conv2d(channels * 2, channels, 1)
        self.b_to_a = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, a: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
        pair = torch.cat((a, b), dim=1)
        gate_a = torch.sigmoid(self.b_to_a(pair))
        gate_b = torch.sigmoid(self.a_to_b(pair))
        return a + gate_a * b, b + gate_b * a

