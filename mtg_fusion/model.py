from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import Tensor, nn

from .blocks import ConvNormAct, LocalCrossAlign, SharedEncoder, SourceEncoder
from .tgce import TGCE
from .tgsf import TGSF, VisualSMSFusion


TEXT_KEYS = ("t_a", "t_b", "t_ab", "t_as", "t_bs", "t_gt")


class AdaptiveTextResidual(nn.Module):
    """Content-adaptive, spatially varying text residual adapter."""

    def __init__(
        self,
        channels: int,
        text_dim: int,
        gate_init: float,
        residual_scale: float,
    ) -> None:
        super().__init__()
        if not 0.0 < gate_init < 1.0:
            raise ValueError("gate_init must be in (0, 1)")
        if residual_scale <= 0.0:
            raise ValueError("residual_scale must be positive")
        self.residual_scale = residual_scale
        self.text_projection = nn.Linear(text_dim, channels)
        self.gate_predictor = nn.Sequential(
            ConvNormAct(channels * 3, channels, 3),
            nn.Conv2d(channels, 1, 1),
        )
        self.residual_adapter = nn.Sequential(
            ConvNormAct(channels, channels, 3),
            nn.Conv2d(channels, channels, 1, bias=False),
        )
        gate_output = self.gate_predictor[-1]
        assert isinstance(gate_output, nn.Conv2d)
        nn.init.zeros_(gate_output.weight)
        nn.init.constant_(gate_output.bias, math.log(gate_init / (1.0 - gate_init)))
        residual_output = self.residual_adapter[-1]
        assert isinstance(residual_output, nn.Conv2d)
        nn.init.dirac_(residual_output.weight)

    def forward(
        self,
        baseline: Tensor,
        text_candidate: Tensor,
        text_context: Tensor,
        strength: float,
    ) -> tuple[Tensor, Tensor]:
        delta = text_candidate - baseline
        context = self.text_projection(text_context).unsqueeze(-1).unsqueeze(-1)
        context = context.expand(-1, -1, baseline.shape[-2], baseline.shape[-1])
        gate_input = torch.cat((baseline, delta, context), dim=1)
        gate = torch.sigmoid(self.gate_predictor(gate_input)) * strength
        correction = self.residual_scale * self.residual_adapter(delta)
        return baseline + gate * correction, gate


class FusionDecoder(nn.Module):
    def __init__(self, base: int, channels: int) -> None:
        super().__init__()
        self.mix = nn.Sequential(
            ConvNormAct(channels * 2, channels, 3),
            ConvNormAct(channels, channels, 3),
        )
        self.up1 = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.GELU(),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(channels, base * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.GELU(),
        )
        self.output = nn.Sequential(
            ConvNormAct(base * 2, base, 3),
            nn.Conv2d(base, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, cmsf: Tensor, sms: Tensor, shallow: Tensor) -> Tensor:
        deep = self.up2(self.up1(self.mix(torch.cat((cmsf, sms), dim=1))))
        deep = deep[..., :shallow.shape[-2], :shallow.shape[-1]]
        return self.output(torch.cat((deep, shallow), dim=1))


class MTGFusion(nn.Module):
    def __init__(
        self,
        base_channels: int = 32,
        feature_channels: int = 64,
        retention_blocks: int = 4,
        retention_heads: int = 4,
        retention_head_dim: int = 16,
        text_dim: int = 256,
        max_text_side: int = 64,
        match_size: int = 16,
        text_gate_init: float = 0.10,
        text_residual_scale: float = 0.10,
    ) -> None:
        super().__init__()
        self.encoder_a = SourceEncoder(1, base_channels, feature_channels)
        self.encoder_b = SourceEncoder(1, base_channels, feature_channels)
        self.encoder_ab = SharedEncoder(
            2,
            base_channels,
            feature_channels,
            retention_blocks,
            retention_heads,
            retention_head_dim,
        )
        self.tgce = TGCE(
            feature_channels, text_dim, max_text_side, match_size
        )
        self.aligner = LocalCrossAlign(feature_channels)
        self.tgsf = TGSF(feature_channels, text_dim, max_text_side)
        self.visual_sms_fusion = VisualSMSFusion(feature_channels)
        self.decoder = FusionDecoder(base_channels, feature_channels)
        self.cmsf_text_adapter = AdaptiveTextResidual(
            feature_channels, text_dim, text_gate_init, text_residual_scale
        )
        self.sms_text_adapter = AdaptiveTextResidual(
            feature_channels, text_dim, text_gate_init, text_residual_scale
        )
        self.register_buffer("inference_text_strength", torch.tensor(1.0))

    def configure_stage2_trainable(
        self,
        finetune_decoder: bool = False,
        finetune_output_layer: bool = False,
    ) -> None:
        """Freeze the Stage 1 visual path and train only text adapters/gates."""
        self.requires_grad_(False)
        self.tgce.requires_grad_(True)
        self.tgsf.requires_grad_(True)
        self.cmsf_text_adapter.requires_grad_(True)
        self.sms_text_adapter.requires_grad_(True)
        if finetune_decoder:
            self.decoder.requires_grad_(True)
        elif finetune_output_layer:
            # Permit only a low-rate luminance calibration at the final 3x3
            # convolution; all preceding Stage 1 decoding remains frozen.
            self.decoder.output[1].requires_grad_(True)

    @staticmethod
    def _validate_text(text: Mapping[str, Tensor] | None) -> None:
        if text is None:
            raise ValueError("Text features are required when use_text=True")
        missing = [key for key in TEXT_KEYS[:-1] if key not in text]
        if missing:
            raise KeyError(f"Missing text feature keys: {missing}")

    def forward(
        self,
        modality_a: Tensor,
        modality_b: Tensor,
        text: Mapping[str, Tensor] | None = None,
        use_text: bool = True,
        text_strength: float | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if text_strength is None:
            text_strength = float(self.inference_text_strength.item())
        if not 0.0 <= text_strength <= 1.0:
            raise ValueError("text_strength must be in [0, 1]")
        shallow_a, feature_a = self.encoder_a(modality_a)
        shallow_b, feature_b = self.encoder_b(modality_b)
        raw_shared = self.encoder_ab(torch.cat((modality_a, modality_b), dim=1))
        diagnostics: dict[str, Tensor] = {"raw_shared": raw_shared}

        if use_text:
            self._validate_text(text)
            assert text is not None
            text_cmsf, tgce_maps = self.tgce(
                raw_shared, text["t_a"], text["t_b"], text["t_ab"]
            )
            cmsf_context = (text["t_a"] + text["t_b"] + text["t_ab"]) / 3.0
            cmsf, cmsf_gate = self.cmsf_text_adapter(
                raw_shared, text_cmsf, cmsf_context, text_strength
            )
            diagnostics.update(tgce_maps)
            diagnostics["text_cmsf_gate"] = cmsf_gate
        else:
            cmsf = raw_shared

        sms_a = feature_a - cmsf
        sms_b = feature_b - cmsf
        sms_a, sms_b = self.aligner(sms_a, sms_b)
        if use_text:
            assert text is not None
            visual_sms, visual_weights = self.visual_sms_fusion(sms_a, sms_b)
            text_sms, text_weights = self.tgsf(
                sms_a, sms_b, text["t_as"], text["t_bs"]
            )
            sms_context = 0.5 * (text["t_as"] + text["t_bs"])
            fused_sms, sms_gate = self.sms_text_adapter(
                visual_sms, text_sms, sms_context, text_strength
            )
            weight_dtype = visual_weights.dtype
            sms_weights = torch.lerp(
                visual_weights,
                text_weights.to(dtype=weight_dtype),
                sms_gate.unsqueeze(1).to(dtype=weight_dtype),
            )
            diagnostics["text_sms_gate"] = sms_gate
        else:
            fused_sms, sms_weights = self.visual_sms_fusion(sms_a, sms_b)

        shallow = torch.maximum(shallow_a, shallow_b)
        fused = self.decoder(cmsf, fused_sms, shallow)
        diagnostics.update(
            {
                "cmsf": cmsf,
                "sms_a": sms_a,
                "sms_b": sms_b,
                "fused_sms": fused_sms,
                "sms_weights": sms_weights,
            }
        )
        return fused, diagnostics
