from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F


DEFAULT_BLIP_ITM_REVISION = "98cf803d8586e8e8b3ec527885ac50c422d514ae"


def image_gradients(image: Tensor) -> tuple[Tensor, Tensor]:
    dx = image[..., :, 1:] - image[..., :, :-1]
    dy = image[..., 1:, :] - image[..., :-1, :]
    return dx, dy


def ssim(image_a: Tensor, image_b: Tensor, window: int = 11) -> Tensor:
    padding = window // 2
    mu_a = F.avg_pool2d(image_a, window, stride=1, padding=padding)
    mu_b = F.avg_pool2d(image_b, window, stride=1, padding=padding)
    sigma_a = F.avg_pool2d(image_a * image_a, window, 1, padding) - mu_a.square()
    sigma_b = F.avg_pool2d(image_b * image_b, window, 1, padding) - mu_b.square()
    sigma_ab = F.avg_pool2d(image_a * image_b, window, 1, padding) - mu_a * mu_b
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)) / (
        (mu_a.square() + mu_b.square() + c1) * (sigma_a + sigma_b + c2)
    ).clamp_min(1e-8)
    return score.mean()


class FrozenBlipAlignment(nn.Module):
    """Differentiable fused-image to cached-text alignment."""

    def __init__(self, model_name: str, revision: str) -> None:
        super().__init__()
        try:
            from transformers import BlipModel
        except ImportError as error:
            raise ImportError("Install transformers to enable BLIP loss") from error
        self.blip = BlipModel.from_pretrained(
            model_name, revision=revision, use_safetensors=True
        )
        self.blip.requires_grad_(False)
        self.register_buffer(
            "mean", torch.tensor((0.48145466, 0.4578275, 0.40821073)).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor((0.26862954, 0.26130258, 0.27577711)).view(1, 3, 1, 1)
        )

    def forward(self, fused: Tensor, text: Mapping[str, Tensor]) -> Tensor:
        self.blip.eval()
        rgb = fused.repeat(1, 3, 1, 1)
        pixels = F.interpolate(rgb, size=(384, 384), mode="bicubic", align_corners=False)
        pixels = (pixels - self.mean) / self.std
        image_output = self.blip.get_image_features(pixel_values=pixels)
        image_embedding = (
            image_output.pooler_output
            if hasattr(image_output, "pooler_output")
            else image_output
        )
        image_embedding = F.normalize(image_embedding, dim=-1)
        losses = []
        for key in ("t_a", "t_b", "t_gt"):
            target = F.normalize(text[key].to(image_embedding.dtype), dim=-1)
            losses.append(1.0 - (image_embedding * target).sum(dim=-1).mean())
        return torch.stack(losses).mean()


class FusionLoss(nn.Module):
    def __init__(
        self,
        mse_weight: float = 1.0,
        ssim_weight: float = 0.5,
        l1_weight: float = 0.6,
        gradient_weight: float = 1.0,
        blip_weight: float = 0.0,
        blip_model: str = "Salesforce/blip-itm-base-coco",
        blip_revision: str = DEFAULT_BLIP_ITM_REVISION,
    ) -> None:
        super().__init__()
        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight
        self.l1_weight = l1_weight
        self.gradient_weight = gradient_weight
        self.blip_weight = blip_weight
        self.text_alignment = (
            FrozenBlipAlignment(blip_model, blip_revision)
            if blip_weight > 0
            else None
        )

    def forward(
        self,
        fused: Tensor,
        modality_a: Tensor,
        modality_b: Tensor,
        text: Mapping[str, Tensor] | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        intensity_target = torch.maximum(modality_a, modality_b)
        mse_loss = F.mse_loss(fused, intensity_target)
        l1_loss = F.l1_loss(fused, intensity_target)
        ssim_loss = 1.0 - 0.5 * (
            ssim(fused, modality_a) + ssim(fused, modality_b)
        )

        fused_dx, fused_dy = image_gradients(fused)
        a_dx, a_dy = image_gradients(modality_a)
        b_dx, b_dy = image_gradients(modality_b)
        target_dx = torch.where(a_dx.abs() >= b_dx.abs(), a_dx, b_dx)
        target_dy = torch.where(a_dy.abs() >= b_dy.abs(), a_dy, b_dy)
        gradient_loss = F.l1_loss(fused_dx, target_dx) + F.l1_loss(fused_dy, target_dy)

        text_loss = fused.new_zeros(())
        if self.text_alignment is not None:
            if text is None:
                raise ValueError("BLIP alignment requires cached text features")
            text_loss = self.text_alignment(fused, text)

        fusion_total = (
            self.mse_weight * mse_loss
            + self.ssim_weight * ssim_loss
            + self.l1_weight * l1_loss
            + self.gradient_weight * gradient_loss
        )
        total = fusion_total + self.blip_weight * text_loss
        components = {
            "mse": mse_loss.detach(),
            "ssim": ssim_loss.detach(),
            "l1": l1_loss.detach(),
            "gradient": gradient_loss.detach(),
            "text": text_loss.detach(),
            "fusion": fusion_total.detach(),
            "total": total.detach(),
        }
        return total, components
