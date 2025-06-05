import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------------------
# Pixel-wise Losses: MSE and L1
# -------------------------------------------------------------------
mse_loss = nn.MSELoss()
l1_loss = nn.L1Loss()


# -------------------------------------------------------------------
# SSIM Loss
# -------------------------------------------------------------------
def ssim_loss(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """
    Placeholder SSIM loss (1 - SSIM). Currently uses L1 difference.
    For better fidelity, install pytorch_msssim and replace.
    """
    return F.l1_loss(img1, img2)


# -------------------------------------------------------------------
# Text-space Distance (cosine distance)
# (§3.6, Eq. (15))
# -------------------------------------------------------------------
def text_distance(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    """
    t1, t2: ℝ^{B×D} (fused-image embedding vs. reference text embedding)
    Returns: scalar = mean_{batch}(1 - cosine_similarity)
    """
    t1_norm = F.normalize(t1, p=2, dim=1)  # (B, D)
    t2_norm = F.normalize(t2, p=2, dim=1)  # (B, D)
    cos_sim = (t1_norm * t2_norm).sum(dim=1)  # (B,)
    dist = 1.0 - cos_sim                       # (B,)
    return dist.mean()


# -------------------------------------------------------------------
# Combined Loss Module
# -------------------------------------------------------------------
class MTG_FusionLoss(nn.Module):
    """
    Computes total loss for MTG-Fusion.
    Stage 1 (use_text=False): L = α·L_MSE + β·L_SSIM + γ·L_L1
    Stage 2 (use_text=True): L += η·L_TEXT
    """
    def __init__(self,
                 alpha: float = 1.0,
                 beta: float = 0.5,
                 gamma: float = 0.6,
                 eta: float = 0.5,
                 use_text: bool = False):
        super(MTG_FusionLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eta = eta
        self.use_text = use_text

    def forward(self,
                fused_image: torch.Tensor,
                srcA: torch.Tensor,
                srcB: torch.Tensor,
                tF_embed: torch.Tensor = None,
                tA_embed: torch.Tensor = None,
                tB_embed: torch.Tensor = None,
                tGT_embed: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
          - fused_image: (B,1,H,W)
          - srcA, srcB: (B,1,H,W)
          - tF_embed, tA_embed, tB_embed, tGT_embed: (B, D) for text embeddings (Stage 2 only)
        Returns:
          - total_loss: scalar tensor
        """
        # Pixel-wise losses
        L_mse_A = mse_loss(fused_image, srcA)
        L_mse_B = mse_loss(fused_image, srcB)
        L_mse = (L_mse_A + L_mse_B) / 2.0

        L_ssim_A = ssim_loss(fused_image, srcA)
        L_ssim_B = ssim_loss(fused_image, srcB)
        L_ssim = (L_ssim_A + L_ssim_B) / 2.0

        L_l1_A = l1_loss(fused_image, srcA)
        L_l1_B = l1_loss(fused_image, srcB)
        L_l1 = (L_l1_A + L_l1_B) / 2.0

        total_loss = self.alpha * L_mse + self.beta * L_ssim + self.gamma * L_l1

        if self.use_text:
            assert tF_embed is not None and tA_embed is not None and tB_embed is not None and tGT_embed is not None
            # Compute L_TEXT = ξ(tF, tA) + ξ(tF, tB) + ξ(tF, tGT)
            L_text_A = text_distance(tF_embed, tA_embed)
            L_text_B = text_distance(tF_embed, tB_embed)
            L_text_GT = text_distance(tF_embed, tGT_embed)
            L_text = L_text_A + L_text_B + L_text_GT
            total_loss = total_loss + self.eta * L_text

        return total_loss
