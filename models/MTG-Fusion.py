import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *

import faiss

import torch
import torch.nn as nn
import torch.nn.functional as F

class CouplingLayer(nn.Module):
    """
    RealNVP-style affine coupling layer.
    Splits channels into two parts: x1, x2.
    Then uses x1 to predict scale and shift parameters for x2.
    """
    def __init__(self, num_channels, hidden_channels=64, mask_even=True):
        """
        num_channels: total #channels input to this layer
        hidden_channels: #channels in the intermediate CNN
        mask_even: if True, the first half channels are used to predict
                   (scale, shift) for the second half; if False, the reverse
        """
        super().__init__()
        self.num_channels = num_channels
        self.hidden_channels = hidden_channels
        self.mask_even = mask_even

        # The net processes x1 and outputs 2*(num_channels//2) channels => (scale, shift).
        self.net = nn.Sequential(
            nn.Conv2d(num_channels // 2, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, (num_channels // 2) * 2, kernel_size=3, padding=1)
        )

    def forward(self, x, reverse=False):
        """
        Forward pass of the coupling layer. If reverse=False, we apply the forward transformation.
        If reverse=True, we invert that transformation.

        x: (B, num_channels, H, W)
        return: transformed x, log_det (log determinant of the Jacobian)
        """
        B, C, H, W = x.shape

        # Split
        if self.mask_even:
            x1 = x[:, 0 : C // 2, :, :]
            x2 = x[:, C // 2 : C, :, :]
        else:
            x1 = x[:, C // 2 : C, :, :]
            x2 = x[:, 0 : C // 2, :, :]

        # Pass x1 through the net to get scale, shift
        out = self.net(x1)  # shape (B, 2*(C//2), H, W)
        s, t = out.chunk(2, dim=1)  # (scale, shift), each (B, C//2, H, W)

        # Clamp scale
        s = 0.64 * torch.tanh(s)  # in [-0.64, 0.64]

        # Forward / reverse
        if not reverse:
            # y2 = (x2 + t)*exp(s)
            y2 = (x2 + t) * torch.exp(s)
            log_det = s.sum(dim=[1, 2, 3])  # sum over channel,H,W
        else:
            # x2 = y2*exp(-s) - t
            y2 = x2 * torch.exp(-s) - t
            log_det = -s.sum(dim=[1, 2, 3])

        # Reassemble
        if self.mask_even:
            out = torch.cat([x1, y2], dim=1)
        else:
            out = torch.cat([y2, x1], dim=1)

        return out, log_det


class INNExtractor(nn.Module):
    """
    INN-based feature extractor for single-channel inputs (e.g., IR, MRI).
    Outputs a 64-channel feature map.
    """

    def __init__(
        self,
        in_channels=1,
        num_blocks=4,
        hidden_channels=64
    ):
        """
        in_channels: #channels in the input (e.g., 1 for a grayscale image)
        num_blocks: how many coupling blocks
        hidden_channels: #channels in intermediate layers
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        # Expand from 1->2 channels if in_channels=1,
        # so RealNVP can split them. Then, after the coupling blocks,
        # we project 2->64 channels for a higher-dim feature map.
        self.pre_conv = None
        self.post_conv = None
        expanded_channels = in_channels

        if in_channels == 1:
            expanded_channels = 2
            # 1->2 for RealNVP
            self.pre_conv = nn.Sequential(
                nn.Conv2d(1, 2, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
            # final 2->64 as a learned projection
            self.post_conv = nn.Conv2d(2, 64, kernel_size=3, padding=1)

        # Build the coupling blocks
        self.transforms = nn.ModuleList()
        mask = True
        for _ in range(num_blocks):
            self.transforms.append(
                CouplingLayer(
                    num_channels=expanded_channels,
                    hidden_channels=hidden_channels,
                    mask_even=mask
                )
            )
            mask = not mask

    def forward(self, x, reverse=False):
        """
        x: (B, in_channels, H, W)
        reverse=False => forward extracting features
        reverse=True  => inverse pass (partially invertible)
        Returns:
          out: (B, 64, H, W) if forward
               if reverse, shape is still (B, in_channels, H, W) or partial
          log_det_total: sum of log-dets from coupling blocks
        """
        B, C, H, W = x.shape
        log_det_total = 0.0

        # 1) Expand to 2 channels for RealNVP, if forward
        if self.pre_conv is not None and not reverse:
            x = self.pre_conv(x)  # (B,2,H,W)

        # 2) Coupling blocks
        if not reverse:
            # normal order
            for t in self.transforms:
                x, log_det = t(x, reverse=False)
                log_det_total += log_det.mean()
        else:
            # reverse order
            for t in reversed(self.transforms):
                x, log_det = t(x, reverse=True)
                log_det_total += log_det.mean()

        # 3) Project 2->64 if forward
        if self.post_conv is not None and not reverse:
            # This yields the final 64-channel feature
            x = self.post_conv(x)  # (B,64,H,W)

        # If reverse=True, we skip inverting post_conv because it's not truly invertible
        return x, log_det_total


class VisualFeatureExtractors(nn.Module):
    """
    Wrapper for E_A and E_B using the above INNExtractor,
    producing 64-channel outputs for each modality.
    """
    def __init__(
        self,
        in_channels_modA=1,
        in_channels_modB=1,
        num_blocks=4,
        hidden_channels=64
    ):
        super().__init__()
        self.EA = INNExtractor(
            in_channels=in_channels_modA,
            num_blocks=num_blocks,
            hidden_channels=hidden_channels
        )
        self.EB = INNExtractor(
            in_channels=in_channels_modB,
            num_blocks=num_blocks,
            hidden_channels=hidden_channels
        )

    def forward(self, imgA, imgB, reverse=False):
        """
        imgA, imgB: (B, 1, H, W) or more if in_channels_modA/B>1
        reverse=False => forward feature extraction => (B,64,H,W) each
        reverse=True  => partial invert attempt
        Returns:
          featA, featB, logdet
        """
        featA, detA = self.EA(imgA, reverse=reverse)  # (B,64,H,W) if forward
        featB, detB = self.EB(imgB, reverse=reverse)  # (B,64,H,W) if forward
        return featA, featB, detA + detB



class MultiScaleRetention(nn.Module):
    """
    A simplified multi-scale retention mechanism, reminiscent of multi-head attention,
    but with the concept that each 'head' can process features with different receptive fields.
    """

    def __init__(self, d_model=64, n_heads=8, expansion=4):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Query, Key, Value projections
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)

        # Sometimes multi-scale is done by having different sized convolution kernels
        # or window-based operations per head. For demonstration, we just do a small set:
        # e.g., separate depthwise conv or a small context gating. We'll keep it simple here.

        # Final output projection
        self.out_proj = nn.Linear(d_model, d_model)

        # Example: small depthwise conv to mimic multi-scale context
        # Each head might get a different kernel size.
        # For simplicity, we just define a single small conv, but you can expand it to multiple.
        self.multi_scale_conv = nn.Conv1d(
            in_channels=d_model, out_channels=d_model,
            kernel_size=3, padding=1, groups=d_model
        )

    def forward(self, x):
        """
        x: (B, N, d_model)
           where N = H*W if we flatten a 2D map, or the token length for a sequence
        """
        B, N, _ = x.shape

        # 1) Project to Q, K, V
        Q = self.query_proj(x)  # (B, N, d_model)
        K = self.key_proj(x)  # (B, N, d_model)
        V = self.value_proj(x)  # (B, N, d_model)

        # 2) Reshape for multi-head: (B, N, n_heads, head_dim)
        Q = Q.view(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, nHeads, N, headDim)
        K = K.view(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # 3) Compute attention weights
        #    scaled dot-product among Q and K
        attn = torch.matmul(Q, K.transpose(-1, -2)) / (self.head_dim ** 0.5)  # (B, nHeads, N, N)
        attn_weights = F.softmax(attn, dim=-1)

        # 4) Weighted sum of values
        out = torch.matmul(attn_weights, V)  # (B, nHeads, N, headDim)

        # 5) Combine heads
        out = out.permute(0, 2, 1, 3).contiguous()  # -> (B, N, nHeads, headDim)
        out = out.view(B, N, self.d_model)  # -> (B, N, d_model)

        # 6) Multi-scale augmentation (1D depthwise conv across N)
        #    We interpret N as "sequence length," so we treat out as (B, d_model, N) for conv1d
        out_for_conv = out.permute(0, 2, 1)  # (B, d_model, N)
        out_for_conv = self.multi_scale_conv(out_for_conv)  # (B, d_model, N)
        out_for_conv = out_for_conv.permute(0, 2, 1)  # (B, N, d_model)

        out = out + out_for_conv  # combine the multi-scale context
        out = self.out_proj(out)  # final linear

        return out


class RetNetBlock(nn.Module):
    """
    One 'RetNet' block containing:
      - a MultiScaleRetention module
      - a Feed Forward Network (FFN)
      - residual connections & layer normalization
    """

    def __init__(self, d_model=64, n_heads=8, expansion=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # Multi-scale retention
        self.msr = MultiScaleRetention(d_model, n_heads, expansion)
        self.norm1 = nn.LayerNorm(d_model)

        # Feed-forward
        hidden_dim = d_model * expansion
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, N, d_model)
        # 1) MSR + residual
        x_norm = self.norm1(x)
        x_msr = self.msr(x_norm)
        x = x + x_msr

        # 2) FFN + residual
        x_norm2 = self.norm2(x)
        x_ffn = self.ffn(x_norm2)
        x = x + x_ffn
        return x


class RetNetExtractor(nn.Module):
    """
    RetNet-based cross-modality shared feature extractor.
    Example configuration:
      - 4 RetNet blocks
      - each with 8 attention heads
      - hidden dimension = 64
    """

    def __init__(
            self,
            in_channels=2,
            d_model=64,
            n_heads=8,
            expansion=4,
            num_layers=4
    ):
        """
        in_channels: e.g. 2 for IR + VIS images
        d_model: dimension of internal representation (64)
        n_heads: multi-head count (6)
        expansion: feed-forward expansion ratio
        num_layers: how many RetNet blocks
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # Initial projection from in_channels -> d_model
        # so we can feed it into the RetNet blocks
        self.proj_in = nn.Conv2d(in_channels, d_model, kernel_size=3, padding=1)

        # Stack of RetNetBlocks
        self.blocks = nn.ModuleList([
            RetNetBlock(d_model=d_model, n_heads=n_heads, expansion=expansion)
            for _ in range(num_layers)
        ])

        # Final layer normalization after the last block
        self.norm_out = nn.LayerNorm(d_model)

        # Optionally, a final 1x1 conv to project from d_model back
        # to d_model (or to a smaller dimension if desired).
        # We'll keep it simple:
        self.proj_out = nn.Conv2d(d_model, d_model, kernel_size=1)

    def forward(self, x):
        """
        x: (B, 2, H, W)  # channel-concat of two source images
        returns: (B, d_model, H, W)  # shared features
        """
        B, C, H, W = x.shape
        # 1) project to (B, d_model, H, W)
        x = self.proj_in(x)  # (B, d_model, H, W)

        # 2) flatten to sequence: (B, H*W, d_model)
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, self.d_model)

        # 3) pass through RetNet blocks
        for block in self.blocks:
            x = block(x)  # (B, H*W, d_model)

        # 4) final LN
        x = self.norm_out(x)  # (B, H*W, d_model)

        # 5) reshape back to (B, d_model, H, W)
        x = x.view(B, H, W, self.d_model).permute(0, 3, 1, 2)

        # 6) optional final 1x1 conv
        x = self.proj_out(x)
        return x


class CycleMLPBlock(nn.Module):
    """
    A minimal CycleMLP-like block:
      - shift-based local mixing to emulate an MLP at each pixel
      - channel mixing
      - residual connection
    """

    def __init__(self, dim, expansion=4):
        super().__init__()
        self.dim = dim
        hidden_dim = dim * expansion

        # MLP1: channel expansion
        self.mlp1 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        # MLP2: "cycle shift" or local mixing.
        # We'll do a simple depthwise conv or shift-based operation
        # to emulate "CycleMLP" behavior.
        self.dw_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3,
                                 padding=1, groups=hidden_dim)
        self.activation = nn.ReLU(inplace=True)

        # Final reduce
        self.mlp2 = nn.Conv2d(hidden_dim, dim, kernel_size=1)

        self.norm = nn.BatchNorm2d(dim)  # or LayerNorm in a 2D sense

    def cycle_shift(self, x):
        """
        Very simplified 'cycle' shift for demonstration.
        x: (B, hidden_dim, H, W)
        We'll shift the feature along width dimension by 1 to emulate cyclical shift.
        """
        B, C, H, W = x.shape
        # shift right by 1 pixel in W dimension, wrap-around
        shifted = torch.roll(x, shifts=1, dims=3)
        return shifted

    def forward(self, x):
        """
        x: (B, dim, H, W)
        returns (B, dim, H, W)
        """
        # residual
        identity = x

        # normalization
        x = self.norm(x)

        # MLP part1
        x = self.mlp1(x)  # (B, hidden_dim, H, W)

        # cycle shift or local mixing
        x = self.cycle_shift(x)
        x = self.dw_conv(x)
        x = self.activation(x)

        # MLP part2
        x = self.mlp2(x)  # (B, dim, H, W)

        # residual
        x = x + identity
        return x


# Text-Guided CMSF Extractor (TGCE)
class TGCE(nn.Module):
    """
    Text-Guided CMSF Extractor (TGCE).

    Steps (unchanged from original):
      1) Build a text cube T of shape (B, H, W, C).
      2) Flatten to (B, H*W, C).
      3) Find top-1 neighbor for each pixel embedding (ANN-based).
      4) Domain transform (horizontal + vertical).
      5) Distance-based weight generation -> final CMSF.

    """

    def __init__(
        self,
        feat_dim=64,       # dimension of visual features
        hidden_dim=256,    # for distance->weight final MLP
        n_cycle_blocks=4,  # how many CycleMLP blocks to build text submaps
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim

        # from the averaged text embeddings, then combine them.

        self.text_A  = self._make_text(feat_dim, n_cycle_blocks)
        self.text_B  = self._make_text(feat_dim, n_cycle_blocks)
        self.text_AB = self._make_text(feat_dim, n_cycle_blocks)

        self.mlp_tv = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        self.mlp_tt = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def _make_text(self, dim, n_blocks):
        """
          - 1) linear => shape (B,dim,1,1)
          - 2) upsample to (H,W) at runtime
          - 3) cyclemlp blocks
        """
        layers = nn.ModuleList()
        # 1) from text embedding (64-d) => (dim,1,1)
        linear = nn.Linear(self.feat_dim, dim)
        # We'll store in a small container
        blockseq = nn.ModuleList([CycleMLPBlock(dim=dim, expansion=4) for _ in range(n_blocks)])
        container = nn.ModuleDict({
            'fc': linear,
            'blocks': blockseq
        })
        return container

    def _build_text_cube(self, tA_avg, tB_avg, tAB_avg, H, W):
        """
        Build the text cube T: (B,H,W,feat_dim).
        Replaces the original approach that used:
          tA_H * tB_W * tAB_C
        We'll produce (B, feat_dim, H, W) from each text, multiply them together => T
        """
        # each text_map_X is a nn.ModuleDict with fc + blocks
        # 1) project => (B,dim)
        # 2) reshape => (B,dim,1,1)
        # 3) upsample => (B,dim,H,W)
        # 4) pass cycle blocks => (B,dim,H,W)

        def run_text(container, text_emb, H, W):
            fc = container['fc']
            blocks = container['blocks']
            # text_emb: (B, feat_dim)
            x = fc(text_emb)  # => (B, dim)
            x = F.relu(x, inplace=True)
            x = x.unsqueeze(-1).unsqueeze(-1)  # => (B, dim, 1, 1)
            x = F.interpolate(x, size=(H,W), mode='bilinear', align_corners=False)
            # pass cyclemlp
            for blk in blocks:
                x = blk(x)
            return x  # shape => (B, dim, H, W)

        # produce three maps
        A  = run_text(self.text_map_A,  tA_avg,  H, W)
        B  = run_text(self.text_map_B,  tB_avg,  H, W)
        AB = run_text(self.text_map_AB, tAB_avg, H, W)

        # multiply all => shape (B, dim, H, W)
        T_3d = A * B * AB
        # rearr => (B,H,W,dim)
        T_4d = T_3d.permute(0,2,3,1)
        return T_4d

    def forward(self, V, tA, tB, tAB):
        """
        V:   (B, feat_dim, H, W)   (the visual features)
        tA:  (B, L_A, feat_dim)
        tB:  (B, L_B, feat_dim)
        tAB: (B, L_AB, feat_dim)

        Returns: V_CMSF => (B, feat_dim, H, W)
        """
        device = V.device
        B, C, H, W = V.shape

        # Step 1) Build text cube T => shape (B,H,W,feat_dim)
        tA_avg  = tA.mean(dim=1)     # (B, feat_dim)
        tB_avg  = tB.mean(dim=1)     # (B, feat_dim)
        tAB_avg = tAB.mean(dim=1)    # (B, feat_dim)

        T_4d = self._build_text_cube(tA_avg, tB_avg, tAB_avg, H, W)  # (B,H,W,feat_dim)

        # Step 2) Flatten
        V_flat = V.permute(0,2,3,1).reshape(B, H*W, C)      # (B, HW, C)
        T_flat = T_4d.view(B, H*W, C)                      # (B, HW, C)

        # Step 3) Reordering with GPU Faiss
        T_reordered_list = []
        faiss_res = faiss.StandardGpuResources()

        for b_idx in range(B):
            text_embed = T_flat[b_idx]   # (HW, C)
            pixel_embed= V_flat[b_idx]   # (HW, C)

            # L2-normalize
            text_embed_np  = text_embed.detach().cpu().numpy().astype('float32')
            pixel_embed_np = pixel_embed.detach().cpu().numpy().astype('float32')
            norm_text  = (text_embed_np**2).sum(axis=1, keepdims=True)**0.5 + 1e-6
            norm_pixel = (pixel_embed_np**2).sum(axis=1, keepdims=True)**0.5 + 1e-6
            text_embed_np  /= norm_text
            pixel_embed_np /= norm_pixel

            d_dim = C
            index_cpu = faiss.IndexFlatL2(d_dim)
            gpu_index = faiss.index_cpu_to_gpu(faiss_res, 0, index_cpu)
            gpu_index.add(text_embed_np)

            distances, indices = gpu_index.search(pixel_embed_np, 1)
            reorder_tensor = torch.from_numpy(text_embed_np[indices.flatten()]).to(device)
            T_reordered_list.append(reorder_tensor.unsqueeze(0))  # (1,HW,C)

        T_reordered = torch.cat(T_reordered_list, dim=0)  # (B, HW, C)

        # Step 4) Domain transform (horizontal + vertical)
        V_dt_list = []
        for b_idx in range(B):
            v_batch = V_flat[b_idx].clone()      # shape (HW, C)
            t_batch = T_reordered[b_idx].clone() # shape (HW, C)

            # pass1: row-wise
            v_batch = self.domain_transform_rows(v_batch, t_batch, H, W)
            # pass2: col-wise
            v_batch = self.domain_transform_cols(v_batch, t_batch, H, W)

            V_dt_list.append(v_batch.unsqueeze(0))

        V_dt = torch.cat(V_dt_list, dim=0)  # (B, HW, C)
        V_dt_4D = V_dt.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)

        # Step 5) Distances => weighting

        vt = V_dt.view(B, H*W, C)
        tr = T_reordered
        D_TV_list = []
        D_TT_list = []

        for b_idx in range(B):
            vt_b = vt[b_idx]  # (HW,C)
            tr_b = tr[b_idx]  # (HW,C)
            tv_vals = []
            tt_vals = []
            for pix_idx in range(H*W - 1):
                dist_tv = cosine_distance(vt_b[pix_idx].unsqueeze(0), tr_b[pix_idx].unsqueeze(0))
                dist_tt = cosine_distance(tr_b[pix_idx].unsqueeze(0), tr_b[pix_idx+1].unsqueeze(0))
                tv_vals.append(dist_tv)
                tt_vals.append(dist_tt)
            # last pixel
            dist_tv = cosine_distance(vt_b[-1].unsqueeze(0), tr_b[-1].unsqueeze(0))
            tv_vals.append(dist_tv)
            tt_vals.append(torch.zeros(1, device=device))
            D_TV_list.append(torch.cat(tv_vals, dim=0).unsqueeze(0))  # (1, HW)
            D_TT_list.append(torch.cat(tt_vals, dim=0).unsqueeze(0))  # (1, HW)

        D_TV_ = torch.cat(D_TV_list, dim=0) # (B, HW)
        D_TT_ = torch.cat(D_TT_list, dim=0) # (B, HW)

        # pass each pixel's distance as a shape (B*HW,1) => mlp => (B*HW,1)
        # then reshape to (B,HW)
        BHW = B*(H*W)

        d_tv = D_TV_.view(BHW,1)   # (BHW,1)
        d_tt = D_TT_.view(BHW,1)   # (BHW,1)

        w_tv = self.mlp_tv(d_tv)   # => (BHW,1)
        w_tt = self.mlp_tt(d_tt)   # => (BHW,1)

        w_fused = (w_tv + w_tt).view(B, 1, H, W)
        w_fused = self.sigmoid(w_fused)

        # final
        V_CMSF = V_dt_4D * w_fused
        return V_CMSF

    def domain_transform_rows(self, v_batch, t_batch, H, W):
        """
        row-wise pass
        v_batch, t_batch: (HW, C)
        """
        v_out = v_batch.clone()
        for r in range(H):
            row_start = r*W
            for c in range(1, W):
                idx_cur = row_start + c
                idx_prev= row_start + (c-1)
                dist = cosine_distance(v_out[idx_cur].unsqueeze(0), t_batch[idx_prev].unsqueeze(0))
                a = torch.exp(-dist)
                v_out[idx_cur] = (1 - a)*v_out[idx_cur] + a*v_out[idx_prev]
        return v_out

    def domain_transform_cols(self, v_batch, t_batch, H, W):
        """
        col-wise pass
        """
        v_out = v_batch.clone()
        for c in range(W):
            for r in range(1, H):
                idx_cur = r*W + c
                idx_prev= (r-1)*W + c
                dist = cosine_distance(v_out[idx_cur].unsqueeze(0), t_batch[idx_prev].unsqueeze(0))
                a = torch.exp(-dist)
                v_out[idx_cur] = (1 - a)*v_out[idx_cur] + a*v_out[idx_prev]
        return v_out


class CrossAttentionRestormer(nn.Module):
    """
    Two-pass cross-attention:
      (1) Update text: Q=text, K=V=visual
      (2) Update visual: Q=visual, K=V=updated_text

    Both passes include a Restormer-like feed-forward
    (one for text, one for visual).
    """

    def __init__(self, d_model=64, n_heads=4, ff_expansion=2):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # ----- Pass 1: text update (Q=text, K=V=visual) -----
        self.norm_text_1 = nn.LayerNorm(d_model)
        self.q_proj_text_1 = nn.Linear(d_model, d_model)  # for text
        self.k_proj_text_1 = nn.Linear(d_model, d_model)  # for visual
        self.v_proj_text_1 = nn.Linear(d_model, d_model)  # for visual
        self.attn_out_text_1 = nn.Linear(d_model, d_model)

        # FF for text
        hidden_dim_text = int(d_model * ff_expansion)
        self.norm_text_ff_1 = nn.LayerNorm(d_model)
        self.ff_text_1 = nn.Sequential(
            nn.Linear(d_model, hidden_dim_text),
            nn.GELU(),
            nn.Linear(hidden_dim_text, d_model),
        )

        # ----- Pass 2: visual update (Q=visual, K=V=updated_text) -----
        self.norm_vis_2 = nn.LayerNorm(d_model)
        self.q_proj_vis_2 = nn.Linear(d_model, d_model)  # for visual
        self.k_proj_vis_2 = nn.Linear(d_model, d_model)  # for updated text
        self.v_proj_vis_2 = nn.Linear(d_model, d_model)  # for updated text
        self.attn_out_vis_2 = nn.Linear(d_model, d_model)

        # FF for visual
        hidden_dim_vis = int(d_model * ff_expansion)
        self.norm_vis_ff_2 = nn.LayerNorm(d_model)
        self.ff_vis_2 = nn.Sequential(
            nn.Linear(d_model, hidden_dim_vis),
            nn.GELU(),
            nn.Linear(hidden_dim_vis, d_model),
        )

    def forward(self, smsf, text_embed):
        """
        Args:
          text_embed: (B, L, d_model)
          smsf:       (B, HW, d_model)
        Returns:
          updated_vis:  (B, HW, d_model)
        """
        B, L, _ = text_embed.shape
        _, HW, _ = smsf.shape
        d_head = self.d_model // self.n_heads

        # =============== Pass 1: Update Text ===============
        # Q = text, K=V = smsf
        # Norm text
        text_norm = self.norm_text_1(text_embed)  # (B, L, d_model)
        Q = self.q_proj_text_1(text_norm)  # (B, L, d_model)

        # We'll just do K=V from the raw SMSF or you could LN it:
        K = self.k_proj_text_1(smsf)  # (B, HW, d_model)
        V = self.v_proj_text_1(smsf)  # (B, HW, d_model)

        # Reshape for multi-head
        Q = Q.view(B, L, self.n_heads, d_head).permute(0, 2, 1, 3)  # (B, nH, L, d_head)
        K = K.view(B, HW, self.n_heads, d_head).permute(0, 2, 1, 3)  # (B, nH, HW,d_head)
        V = V.view(B, HW, self.n_heads, d_head).permute(0, 2, 1, 3)

        # scaled dot-product
        attn_logits = torch.matmul(Q, K.transpose(-1, -2)) / (d_head ** 0.5)  # (B,nH,L,HW)
        attn_weights = F.softmax(attn_logits, dim=-1)
        out_text = torch.matmul(attn_weights, V)  # (B,nH,L,d_head)

        # combine heads
        out_text = out_text.permute(0, 2, 1, 3).contiguous()  # (B, L, nH, d_head)
        out_text = out_text.view(B, L, self.d_model)  # (B, L, d_model)
        out_text = self.attn_out_text_1(out_text)

        # residual
        updated_text = text_embed + out_text

        # feed-forward for text
        text_norm2 = self.norm_text_ff_1(updated_text)
        ff_text = self.ff_text_1(text_norm2)
        updated_text = updated_text + ff_text  # final updated text

        # =============== Pass 2: Update Visual ===============
        # Q = smsf, K=V = updated_text
        vis_norm = self.norm_vis_2(smsf)  # (B, HW, d_model)
        Qv = self.q_proj_vis_2(vis_norm)  # (B, HW, d_model)

        Kt = self.k_proj_vis_2(updated_text)  # (B, L, d_model)
        Vt = self.v_proj_vis_2(updated_text)  # (B, L, d_model)

        # multi-head
        Qv = Qv.view(B, HW, self.n_heads, d_head).permute(0, 2, 1, 3)  # (B,nH,HW,d_head)
        Kt = Kt.view(B, L, self.n_heads, d_head).permute(0, 2, 1, 3)  # (B,nH,L, d_head)
        Vt = Vt.view(B, L, self.n_heads, d_head).permute(0, 2, 1, 3)

        attn_logits_v = torch.matmul(Qv, Kt.transpose(-1, -2)) / (d_head ** 0.5)  # (B,nH,HW,L)
        attn_weights_v = F.softmax(attn_logits_v, dim=-1)
        out_vis = torch.matmul(attn_weights_v, Vt)  # (B,nH,HW,d_head)

        out_vis = out_vis.permute(0, 2, 1, 3).contiguous()  # (B, HW, nH, d_head)
        out_vis = out_vis.view(B, HW, self.d_model)
        out_vis = self.attn_out_vis_2(out_vis)

        updated_vis = smsf + out_vis

        # feed-forward for visual
        vis_norm2 = self.norm_vis_ff_2(updated_vis)
        ff_vis = self.ff_vis_2(vis_norm2)
        updated_vis = updated_vis + ff_vis

        return updated_vis
        #return updated_text, updated_vis


# -------------------------------------------------------------------
# 2) Text-Guided SMSF Fusion (TGSF)
# -------------------------------------------------------------------
class TGSF(nn.Module):
    """
    TGSF:
      1) Aligns visual (SMSF) & text features (tAS, tBS) with cross-attn modules.
         We do 4 blocks => channel-concat => (B, d_model*4, H, W) for each side.
      2) Dimension-specific text-vision fusion.
    """

    def __init__(
        self,
        embed_dim=64,  # dimension of text embeddings
        d_model=64,    # dimension of visual features
        n_heads=4,
        ff_expansion=2,
        num_blocks=4,  # cross-attn modules
        H=256,
        W=256,
        C=64,          # final channel dimension
        hidden_dim=32, # internal dimension for TGSF's CNN
        cycle_expansion=4,  # expansion factor for cycle blocks
        n_cycle_blocks=4    # how many cycle blocks used for "dimension-specific" expansions
    ):
        super().__init__()
        self.num_blocks = num_blocks
        # self.H = H   # not used at init
        # self.W = W
        self.C = C
        self.d_model = d_model

        self.cross_blocks_A = nn.ModuleList([
            CrossAttentionRestormer(d_model=d_model, n_heads=n_heads, ff_expansion=ff_expansion)
            for _ in range(num_blocks)
        ])
        self.cross_blocks_B = nn.ModuleList([
            CrossAttentionRestormer(d_model=d_model, n_heads=n_heads, ff_expansion=ff_expansion)
            for _ in range(num_blocks)
        ])

        # ============ dimension expansions replaced with CycleMLP approach ============#
        self.textDimA = self._make_cycle_dim(embed_dim, n_cycle_blocks, cycle_expansion)
        self.textDimB = self._make_cycle_dim(embed_dim, n_cycle_blocks, cycle_expansion)

        # CNN for weighting
        self.cnnA = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.cnnB = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # final reduce
        self.final_reducer = nn.Conv2d(d_model*num_blocks, self.C, kernel_size=1)

    def _make_cycle_dim(self, embed_dim, n_blocks, expansion):
        modules = nn.ModuleDict()
        modules["fc"] = nn.Linear(embed_dim, self.d_model)
        modules["cycle_blocks"] = nn.ModuleList([
            CycleMLPBlock(dim=self.d_model, expansion=expansion)
            for _ in range(n_blocks)
        ])
        return modules

    def _text_dim(self, container, text_emb, H, W):
        """
        run forward pass:
          1) fc => (B,d_model)
          2) unsqueeze => (B,d_model,1,1)
          3) upsample => (B,d_model,H,W)
          4) pass n cycle blocks
        => shape (B,d_model,H,W)
        """
        fc = container["fc"]
        blocks = container["cycle_blocks"]
        x = fc(text_emb)  # => (B,d_model)
        x = F.relu(x, inplace=True)
        x = x.unsqueeze(-1).unsqueeze(-1)  # (B,d_model,1,1)
        x = F.interpolate(x, size=(H,W), mode='bilinear', align_corners=False)
        for blk in blocks:
            x = blk(x)
        return x

    def _reshape_hw(self, x, B, H, W):

        return x.view(B, 1, H, W)

    def _reshape_hc_to_hw(self, x, B, H, W, C):

        # we'll just interpret x as (B,H*C) => produce a shape (B,1,H,W) from the cycle approach
        x_2d = x.view(B, H, C).unsqueeze(1)
        x_2d = F.interpolate(x_2d, size=(H, W), mode='bilinear', align_corners=False)
        return x_2d

    def _reshape_wc_to_hw(self, x, B, H, W, C):
        x_2d = x.view(B, W, C).unsqueeze(1)
        x_2d = F.interpolate(x_2d, size=(H, W), mode='bilinear', align_corners=False)
        return x_2d

    def forward(self, SMSF_A, SMSF_B, tAS, tBS):
        """
        Steps 1-6 from the original. We only modify how dimension expansions are done.
        """
        B, dC, H, W = SMSF_A.shape

        # ========== Step 1) Flatten SMSFs ========== #
        SMSF_Af = SMSF_A.permute(0, 2, 3, 1).reshape(B, H*W, dC)
        SMSF_Bf = SMSF_B.permute(0, 2, 3, 1).reshape(B, H*W, dC)

        # ========== Step 2) cross-attn blocks A side ========== #
        tAS_ave = tAS
        smsfa_aligned_list = []
        smsfa_current = SMSF_Af
        for i in range(self.num_blocks):
            smsfa_current = self.cross_blocks_A[i](smsfa_current, tAS_ave)
            smsfa_aligned_list.append(smsfa_current)

        # ========== Step 3) channel-concat => (B, HW, d_model * 4) ========== #
        smsfa_cat = torch.cat(smsfa_aligned_list, dim=-1)
        # ========== Step 4) Reshape back ========== #
        d_model_cat = dC * self.num_blocks
        smsfa_cat_4d = smsfa_cat.view(B, H, W, d_model_cat).permute(0, 3, 1, 2)

        # ========== Step 5) Repeat for B side ========== #
        tBS_ave = tBS
        smsfb_aligned_list = []
        smsfb_current = SMSF_Bf
        for i in range(self.num_blocks):
            smsfb_current = self.cross_blocks_B[i](smsfb_current, tBS_ave)
            smsfb_aligned_list.append(smsfb_current)

        smsfb_cat = torch.cat(smsfb_aligned_list, dim=-1)
        smsfb_cat_4d = smsfb_cat.view(B, H, W, d_model_cat).permute(0, 3, 1, 2)

        # ========== Step 6) dimension-specific text-vision fusion ========== #

        # We'll do a cycle-based approach but preserve the "3 channels" notion for CNN input.
        # => We'll produce 3 distinct text-based maps => cat => shape (B,3,H,W).
        # Then pass self.cnnA. We'll do that for both A, B.

        # produce text maps => shape (B, d_model, H, W)
        # we can do one map for "hw", "hc", "wc". We'll replicate below:

        tAS_mean = tAS.mean(dim=1)  # (B, embed_dim)
        tBS_mean = tBS.mean(dim=1)  # (B, embed_dim)

        # We'll produce 3 cycle-based expansions for each side,
        # to emulate "hw", "hc", "wc" expansions. Then cat => (B,3*d_model,H,W).
        # Then do a 1x1 conv => (B,3,H,W).

        # -- for A
        map_hw_A = self._text_dim_map(self.textDimA, tAS_mean, H, W)  # (B,d_model,H,W)
        map_hc_A = self._text_dim_map(self.textDimA, tAS_mean, H, W)
        map_wc_A = self._text_dim_map(self.textDimA, tAS_mean, H, W)

        mapsA = torch.cat([map_hw_A, map_hc_A, map_wc_A], dim=1)  # (B, 3*d_model, H, W)
        # reduce to 3 channels
        reduce3A = nn.Conv2d(3*dC, 3, kernel_size=1).to(mapsA.device)
        A_3ch = reduce3A(mapsA)  # (B,3,H,W)

        # -- for B
        map_hw_B = self._text_dim_map(self.textDimB, tBS_mean, H, W)
        map_hc_B = self._text_dim_map(self.textDimB, tBS_mean, H, W)
        map_wc_B = self._text_dim_map(self.textDimB, tBS_mean, H, W)

        mapsB = torch.cat([map_hw_B, map_hc_B, map_wc_B], dim=1)
        reduce3B = nn.Conv2d(3*dC, 3, kernel_size=1).to(mapsB.device)
        B_3ch = reduce3B(mapsB)  # (B,3,H,W)

        # Now pass them to cnnA, cnnB
        M_A = self.cnnA(A_3ch)  # (B,1,H,W)
        M_B = self.cnnB(B_3ch)

        # final reduce for the cat_4d
        final_reducer = nn.Conv2d(d_model_cat, self.C, kernel_size=1).to(smsfa_cat_4d.device)
        smsfa_red = final_reducer(smsfa_cat_4d)  # (B,C,H,W)
        smsfb_red = final_reducer(smsfb_cat_4d)

        SMSF_fused = smsfa_red * M_A + smsfb_red * M_B
        return SMSF_fused

    def _text_dim(self, container, text_emb, H, W):
        """
        Similar to building a shape (B,d_model,H,W) map from text
        using cycle blocks
        """
        fc = container["fc"]
        blocks = container["cycle_blocks"]

        x = fc(text_emb)  # => (B,d_model)
        x = F.relu(x, inplace=True)
        x = x.unsqueeze(-1).unsqueeze(-1)  # (B,d_model,1,1)
        x = F.interpolate(x, size=(H,W), mode='bilinear', align_corners=False)
        for blk in blocks:
            x = blk(x)
        return x


class RetNetReconstructor(nn.Module):
    """
    A reconstruction module that:
      1) Accepts CMSF and fused SMSF, each of shape (B, C, H, W).
      2) Concatenates them along the channel dimension.
      3) Projects them to a d_model dimension.
      4) Flattens to (B, H*W, d_model).
      5) Processes with multiple RetNetBlocks.
      6) Reshapes back to (B, d_model, H, W).
      7) Outputs a single-channel grayscale image via a final 1x1 conv.
    """
    def __init__(
        self,
        cmsf_channels=64,    # channel dimension of CMSF from TGCE
        smsf_channels=64,    # channel dimension of fused SMSF from TGSF
        d_model=64,          # internal dimension for RetNet
        n_heads=8,           # consistent with your RetNetBlock
        expansion=4,
        num_layers=4         # how many RetNetBlocks
    ):
        super().__init__()
        # total input channels after concat
        self.in_channels = cmsf_channels + smsf_channels

        # initial projection: in_channels -> d_model
        self.proj_in = nn.Conv2d(self.in_channels, d_model, kernel_size=3, padding=1)

        # a stack of RetNetBlocks
        self.ret_blocks = nn.ModuleList([
            RetNetBlock(d_model=d_model, n_heads=n_heads, expansion=expansion)
            for _ in range(num_layers)
        ])

        # final conv to produce single-channel grayscale
        self.final_conv = nn.Conv2d(d_model, 1, kernel_size=1)

    def forward(self, cmsf, smsf):
        """
        Args:
          cmsf: (B, cmsf_channels, H, W) from TGCE
          smsf: (B, smsf_channels, H, W) from TGSF
        Returns:
          fused_img: (B, 1, H, W) single-channel grayscale
        """
        B, _, H, W = cmsf.shape

        # 1) concat CMSF + SMSF along channels
        x = torch.cat([cmsf, smsf], dim=1)  # (B, cmsf_channels + smsf_channels, H, W)

        # 2) project to d_model
        x = self.proj_in(x)  # (B, d_model, H, W)

        # 3) flatten to sequence for RetNet
        x = x.permute(0, 2, 3, 1).reshape(B, H*W, -1)  # (B, N, d_model), N=H*W

        # 4) pass through RetNet blocks
        for block in self.ret_blocks:
            x = block(x)  # still (B, N, d_model)

        # 5) reshape back
        x = x.view(B, H, W, -1).permute(0, 3, 1, 2)  # (B, d_model, H, W)

        # 6) final 1x1 conv -> single-channel grayscale
        fused_img = self.final_conv(x)  # (B, 1, H, W)

        return fused_img
