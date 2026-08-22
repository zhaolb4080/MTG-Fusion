from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class TGCE(nn.Module):
    """Text-guided cross-modality shared-feature extractor.

    The paper's dense HW-by-HW lookup is approximated by matching every visual
    position to a pooled text codebook. This retains hard reordering while
    keeping memory bounded.
    """

    def __init__(
        self,
        channels: int,
        text_dim: int,
        max_side: int = 64,
        match_size: int = 16,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.max_side = max_side
        self.match_size = match_size
        self.h_projection = nn.Linear(text_dim, max_side)
        self.w_projection = nn.Linear(text_dim, max_side)
        self.c_projection = nn.Linear(text_dim, channels)
        self.weight_encoder = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, 1),
        )
        self.log_sigma = nn.Parameter(torch.tensor(1.0))

    def _axis_values(self, projection: nn.Linear, text: Tensor, size: int) -> Tensor:
        values = projection(text).unsqueeze(1)
        if size != self.max_side:
            values = F.interpolate(values, size=size, mode="linear", align_corners=False)
        return values.squeeze(1)

    def _text_cube(
        self, t_a: Tensor, t_b: Tensor, t_ab: Tensor, height: int, width: int
    ) -> Tensor:
        h_values = self._axis_values(self.h_projection, t_a, height)
        w_values = self._axis_values(self.w_projection, t_b, width)
        spatial = torch.tanh(h_values.unsqueeze(-1) + w_values.unsqueeze(-2))
        channel = torch.tanh(self.c_projection(t_ab)).unsqueeze(-1).unsqueeze(-1)
        return torch.tanh(channel + spatial.unsqueeze(1))

    def _reorder(self, visual: Tensor, cube: Tensor) -> Tensor:
        batch, channels, height, width = visual.shape
        pooled = F.adaptive_avg_pool2d(cube, (self.match_size, self.match_size))
        query = F.normalize(visual.flatten(2).transpose(1, 2), dim=-1)
        codebook = F.normalize(pooled.flatten(2).transpose(1, 2), dim=-1)
        indices = torch.bmm(query, codebook.transpose(1, 2)).argmax(dim=-1)
        gather_index = indices.unsqueeze(-1).expand(-1, -1, channels)
        matched = torch.gather(codebook, 1, gather_index)
        return matched.transpose(1, 2).reshape(batch, channels, height, width)

    @staticmethod
    def _cosine_distance(a: Tensor, b: Tensor, dim: int = 1) -> Tensor:
        return 1.0 - F.cosine_similarity(a, b, dim=dim, eps=1e-6)

    def _recursive_horizontal(self, x: Tensor, distance: Tensor) -> Tensor:
        sigma = F.softplus(self.log_sigma)
        conductance = torch.exp(-sigma * distance).unsqueeze(1)
        left: list[Tensor] = [x[:, :, :, 0]]
        for index in range(1, x.shape[-1]):
            alpha = conductance[:, :, :, index - 1]
            left.append((1.0 - alpha) * x[:, :, :, index] + alpha * left[-1])
        left_tensor = torch.stack(left, dim=-1)

        right: list[Tensor] = [x[:, :, :, -1]]
        for index in range(x.shape[-1] - 2, -1, -1):
            alpha = conductance[:, :, :, index]
            right.append((1.0 - alpha) * x[:, :, :, index] + alpha * right[-1])
        right_tensor = torch.stack(list(reversed(right)), dim=-1)
        return 0.5 * (left_tensor + right_tensor)

    def _recursive_vertical(self, x: Tensor, distance: Tensor) -> Tensor:
        transposed = x.transpose(-1, -2)
        filtered = self._recursive_horizontal(transposed, distance.transpose(-1, -2))
        return filtered.transpose(-1, -2)

    def forward(
        self, visual: Tensor, t_a: Tensor, t_b: Tensor, t_ab: Tensor
    ) -> tuple[Tensor, dict[str, Tensor]]:
        height, width = visual.shape[-2:]
        cube = self._text_cube(t_a, t_b, t_ab, height, width)
        reordered = self._reorder(visual, cube)

        pixel_to_word = self._cosine_distance(visual, reordered).unsqueeze(1)
        horizontal = self._cosine_distance(
            reordered[:, :, :, 1:], reordered[:, :, :, :-1]
        )
        horizontal = torch.cat((horizontal, horizontal[:, :, -1:]), dim=2)
        vertical = self._cosine_distance(
            reordered[:, :, 1:, :], reordered[:, :, :-1, :]
        )
        vertical = torch.cat((vertical, vertical[:, -1:, :]), dim=1)
        word_to_word = 0.5 * (horizontal + vertical)

        filtered_h = self._recursive_horizontal(visual, horizontal[:, :, :-1])
        filtered = self._recursive_vertical(filtered_h, vertical[:, :-1, :])
        weight = torch.sigmoid(
            self.weight_encoder(torch.cat((pixel_to_word, word_to_word.unsqueeze(1)), 1))
        )
        cmsf = filtered * (0.5 + weight)
        diagnostics = {
            "text_cube": cube,
            "reordered_text": reordered,
            "pixel_to_word": pixel_to_word,
            "word_to_word": word_to_word.unsqueeze(1),
            "cmsf_weight": weight,
        }
        return cmsf, diagnostics
