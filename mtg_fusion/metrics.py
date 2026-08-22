from __future__ import annotations

import math
from typing import Callable

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F


def _as_tensor(image: np.ndarray) -> Tensor:
    array = image.astype(np.float32)
    if array.max() > 1.5:
        array = array / 255.0
    return torch.from_numpy(array).view(1, 1, *array.shape)


def entropy(image: np.ndarray) -> float:
    values = np.clip(np.rint(image), 0, 255).astype(np.uint8)
    histogram = np.bincount(values.ravel(), minlength=256).astype(np.float64)
    probability = histogram[histogram > 0] / histogram.sum()
    return float(-(probability * np.log2(probability)).sum())


def standard_deviation(image: np.ndarray) -> float:
    return float(np.std(image.astype(np.float64), ddof=1))


def spatial_frequency(image: np.ndarray) -> float:
    image = image.astype(np.float64)
    row = np.sqrt(np.mean(np.diff(image, axis=0) ** 2))
    column = np.sqrt(np.mean(np.diff(image, axis=1) ** 2))
    return float(np.sqrt(row * row + column * column))


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denominator = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return float(np.sum(a * b) / max(denominator, 1e-12))


def sum_correlations_difference(a: np.ndarray, b: np.ndarray, fused: np.ndarray) -> float:
    return _correlation(fused - a, b) + _correlation(fused - b, a)


def vif(a: np.ndarray, b: np.ndarray, fused: np.ndarray) -> float:
    try:
        from piq import vif_p
    except ImportError as error:
        raise ImportError("Install piq to calculate VIF") from error
    fused_t = _as_tensor(fused)
    a_t = _as_tensor(a)
    b_t = _as_tensor(b)
    minimum = 41
    if min(fused_t.shape[-2:]) < minimum:
        size = (max(minimum, fused_t.shape[-2]), max(minimum, fused_t.shape[-1]))
        fused_t = F.interpolate(fused_t, size=size, mode="bilinear", align_corners=False)
        a_t = F.interpolate(a_t, size=size, mode="bilinear", align_corners=False)
        b_t = F.interpolate(b_t, size=size, mode="bilinear", align_corners=False)
    with torch.no_grad():
        score_a = vif_p(fused_t, a_t, data_range=1.0)
        score_b = vif_p(fused_t, b_t, data_range=1.0)
    return float(0.5 * (score_a + score_b))


def _sobel(image: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    kernel_x = image.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    ).view(1, 1, 3, 3)
    kernel_y = kernel_x.transpose(-1, -2)
    gx = F.conv2d(image, kernel_x, padding=1)
    gy = F.conv2d(image, kernel_y, padding=1)
    magnitude = torch.sqrt(gx.square() + gy.square() + 1e-12)
    orientation = torch.atan2(gy, gx)
    return gx, gy, magnitude, orientation


def _edge_quality(source: Tensor, fused: Tensor) -> tuple[Tensor, Tensor]:
    _, _, source_g, source_a = _sobel(source)
    _, _, fused_g, fused_a = _sobel(fused)
    ratio = torch.minimum(source_g, fused_g) / torch.maximum(source_g, fused_g).clamp_min(1e-12)
    angle_difference = torch.remainder(source_a - fused_a + math.pi, 2 * math.pi) - math.pi
    angle_score = 1.0 - angle_difference.abs() / math.pi
    q_g = 0.9994 / (1.0 + torch.exp(-15.0 * (ratio - 0.5)))
    q_a = 0.9879 / (1.0 + torch.exp(-22.0 * (angle_score - 0.8)))
    return q_g * q_a, source_g


def qabf(a: np.ndarray, b: np.ndarray, fused: np.ndarray) -> float:
    a_t, b_t, fused_t = _as_tensor(a), _as_tensor(b), _as_tensor(fused)
    q_a, weight_a = _edge_quality(a_t, fused_t)
    q_b, weight_b = _edge_quality(b_t, fused_t)
    numerator = (q_a * weight_a + q_b * weight_b).sum()
    denominator = (weight_a + weight_b).sum().clamp_min(1e-12)
    return float(numerator / denominator)


def all_metrics(a: np.ndarray, b: np.ndarray, fused: np.ndarray) -> dict[str, float]:
    return {
        "EN": entropy(fused),
        "SD": standard_deviation(fused),
        "SF": spatial_frequency(fused),
        "SCD": sum_correlations_difference(a, b, fused),
        "VIF": vif(a, b, fused),
        "Qabf": qabf(a, b, fused),
    }

