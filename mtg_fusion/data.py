from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset
import torch.nn.functional as F


TEXT_KEYS = ("t_a", "t_b", "t_ab", "t_as", "t_bs", "t_gt")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    manifest = Path(path)
    rows: list[dict[str, Any]] = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}: {manifest}") from error
            for key in ("id", "ir", "vis"):
                if key not in row:
                    raise KeyError(f"Manifest line {line_number} is missing '{key}'")
            rows.append(row)
    if not rows:
        raise ValueError(f"Manifest contains no samples: {manifest}")
    return rows


def resolve_path(value: str, manifest_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (manifest_dir / path).resolve()


def _image_tensor(image: Image.Image) -> Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = array[..., None]
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _pad_to(tensor: Tensor, height: int, width: int) -> Tensor:
    pad_h = max(0, height - tensor.shape[-2])
    pad_w = max(0, width - tensor.shape[-1])
    if pad_h == 0 and pad_w == 0:
        return tensor
    mode = "reflect" if tensor.shape[-2] > 1 and tensor.shape[-1] > 1 else "replicate"
    return F.pad(tensor, (0, pad_w, 0, pad_h), mode=mode)


class PairedFusionDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        text_cache: str | Path | None = None,
        crop_size: int | None = None,
        augment: bool = False,
        require_text: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest).resolve()
        self.rows = read_jsonl(self.manifest_path)
        self.crop_size = crop_size
        self.augment = augment
        self.require_text = require_text
        self.text_cache: dict[str, dict[str, Tensor]] | None = None
        if text_cache:
            loaded = torch.load(text_cache, map_location="cpu", weights_only=True)
            if not isinstance(loaded, dict):
                raise TypeError("Text cache must be a dictionary keyed by sample ID")
            self.text_cache = loaded
        if require_text and self.text_cache is None:
            raise ValueError("Stage 2 requires --text-cache")

    def __len__(self) -> int:
        return len(self.rows)

    def _crop(self, tensors: list[Tensor]) -> list[Tensor]:
        if self.crop_size is None:
            return tensors
        size = self.crop_size
        tensors = [_pad_to(tensor, size, size) for tensor in tensors]
        height, width = tensors[0].shape[-2:]
        top = random.randint(0, height - size) if self.augment else (height - size) // 2
        left = random.randint(0, width - size) if self.augment else (width - size) // 2
        return [tensor[..., top : top + size, left : left + size] for tensor in tensors]

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        base = self.manifest_path.parent
        ir_path = resolve_path(str(row["ir"]), base)
        vis_path = resolve_path(str(row["vis"]), base)
        with Image.open(ir_path) as image:
            ycbcr_a = _image_tensor(image.convert("RGB").convert("YCbCr"))
        with Image.open(vis_path) as image:
            ycbcr_b = _image_tensor(image.convert("RGB").convert("YCbCr"))
        if ycbcr_a.shape[-2:] != ycbcr_b.shape[-2:]:
            raise ValueError(
                f"Unaligned pair '{row['id']}': A={ycbcr_a.shape[-2:]}, B={ycbcr_b.shape[-2:]}"
            )
        ir = ycbcr_a[:1]
        vis_y = ycbcr_b[:1]
        color_source = str(row.get("color_source", "b")).lower()
        if color_source not in {"a", "b"}:
            raise ValueError(f"color_source must be 'a' or 'b', got '{color_source}'")
        chroma = ycbcr_a[1:] if color_source == "a" else ycbcr_b[1:]
        ir, vis_y, chroma = self._crop([ir, vis_y, chroma])
        if self.augment:
            if random.random() < 0.5:
                ir, vis_y, chroma = [torch.flip(x, dims=(-1,)) for x in (ir, vis_y, chroma)]
            if random.random() < 0.5:
                ir, vis_y, chroma = [torch.flip(x, dims=(-2,)) for x in (ir, vis_y, chroma)]

        result: dict[str, Any] = {
            "id": str(row["id"]),
            "ir": ir,
            "vis": vis_y,
            "chroma": chroma,
        }
        if self.text_cache is not None:
            sample_id = str(row["id"])
            if sample_id not in self.text_cache:
                raise KeyError(f"Text cache has no entry for '{sample_id}'")
            cached = self.text_cache[sample_id]
            missing = [key for key in TEXT_KEYS if key not in cached]
            if missing:
                raise KeyError(f"Text cache entry '{sample_id}' is missing {missing}")
            result["text"] = {key: cached[key].float() for key in TEXT_KEYS}
        return result
