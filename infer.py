from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from mtg_fusion import MTGFusion
from mtg_fusion.data import PairedFusionDataset
from mtg_fusion.utils import choose_device, load_config, load_model_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MTG-Fusion inference")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--text-cache")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--restore-color", action="store_true")
    parser.add_argument("--no-text", action="store_true")
    return parser.parse_args()


def safe_name(sample_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id)


def pad_multiple(image: torch.Tensor, divisor: int = 4) -> tuple[torch.Tensor, int, int]:
    height, width = image.shape[-2:]
    pad_h = (divisor - height % divisor) % divisor
    pad_w = (divisor - width % divisor) % divisor
    mode = "reflect" if height > 1 and width > 1 else "replicate"
    return F.pad(image, (0, pad_w, 0, pad_h), mode=mode), height, width


def save_result(
    fused_y: torch.Tensor,
    chroma: torch.Tensor,
    path: Path,
    restore_color: bool,
) -> None:
    y = (fused_y.squeeze().clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
    if not restore_color:
        Image.fromarray(y, mode="L").save(path)
        return
    cbcr = (chroma.squeeze(0).clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
    ycbcr = np.stack((y, cbcr[0], cbcr[1]), axis=-1)
    Image.fromarray(ycbcr, mode="YCbCr").convert("RGB").save(path)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = choose_device(str(config.get("device", "cuda")))
    use_text = not args.no_text
    if use_text and not args.text_cache:
        raise ValueError("Full inference requires --text-cache; otherwise pass --no-text")
    dataset = PairedFusionDataset(
        args.manifest,
        text_cache=args.text_cache,
        crop_size=None,
        augment=False,
        require_text=use_text,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model = MTGFusion(**config["model"]).to(device)
    load_model_state(model, args.checkpoint)
    model.eval()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    amp_enabled = bool(config.get("inference_amp", config.get("amp", True))) and device.type == "cuda"

    with torch.inference_mode():
        for batch in tqdm(loader, desc="inference"):
            ir, height, width = pad_multiple(batch["ir"].to(device))
            vis, _, _ = pad_multiple(batch["vis"].to(device))
            text = None
            if use_text:
                text = {key: value.to(device) for key, value in batch["text"].items()}
            with torch.amp.autocast(device.type, enabled=amp_enabled):
                fused, _ = model(ir, vis, text=text, use_text=use_text)
            fused = fused[..., :height, :width]
            sample_id = str(batch["id"][0])
            save_result(
                fused,
                batch["chroma"][..., :height, :width],
                output_dir / f"{safe_name(sample_id)}.png",
                args.restore_color,
            )


if __name__ == "__main__":
    main()
