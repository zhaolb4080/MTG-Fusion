from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
from PIL import Image
from tqdm import tqdm

from mtg_fusion.data import read_jsonl, resolve_path
from mtg_fusion.metrics import all_metrics
from mtg_fusion.utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fusion results")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--fused-dir", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def safe_name(sample_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id)


def luminance(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.float32)


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    rows = read_jsonl(manifest_path)
    fused_dir = Path(args.fused_dir)
    per_image: dict[str, dict[str, float]] = {}
    for row in tqdm(rows, desc="metrics"):
        sample_id = str(row["id"])
        a = luminance(resolve_path(str(row["ir"]), manifest_path.parent))
        b = luminance(resolve_path(str(row["vis"]), manifest_path.parent))
        fused_path = fused_dir / f"{safe_name(sample_id)}.png"
        if not fused_path.exists():
            raise FileNotFoundError(f"Missing fused result: {fused_path}")
        fused = luminance(fused_path)
        if a.shape != b.shape or a.shape != fused.shape:
            raise ValueError(
                f"Shape mismatch for {sample_id}: A={a.shape}, B={b.shape}, F={fused.shape}"
            )
        per_image[sample_id] = all_metrics(a, b, fused)
    names = next(iter(per_image.values())).keys()
    mean = {
        name: float(np.mean([metrics[name] for metrics in per_image.values()]))
        for name in names
    }
    write_json(args.output, {"mean": mean, "per_image": per_image})
    print(" ".join(f"{name}={value:.4f}" for name, value in mean.items()))


if __name__ == "__main__":
    main()

