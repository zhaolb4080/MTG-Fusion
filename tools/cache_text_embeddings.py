from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm


TEXT_KEYS = ("t_a", "t_b", "t_ab", "t_as", "t_bs", "t_gt")
DEFAULT_BLIP_ITM_REVISION = "98cf803d8586e8e8b3ec527885ac50c422d514ae"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache frozen BLIP text embeddings")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="Salesforce/blip-itm-base-coco")
    parser.add_argument(
        "--revision",
        default=DEFAULT_BLIP_ITM_REVISION,
        help="Model revision containing safetensors weights",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from transformers import BlipModel, BlipProcessor

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    processor = BlipProcessor.from_pretrained(
        args.model, revision=args.revision, use_fast=False
    )
    model = BlipModel.from_pretrained(
        args.model, revision=args.revision, use_safetensors=True
    ).to(device).eval()
    model.requires_grad_(False)
    rows = [
        json.loads(line)
        for line in Path(args.manifest).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    flattened: list[tuple[str, str, str]] = []
    for row in rows:
        missing = [key for key in TEXT_KEYS if not isinstance(row.get(key), str)]
        if missing:
            raise KeyError(f"Sample '{row.get('id')}' is missing descriptions: {missing}")
        flattened.extend((str(row["id"]), key, row[key]) for key in TEXT_KEYS)

    cache: dict[str, dict[str, torch.Tensor]] = {}
    with torch.inference_mode():
        for start in tqdm(range(0, len(flattened), args.batch_size), desc="BLIP text"):
            batch = flattened[start : start + args.batch_size]
            encoded = processor(
                text=[item[2] for item in batch],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            text_output = model.get_text_features(
                input_ids=encoded["input_ids"],
                attention_mask=encoded.get("attention_mask"),
            )
            features = (
                text_output.pooler_output
                if hasattr(text_output, "pooler_output")
                else text_output
            )
            features = F.normalize(features, dim=-1).cpu()
            for (sample_id, key, _), feature in zip(batch, features):
                cache.setdefault(sample_id, {})[key] = feature
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, output)
    dimension = next(iter(next(iter(cache.values())).values())).numel()
    print(f"samples={len(cache)} dimension={dimension} output={output}")


if __name__ == "__main__":
    main()
