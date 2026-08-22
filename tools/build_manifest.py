from __future__ import annotations

import argparse
import json
from pathlib import Path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an aligned pair manifest")
    parser.add_argument("--ir-dir", required=True)
    parser.add_argument("--vis-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--color-source", choices=("a", "b"), default="b")
    return parser.parse_args()


def index_images(directory: Path, recursive: bool) -> dict[str, Path]:
    iterator = directory.rglob("*") if recursive else directory.glob("*")
    result: dict[str, Path] = {}
    for path in iterator:
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            key = path.stem.lower()
            if key in result:
                raise ValueError(f"Duplicate image stem '{key}' in {directory}")
            result[key] = path.resolve()
    return result


def main() -> None:
    args = parse_args()
    ir_images = index_images(Path(args.ir_dir), args.recursive)
    vis_images = index_images(Path(args.vis_dir), args.recursive)
    common = sorted(ir_images.keys() & vis_images.keys())
    if not common:
        raise RuntimeError("No aligned filename stems were found")
    only_ir = sorted(ir_images.keys() - vis_images.keys())
    only_vis = sorted(vis_images.keys() - ir_images.keys())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for key in common:
            row = {
                "id": key,
                "ir": str(ir_images[key]),
                "vis": str(vis_images[key]),
                "color_source": args.color_source,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"wrote={len(common)} unmatched_ir={len(only_ir)} "
        f"unmatched_vis={len(only_vis)} output={output}"
    )


if __name__ == "__main__":
    main()
