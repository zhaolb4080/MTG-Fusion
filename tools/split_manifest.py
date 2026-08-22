from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import random
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a JSONL manifest into training and validation rows"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--val-output", required=True)
    parser.add_argument("--val-count", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stratify-last-character",
        action="store_true",
        help="Preserve groups such as MSRS day/night ID suffixes",
    )
    return parser.parse_args()


def allocate_counts(group_sizes: dict[str, int], total: int) -> dict[str, int]:
    population = sum(group_sizes.values())
    raw = {key: total * size / population for key, size in group_sizes.items()}
    counts = {key: int(value) for key, value in raw.items()}
    remaining = total - sum(counts.values())
    order = sorted(raw, key=lambda key: raw[key] - counts[key], reverse=True)
    for key in order[:remaining]:
        counts[key] += 1
    return counts


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    source = Path(args.input)
    rows = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not 0 < args.val_count < len(rows):
        raise ValueError("--val-count must be between 1 and number of rows - 1")
    rng = random.Random(args.seed)
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sample_id = str(row["id"])
        key = sample_id[-1].lower() if args.stratify_last_character else "all"
        groups[key].append(row)
    counts = allocate_counts(
        {key: len(values) for key, values in groups.items()}, args.val_count
    )
    validation_ids: set[str] = set()
    for key, values in groups.items():
        shuffled = values.copy()
        rng.shuffle(shuffled)
        validation_ids.update(str(row["id"]) for row in shuffled[: counts[key]])
    train_rows = [row for row in rows if str(row["id"]) not in validation_ids]
    val_rows = [row for row in rows if str(row["id"]) in validation_ids]
    write_rows(Path(args.train_output), train_rows)
    write_rows(Path(args.val_output), val_rows)
    group_summary = " ".join(
        f"{key}={sum(str(row['id']) in validation_ids for row in groups[key])}"
        for key in sorted(groups)
    )
    print(
        f"train={len(train_rows)} validation={len(val_rows)} "
        f"seed={args.seed} {group_summary}"
    )


if __name__ == "__main__":
    main()
