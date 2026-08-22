from __future__ import annotations

import json
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import nn
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError("Configuration root must be a mapping")
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    config: dict[str, Any],
    best_loss: float,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "config": config,
            "best_loss": best_loss,
        },
        temporary,
    )
    os.replace(temporary, destination)


def load_model_state(model: nn.Module, checkpoint: str | Path) -> dict[str, Any]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    parameters = state["model"] if isinstance(state, dict) and "model" in state else state
    incompatible = model.load_state_dict(parameters, strict=False)
    allowed_missing_prefixes = (
        "cmsf_text_adapter.",
        "sms_text_adapter.",
        "inference_text_strength",
    )
    unexpected_missing = {
        key
        for key in incompatible.missing_keys
        if not key.startswith(allowed_missing_prefixes)
    }
    if unexpected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint is incompatible: "
            f"missing={sorted(unexpected_missing)} "
            f"unexpected={sorted(incompatible.unexpected_keys)}"
        )
    return state if isinstance(state, dict) else {}


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
