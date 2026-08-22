from __future__ import annotations

import argparse
from collections import defaultdict
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from mtg_fusion import MTGFusion
from mtg_fusion.data import PairedFusionDataset
from mtg_fusion.losses import FusionLoss
from mtg_fusion.metrics import all_metrics
from mtg_fusion.utils import (
    choose_device,
    load_config,
    load_model_state,
    save_checkpoint,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MTG-Fusion")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", type=int, choices=(1, 2), required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--text-cache")
    parser.add_argument("--val-manifest")
    parser.add_argument("--val-text-cache")
    parser.add_argument("--resume")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


METRIC_NAMES = ("EN", "SD", "SF", "SCD", "VIF", "Qabf")
SCORE_METRIC_NAMES = ("EN", "SD", "SF", "VIF", "Qabf")


def _uint8_image(tensor: torch.Tensor) -> np.ndarray:
    return (
        tensor.squeeze().clamp(0, 1).mul(255.0).round().byte().cpu().numpy()
    )


def validation_metrics(
    model: MTGFusion,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    use_text: bool,
    text_strength: float | None = None,
) -> dict[str, float]:
    model.eval()
    values: defaultdict[str, list[float]] = defaultdict(list)
    with torch.inference_mode():
        for batch in loader:
            ir = batch["ir"].to(device)
            vis = batch["vis"].to(device)
            text = None
            if use_text:
                text = {key: value.to(device) for key, value in batch["text"].items()}
            with torch.amp.autocast(device.type, enabled=amp_enabled):
                fused, _ = model(
                    ir,
                    vis,
                    text=text,
                    use_text=use_text,
                    text_strength=text_strength,
                )
            metrics = all_metrics(
                _uint8_image(ir[0]),
                _uint8_image(vis[0]),
                _uint8_image(fused[0]),
            )
            for name, value in metrics.items():
                values[name].append(value)
    return {name: float(np.mean(values[name])) for name in METRIC_NAMES}


def relative_validation_score(
    candidate: dict[str, float], baseline: dict[str, float]
) -> tuple[float, float]:
    ratios = {
        name: 1.0
        + (candidate[name] - baseline[name]) / max(abs(baseline[name]), 1e-6)
        for name in SCORE_METRIC_NAMES
    }
    score = float(np.mean(list(ratios.values())))
    return score, float(min(ratios.values()))


def validation_guard(
    candidate: dict[str, float],
    baseline: dict[str, float],
    minimum_ratio: float,
    scd_tolerance: float,
) -> tuple[bool, float, float, float]:
    score, stable_minimum_ratio = relative_validation_score(candidate, baseline)
    scd_delta = candidate["SCD"] - baseline["SCD"]
    passed = (
        stable_minimum_ratio >= minimum_ratio
        and scd_delta >= -scd_tolerance
    )
    return passed, score, stable_minimum_ratio, scd_delta


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config.get("seed", 42)))
    device = choose_device(str(config.get("device", "cuda")))
    use_text = args.stage == 2
    if use_text and not args.resume:
        raise ValueError("Stage 2 must resume from a converged Stage 1 checkpoint")
    if use_text and not args.val_manifest:
        raise ValueError("Stage 2 requires --val-manifest for guarded selection")

    dataset = PairedFusionDataset(
        args.manifest,
        text_cache=args.text_cache,
        crop_size=int(config.get("crop_size", 256)),
        augment=True,
        require_text=use_text,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 4)),
        shuffle=True,
        num_workers=int(config.get("num_workers", 4)),
        pin_memory=device.type == "cuda",
        drop_last=len(dataset) >= int(config.get("batch_size", 4)),
        persistent_workers=int(config.get("num_workers", 4)) > 0,
    )
    val_loader = None
    if args.val_manifest:
        val_dataset = PairedFusionDataset(
            args.val_manifest,
            text_cache=args.val_text_cache or args.text_cache,
            crop_size=None,
            augment=False,
            require_text=use_text,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=1, shuffle=False, num_workers=0
        )
    model = MTGFusion(**config["model"]).to(device)
    if args.resume:
        load_model_state(model, args.resume)
    teacher = None
    if use_text:
        teacher = copy.deepcopy(model).eval()
        teacher.requires_grad_(False)

    stage_config = config[f"stage{args.stage}"]
    finetune_output_layer = False
    if use_text:
        finetune_output_layer = bool(
            stage_config.get("finetune_output_layer", False)
        )
        model.configure_stage2_trainable(
            finetune_decoder=bool(stage_config.get("finetune_decoder", False)),
            finetune_output_layer=finetune_output_layer,
        )
    loss_config = dict(config["loss"])
    if not use_text:
        loss_config["blip_weight"] = 0.0
    criterion = FusionLoss(**loss_config).to(device)
    output_parameters = (
        [
            parameter
            for parameter in model.decoder.output[1].parameters()
            if parameter.requires_grad
        ]
        if use_text and finetune_output_layer
        else []
    )
    output_parameter_ids = {id(parameter) for parameter in output_parameters}
    primary_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in output_parameter_ids
    ] + [parameter for parameter in criterion.parameters() if parameter.requires_grad]
    trainable = primary_parameters + output_parameters
    if not trainable:
        raise RuntimeError("No trainable parameters were selected")
    optimizer_groups: list[dict[str, object]] = [
        {
            "params": primary_parameters,
            "lr": float(stage_config["learning_rate"]),
        }
    ]
    if output_parameters:
        optimizer_groups.append(
            {
                "params": output_parameters,
                "lr": float(
                    stage_config.get(
                        "output_learning_rate", stage_config["learning_rate"]
                    )
                ),
            }
        )
    optimizer = torch.optim.Adam(
        optimizer_groups,
        weight_decay=float(stage_config.get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(stage_config.get("lr_step", 20)),
        gamma=float(stage_config.get("lr_gamma", 0.5)),
    )
    amp_enabled = bool(config.get("amp", True)) and device.type == "cuda"
    validation_amp_enabled = bool(config.get("validation_amp", config.get("amp", True))) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    best_validation_score = 1.0
    total_epochs = int(stage_config["epochs"])
    warmup_epochs = int(stage_config.get("text_warmup_epochs", 0)) if use_text else 0
    consistency_start = (
        float(stage_config.get("consistency_start", 1.0)) if use_text else 0.0
    )
    consistency_end = (
        float(stage_config.get("consistency_end", 0.5)) if use_text else 0.0
    )
    validation_strengths = (
        [float(value) for value in stage_config.get("validation_text_strengths", [1.0])]
        if use_text
        else []
    )
    if any(not 0.0 < value <= 1.0 for value in validation_strengths):
        raise ValueError("validation_text_strengths values must be in (0, 1]")
    trainable_count = sum(parameter.numel() for parameter in trainable)
    print(
        f"stage={args.stage} trainable_parameters={trainable_count:,} "
        f"text_warmup_epochs={warmup_epochs} "
        f"consistency={consistency_start:.3f}->{consistency_end:.3f}"
    )
    baseline_metrics = None
    if use_text:
        assert teacher is not None and val_loader is not None
        baseline_metrics = validation_metrics(
            teacher, val_loader, device, validation_amp_enabled, use_text=False
        )
        model.inference_text_strength.fill_(0.0)
        save_checkpoint(
            output_dir / "best.pt",
            model,
            optimizer,
            scheduler,
            0,
            config,
            -best_validation_score,
        )
        model.inference_text_strength.fill_(1.0)
        print(
            "validation_baseline "
            + " ".join(f"{name}={baseline_metrics[name]:.4f}" for name in METRIC_NAMES)
        )

    for epoch in range(1, total_epochs + 1):
        text_strength = (
            min(1.0, epoch / max(1, warmup_epochs))
            if use_text and warmup_epochs > 0
            else 1.0
        )
        schedule_progress = (epoch - 1) / max(1, total_epochs - 1)
        consistency_weight = (
            consistency_start
            + schedule_progress * (consistency_end - consistency_start)
        )
        model.train()
        totals: defaultdict[str, float] = defaultdict(float)
        for batch in loader:
            ir = batch["ir"].to(device, non_blocking=True)
            vis = batch["vis"].to(device, non_blocking=True)
            text = None
            if use_text:
                text = {
                    key: value.to(device, non_blocking=True)
                    for key, value in batch["text"].items()
                }
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=amp_enabled):
                visual_reference = None
                if use_text:
                    assert teacher is not None
                    with torch.no_grad():
                        visual_reference, _ = teacher(ir, vis, use_text=False)
                fused, _ = model(
                    ir,
                    vis,
                    text=text,
                    use_text=use_text,
                    text_strength=text_strength,
                )
                loss, components = criterion(fused, ir, vis, text=text)
                consistency_loss = fused.new_zeros(())
                if visual_reference is not None:
                    consistency_loss = F.l1_loss(fused, visual_reference)
                    loss = loss + consistency_weight * consistency_loss
                components["consistency"] = consistency_loss.detach()
                components["total"] = loss.detach()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            for name, value in components.items():
                totals[name] += float(value)
        scheduler.step()

        selection_loss = totals["fusion"] / max(1, len(loader))
        save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            config,
            min(best_loss, selection_loss),
        )
        accepted = False
        validation_summary = ""
        if use_text:
            assert baseline_metrics is not None and val_loader is not None
            required_ratio = float(stage_config.get("validation_min_ratio", 1.0))
            scd_tolerance = float(stage_config.get("validation_scd_tolerance", 0.0))
            score_margin = float(stage_config.get("validation_score_margin", 0.0))
            records: list[
                tuple[float, float, float, float, bool, dict[str, float]]
            ] = []
            for strength in validation_strengths:
                candidate_metrics = validation_metrics(
                    model,
                    val_loader,
                    device,
                    validation_amp_enabled,
                    use_text=True,
                    text_strength=strength,
                )
                guard_passed, score, minimum_ratio, scd_delta = validation_guard(
                    candidate_metrics,
                    baseline_metrics,
                    required_ratio,
                    scd_tolerance,
                )
                records.append(
                    (
                        score,
                        minimum_ratio,
                        scd_delta,
                        strength,
                        guard_passed,
                        candidate_metrics,
                    )
                )
            guarded = [
                record
                for record in records
                if record[4] and record[0] >= 1.0 + score_margin
            ]
            selected = max(guarded, key=lambda record: record[0]) if guarded else None
            reported = selected or max(records, key=lambda record: record[0])
            score, minimum_ratio, scd_delta, selected_strength, _, _ = reported
            accepted = selected is not None and score > best_validation_score
            if accepted:
                best_validation_score = score
                model.inference_text_strength.fill_(selected_strength)
                save_checkpoint(
                    output_dir / "best.pt",
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    config,
                    -best_validation_score,
                )
                model.inference_text_strength.fill_(1.0)
            validation_summary = (
                f" val_strength={selected_strength:.3f} "
                f"val_score={score:.6f} val_min_ratio={minimum_ratio:.6f} "
                f"val_scd_delta={scd_delta:+.6f} "
                f"accepted={int(accepted)}"
            )
        elif selection_loss < best_loss:
            best_loss = selection_loss
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                config,
                best_loss,
            )
        summary = " ".join(
            f"{name}={value / max(1, len(loader)):.4f}"
            for name, value in sorted(totals.items())
        )
        learning_rates = scheduler.get_last_lr()
        lr_summary = f"lr={learning_rates[0]:.3e}"
        if len(learning_rates) > 1:
            lr_summary += f" output_lr={learning_rates[1]:.3e}"
        print(
            f"epoch={epoch} {lr_summary} "
            f"text_strength={text_strength:.3f} "
            f"consistency_weight={consistency_weight:.3f} "
            f"selection={selection_loss:.4f}{validation_summary} {summary}"
        )


if __name__ == "__main__":
    main()
