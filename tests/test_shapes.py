import torch

from mtg_fusion import MTGFusion
from mtg_fusion.losses import FusionLoss
from mtg_fusion.utils import load_model_state


def make_model() -> MTGFusion:
    return MTGFusion(
        base_channels=8,
        feature_channels=16,
        retention_blocks=1,
        retention_heads=2,
        retention_head_dim=8,
        text_dim=32,
        max_text_side=16,
        match_size=4,
    )


def test_visual_forward_shape() -> None:
    model = make_model()
    a = torch.rand(2, 1, 32, 40)
    b = torch.rand(2, 1, 32, 40)
    fused, maps = model(a, b, use_text=False)
    assert fused.shape == a.shape
    assert maps["cmsf"].shape[-2:] == (8, 10)


def test_text_forward_shape_and_backward() -> None:
    model = make_model()
    a = torch.rand(1, 1, 32, 32)
    b = torch.rand(1, 1, 32, 32)
    text = {
        key: torch.rand(1, 32)
        for key in ("t_a", "t_b", "t_ab", "t_as", "t_bs", "t_gt")
    }
    fused, maps = model(a, b, text=text, use_text=True)
    loss, _ = FusionLoss(blip_weight=0.0)(fused, a, b, text=text)
    loss.backward()
    assert fused.shape == a.shape
    assert maps["cmsf_weight"].shape == (1, 1, 8, 8)
    assert maps["text_cmsf_gate"].shape == (1, 1, 8, 8)
    assert maps["text_sms_gate"].shape == (1, 1, 8, 8)
    assert torch.all(
        (0.0 <= maps["text_cmsf_gate"]) & (maps["text_cmsf_gate"] <= 1.0)
    )
    assert torch.all(
        (0.0 <= maps["text_sms_gate"]) & (maps["text_sms_gate"] <= 1.0)
    )
    assert torch.isfinite(loss)


def test_stage2_freezes_visual_path() -> None:
    model = make_model()
    model.configure_stage2_trainable(finetune_decoder=True)
    trainable = {name for name, value in model.named_parameters() if value.requires_grad}
    assert any(name.startswith("cmsf_text_adapter.") for name in trainable)
    assert any(name.startswith("sms_text_adapter.") for name in trainable)
    assert any(name.startswith("tgce.") for name in trainable)
    assert any(name.startswith("tgsf.") for name in trainable)
    assert not any(name.startswith("encoder_a.") for name in trainable)
    assert any(name.startswith("decoder.") for name in trainable)


def test_zero_text_strength_recovers_visual_output() -> None:
    model = make_model().eval()
    a = torch.rand(1, 1, 32, 32)
    b = torch.rand(1, 1, 32, 32)
    text = {
        key: torch.rand(1, 32)
        for key in ("t_a", "t_b", "t_ab", "t_as", "t_bs", "t_gt")
    }
    with torch.no_grad():
        visual, _ = model(a, b, use_text=False)
        conditioned, _ = model(
            a, b, text=text, use_text=True, text_strength=0.0
        )
    assert torch.allclose(conditioned, visual, atol=1e-6, rtol=1e-5)


def test_old_stage1_checkpoint_allows_missing_text_adapters(tmp_path) -> None:
    source = make_model()
    old_state = {
        key: value
        for key, value in source.state_dict().items()
        if not key.startswith(
            (
                "cmsf_text_adapter.",
                "sms_text_adapter.",
                "inference_text_strength",
            )
        )
    }
    checkpoint = tmp_path / "old_stage1.pt"
    torch.save({"model": old_state}, checkpoint)
    load_model_state(make_model(), checkpoint)
