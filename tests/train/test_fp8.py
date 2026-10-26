"""Testes de FP8 via torchao (alavanca G)."""
import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from sore.model import SoreLM, get_preset
from sore.train.config import TrainConfig
from sore.train.fp8 import apply_fp8, fp8_supported
from sore.train.optim import split_param_groups
from sore.train.trainer import train


def _eligible_linear_names(model: nn.Module) -> list[str]:
    """Nomes dos nn.Linear elegíveis para conversão FP8 segundo o filtro da spec."""
    eligible = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if not (name.startswith("blocks.") or name.startswith("mtp.")):
            continue
        if "res_mixer" in name or "res_ffn" in name:
            continue
        if mod.in_features % 16 == 0 and mod.out_features % 16 == 0:
            eligible.append(name)
    return eligible


def test_train_config_fp8_validation(tmp_path: Path):
    """TrainConfig valida os valores de fp8 e fp8_emulate."""
    base_kwargs = {
        "model_preset": "tiny",
        "data_paths": [str(tmp_path / "dummy_*.bin")],
        "eos_id": 0,
        "grad_accum": 1,
        "stages": [{"until_tokens": 1024, "seq_len": 128, "micro_batch": 1}],
        "total_tokens": 1024,
        "ckpt_every_tokens": 1024,
        "out_dir": str(tmp_path / "run"),
    }
    cfg_off = TrainConfig(**base_kwargs)
    assert cfg_off.fp8 == "off"
    assert not cfg_off.fp8_emulate

    cfg_tensor = TrainConfig(**base_kwargs, fp8="tensorwise", fp8_emulate=True)
    assert cfg_tensor.fp8 == "tensorwise"
    assert cfg_tensor.fp8_emulate

    cfg_row = TrainConfig(**base_kwargs, fp8="rowwise")
    assert cfg_row.fp8 == "rowwise"

    with pytest.raises(ValueError, match="fp8"):
        TrainConfig(**base_kwargs, fp8="invalid_recipe")


def test_fp8_supported():
    """fp8_supported devolve False em CPU e confere capacidade >= (8, 9) em CUDA."""
    assert not fp8_supported(torch.device("cpu"))
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        assert fp8_supported(torch.device("cuda")) == (cap >= (8, 9))


def test_apply_fp8_invalid_recipe():
    """apply_fp8 recusa receita desconhecida com ValueError."""
    model = SoreLM(get_preset("tiny"))
    with pytest.raises(ValueError, match="receita|recipe|fp8"):
        apply_fp8(model, "invalid_recipe", emulate=True)


def test_apply_fp8_filter_and_names(device: str):
    """Filtro de módulos elegíveis, preservação de state_dict e nomes no split_param_groups."""
    from torchao.float8.float8_linear import Float8Linear

    dev = torch.device(device)
    cfg = get_preset("tiny", d_model=128, residual="hyper")
    model = SoreLM(cfg).to(dev)

    eligible_names = _eligible_linear_names(model)
    assert len(eligible_names) > 0

    sd_before = {k: v.clone() for k, v in model.state_dict().items()}
    muon_before, adamw_before = split_param_groups(model)
    muon_names_before = {name for name, p in model.named_parameters() if any(p is mp for mp in muon_before)}
    adamw_names_before = {name for name, p in model.named_parameters() if any(p is ap for ap in adamw_before)}

    n_converted = apply_fp8(model, "tensorwise", emulate=True)
    assert n_converted == len(eligible_names)

    # Verifica que exatamente os módulos elegíveis foram convertidos
    converted_names = [name for name, mod in model.named_modules() if isinstance(mod, Float8Linear)]
    assert sorted(converted_names) == sorted(eligible_names)

    # a_proj e b_proj não foram convertidos (v_heads não é múltiplo de 16)
    for name, mod in model.named_modules():
        if name.endswith(".a_proj") or name.endswith(".b_proj"):
            assert isinstance(mod, nn.Linear) and not isinstance(mod, Float8Linear)

    # res_mixer e res_ffn (hyper-residual) não foram convertidos
    for name, mod in model.named_modules():
        if "res_mixer" in name or "res_ffn" in name:
            assert not isinstance(mod, Float8Linear)

    # embed e lm_head não foram convertidos
    assert not isinstance(model.embed, Float8Linear)
    if model.lm_head is not None:
        assert not isinstance(model.lm_head, Float8Linear)

    # state_dict tem exatamente as mesmas chaves
    sd_after = model.state_dict()
    assert set(sd_before.keys()) == set(sd_after.keys())
    for k in sd_before:
        assert sd_before[k].shape == sd_after[k].shape

    # split_param_groups dá os mesmos conjuntos de nomes
    muon_after, adamw_after = split_param_groups(model)
    muon_names_after = {name for name, p in model.named_parameters() if any(p is mp for mp in muon_after)}
    adamw_names_after = {name for name, p in model.named_parameters() if any(p is ap for ap in adamw_after)}
    assert muon_names_before == muon_names_after
    assert adamw_names_before == adamw_names_after


def test_fp8_emulated_loss_close(device: str):
    """tiny com emulate=True tem loss finita e próxima (< 5% rel) da loss base."""
    dev = torch.device(device)
    cfg = get_preset("tiny", d_model=128)

    torch.manual_seed(42)
    model_base = SoreLM(cfg).to(dev)

    torch.manual_seed(42)
    model_fp8 = SoreLM(cfg).to(dev)

    apply_fp8(model_fp8, "tensorwise", emulate=True)

    seq_len = 64
    x = torch.randint(1, cfg.vocab_size, (1, seq_len), device=dev)
    labels = torch.randint(0, cfg.vocab_size, (1, seq_len), device=dev)
    cu_seqlens = torch.tensor([0, seq_len // 2, seq_len], dtype=torch.long, device=dev)

    # Base forward + backward
    out_base = model_base(x, cu_seqlens=cu_seqlens, labels=labels)
    assert out_base.loss is not None
    loss_base = out_base.loss
    assert math.isfinite(loss_base.item())
    loss_base.backward()

    # FP8 emulated forward + backward
    out_fp8 = model_fp8(x, cu_seqlens=cu_seqlens, labels=labels)
    assert out_fp8.loss is not None
    loss_fp8 = out_fp8.loss
    assert math.isfinite(loss_fp8.item())
    loss_fp8.backward()

    rel_diff = abs(loss_fp8.item() - loss_base.item()) / loss_base.item()
    assert rel_diff < 0.05, f"rel diff {rel_diff:.4f} >= 5% (fp8={loss_fp8.item():.4f}, base={loss_base.item():.4f})"

    # Gradientes de pesos convertidos são finitos
    for name, p in model_fp8.named_parameters():
        if p.requires_grad and p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"gradiente não finito em {name}"


@pytest.mark.cuda
def test_fp8_refuses_without_hardware():
    """Review Focus 5: apply_fp8 sem emulação em GPU < (8, 9) levanta RuntimeError com 'FP8' na mensagem."""
    if torch.cuda.get_device_capability() >= (8, 9):
        pytest.skip("hardware suporta FP8 nativo (sm89+)")

    model = SoreLM(get_preset("tiny")).to("cuda")
    with pytest.raises(RuntimeError) as exc_info:
        apply_fp8(model, "tensorwise", emulate=False)
    assert "FP8" in str(exc_info.value)


def _make_dummy_shard(path: Path, n_tokens: int = 4096, vocab_size: int = 512) -> None:
    """Gera um shard .bin simples com sidecar .json."""
    import numpy as np
    from sore.train.data import write_token_shard

    tokens = np.random.randint(0, vocab_size, size=n_tokens, dtype=np.uint32)
    write_token_shard(path, tokens)


def test_train_fp8_resume_cross_toggle(tmp_path: Path, device: str):
    """Treino tiny com fp8 retoma com fp8='off' e o inverso sem erro."""
    shard = tmp_path / "data_000.bin"
    _make_dummy_shard(shard, n_tokens=8192, vocab_size=512)

    can_compile = device == "cuda"
    out_dir_1 = tmp_path / "run_fp8_to_off"
    cfg_fp8 = TrainConfig(
        model_preset="tiny",
        data_paths=[str(shard)],
        eos_id=0,
        grad_accum=1,
        stages=[{"until_tokens": 8192, "seq_len": 128, "micro_batch": 1}],
        total_tokens=8192,
        ckpt_every_tokens=256,
        warmup_steps=1,
        compile=can_compile,
        out_dir=str(out_dir_1),
        device=device,
        fp8="tensorwise",
        fp8_emulate=True,
    )
    # Roda 2 passos com FP8
    train(cfg_fp8, max_steps=2)
    assert (out_dir_1 / "last.pt").exists()

    # Retoma com fp8="off"
    cfg_off = TrainConfig(
        model_preset="tiny",
        data_paths=[str(shard)],
        eos_id=0,
        grad_accum=1,
        stages=[{"until_tokens": 8192, "seq_len": 128, "micro_batch": 1}],
        total_tokens=8192,
        ckpt_every_tokens=256,
        warmup_steps=1,
        compile=can_compile,
        out_dir=str(out_dir_1),
        device=device,
        fp8="off",
    )
    summary = train(cfg_off, resume=out_dir_1 / "last.pt", max_steps=2)
    assert summary.steps == 4

    # E o caminho inverso: treina 2 passos com off, retoma com fp8
    out_dir_2 = tmp_path / "run_off_to_fp8"
    cfg_off_2 = TrainConfig(
        model_preset="tiny",
        data_paths=[str(shard)],
        eos_id=0,
        grad_accum=1,
        stages=[{"until_tokens": 8192, "seq_len": 128, "micro_batch": 1}],
        total_tokens=8192,
        ckpt_every_tokens=256,
        warmup_steps=1,
        compile=can_compile,
        out_dir=str(out_dir_2),
        device=device,
        fp8="off",
    )
    train(cfg_off_2, max_steps=2)
    assert (out_dir_2 / "last.pt").exists()

    cfg_fp8_2 = TrainConfig(
        model_preset="tiny",
        data_paths=[str(shard)],
        eos_id=0,
        grad_accum=1,
        stages=[{"until_tokens": 8192, "seq_len": 128, "micro_batch": 1}],
        total_tokens=8192,
        ckpt_every_tokens=256,
        warmup_steps=1,
        compile=can_compile,
        out_dir=str(out_dir_2),
        device=device,
        fp8="tensorwise",
        fp8_emulate=True,
    )
    summary_2 = train(cfg_fp8_2, resume=out_dir_2 / "last.pt", max_steps=2)
    assert summary_2.steps == 4


@pytest.mark.fp8
def test_fp8_real_smoke():
    """Smoke test real na 4090: s150m com seq 2048, 3 passos com fp8='rowwise'."""
    device = torch.device("cuda")
    cfg = get_preset("s150m")
    model = SoreLM(cfg).to(device)
    apply_fp8(model, "rowwise", emulate=False)

    seq_len = 2048
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for _ in range(3):
        x = torch.randint(1, cfg.vocab_size, (1, seq_len), device=device)
        labels = torch.randint(0, cfg.vocab_size, (1, seq_len), device=device)
        cu_seqlens = torch.tensor([0, seq_len // 2, seq_len], dtype=torch.long, device=device)
        out = model(x, cu_seqlens=cu_seqlens, labels=labels)
        assert out.loss is not None
        assert math.isfinite(out.loss.item())
        opt.zero_grad()
        out.loss.backward()
        opt.step()
