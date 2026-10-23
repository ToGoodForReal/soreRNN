"""Testes da divisão Muon/AdamW e dos grupos de parâmetros."""
import pytest
import torch

from sore.model import SoreLM, get_preset
from sore.train.optim import build_optimizers, split_param_groups

VARIANTS = [{}, {"residual": "hyper"}, {"tie_embeddings": False}]
IDS = ["plain", "hyper", "untied"]


def _model(seed=0, **overrides):
    torch.manual_seed(seed)
    return SoreLM(get_preset("tiny", **overrides))


def _ids(p):
    return {id(x) for x in p}


@pytest.mark.parametrize("overrides", VARIANTS, ids=IDS)
def test_split(overrides):
    model = _model(**overrides)
    muon, adamw = split_param_groups(model)
    named = dict(model.named_parameters())
    muon_ids, adamw_ids = _ids(muon), _ids(adamw)

    # nenhum repetido, nenhum faltando, sem interseção
    assert len(muon_ids) == len(muon) and len(adamw_ids) == len(adamw)
    assert not (muon_ids & adamw_ids)
    assert muon_ids | adamw_ids == _ids(model.parameters())

    assert id(named["embed.weight"]) in adamw_ids
    assert id(named["blocks.0.mixer.q_proj.weight"]) in muon_ids
    assert id(named["blocks.0.mixer.conv_q"]) in adamw_ids
    assert id(named["blocks.0.ffn.down_proj.weight"]) in muon_ids
    assert id(named["mtp.proj.weight"]) in muon_ids
    assert id(named["mtp.block.ffn.gate_proj.weight"]) in muon_ids
    assert id(named["blocks.0.mixer.A_log"]) in adamw_ids
    assert id(named["blocks.0.norm1.weight"]) in adamw_ids

    # Muon só aceita 2-D e nada de conv ou residual
    for name, p in named.items():
        if id(p) in muon_ids:
            assert p.ndim == 2 and name.startswith(("blocks.", "mtp."))
            assert "conv" not in name and "res_mixer" not in name and "res_ffn" not in name


def test_split_hyper_residual_goes_to_adamw():
    # R4: os pesos 2-D do hyper-residual (down/up/write_proj) ficam no AdamW
    model = _model(residual="hyper")
    muon, adamw = split_param_groups(model)
    muon_ids, adamw_ids = _ids(muon), _ids(adamw)
    hyper = {
        n: p for n, p in model.named_parameters() if "res_mixer" in n or "res_ffn" in n or n.startswith("stream_final")
    }
    assert any(p.ndim == 2 for p in hyper.values())
    for name, p in hyper.items():
        assert id(p) in adamw_ids and id(p) not in muon_ids, name


def test_split_untied_head_goes_to_adamw():
    model = _model(tie_embeddings=False)
    muon, adamw = split_param_groups(model)
    assert id(model.lm_head.weight) in _ids(adamw)
    assert id(model.lm_head.weight) not in _ids(muon)


def test_split_skips_frozen_params():
    model = _model()
    model.embed.weight.requires_grad_(False)
    model.blocks[0].mixer.q_proj.weight.requires_grad_(False)
    muon, adamw = split_param_groups(model)
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    assert _ids(muon) | _ids(adamw) == trainable


def test_build_optimizers_groups():
    model = _model()
    muon_opt, adamw_opt = build_optimizers(model, lr_muon=0.02, lr_adamw=3e-3, weight_decay=0.1)
    assert isinstance(muon_opt, torch.optim.Muon)
    assert isinstance(adamw_opt, torch.optim.AdamW)

    (mg,) = muon_opt.param_groups
    assert mg["lr"] == mg["initial_lr"] == 0.02
    assert mg["weight_decay"] == 0.1
    assert mg["adjust_lr_fn"] == "match_rms_adamw"
    assert mg["momentum"] == 0.95 and mg["nesterov"] is True and mg["ns_steps"] == 5

    decay, no_decay = adamw_opt.param_groups
    for g in (decay, no_decay):
        assert g["lr"] == g["initial_lr"] == 3e-3
        assert g["betas"] == (0.9, 0.95) and g["eps"] == 1e-8
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    assert all(p.ndim >= 2 for p in decay["params"])
    assert all(p.ndim < 2 for p in no_decay["params"])


def test_build_optimizers_omits_muon_without_matrices():
    model = _model()
    for name, p in model.named_parameters():
        if name.startswith(("blocks.", "mtp.")) and p.ndim == 2 and "conv" not in name:
            p.requires_grad_(False)
    opts = build_optimizers(model, lr_muon=0.02, lr_adamw=3e-3, weight_decay=0.1)
    assert len(opts) == 1 and isinstance(opts[0], torch.optim.AdamW)


@pytest.mark.parametrize("overrides", VARIANTS, ids=IDS)
def test_one_step_updates_all(overrides):
    model = _model(**overrides)
    ids = torch.randint(0, 512, (1, 30), generator=torch.Generator().manual_seed(1))
    labels = torch.randint(0, 512, (1, 30), generator=torch.Generator().manual_seed(2))
    out = model(ids, labels=labels)
    out.loss.backward()

    with_grad = {n: p for n, p in model.named_parameters() if p.grad is not None}
    assert with_grad
    before = {n: p.detach().clone() for n, p in with_grad.items()}
    for opt in build_optimizers(model, lr_muon=0.02, lr_adamw=3e-3, weight_decay=0.1):
        opt.step()
    unchanged = [n for n, p in with_grad.items() if torch.equal(p, before[n])]
    assert not unchanged, unchanged


@pytest.mark.cuda
def test_one_step_on_cuda():
    # no CUDA o AdamW usa fused=True; o passo tem que rodar e atualizar os parâmetros
    model = _model().cuda()
    ids = torch.randint(0, 512, (1, 30), device="cuda")
    model(ids, labels=ids).loss.backward()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    opts = build_optimizers(model, lr_muon=0.02, lr_adamw=3e-3, weight_decay=0.1)
    assert opts[1].param_groups[0]["fused"] is True
    for opt in opts:
        opt.step()
    unchanged = [n for n, p in model.named_parameters() if torch.equal(p, before[n])]
    assert not unchanged, unchanged
