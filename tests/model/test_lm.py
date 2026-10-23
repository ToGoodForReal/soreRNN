"""Testes do SoreLM: init, empacotamento sem vazamento (tronco e MTP), backward e contagem de params."""
import math

import pytest
import torch

import sore.model.lm as lm_module
from sore.model.config import get_preset
from sore.model.lm import LMOutput, SoreLM
from sore.model.mtp import mtp_labels, next_token_embeddings
from sore.model.residual import GatedHyperResidual

CU = [0, 9, 30]
DOCS = list(zip(CU[:-1], CU[1:]))


def _cu(values):
    return torch.tensor(values, dtype=torch.long)


def _tokens(n, seed, vocab=512):
    return torch.randint(0, vocab, (1, n), generator=torch.Generator().manual_seed(seed))


def _model(seed=0, **overrides):
    torch.manual_seed(seed)
    return SoreLM(get_preset("tiny", **overrides))


def _perturb_hyper(model):
    # Sai do init do hyper-residual (leitura uniforme, W_w = 0, mistura final uniforme) para que o
    # caminho dependente de dados pese na saída: um vazamento no tempo apareceria nos testes.
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, GatedHyperResidual):
                m.up.weight.normal_(0, 0.1)
                m.write_proj.weight.normal_(0, 0.1)
        model.stream_final.mix_logits.normal_(0, 1)


def _labels(n=30):
    labels = _tokens(n, seed=2)
    labels[0, [3, 15, 29]] = -100  # alguns ignorados; labels[9] (1º do doc 2) continua válido
    return labels


def test_init_loss_near_log_vocab():
    model = _model()
    out = model(_tokens(64, seed=1), labels=_tokens(64, seed=2))
    assert isinstance(out, LMOutput)
    assert out.n_tokens == 64
    assert abs(out.lm_loss.item() - math.log(512)) < 0.5
    assert abs(out.mtp_loss.item() - math.log(512)) < 0.5
    torch.testing.assert_close(out.loss, out.lm_loss + model.cfg.mtp_weight * out.mtp_loss)


def test_init_weights():
    model = _model(residual="hyper")
    cfg = model.cfg
    std = cfg.init_std
    scaled = std / math.sqrt(2 * cfg.n_layers)
    gdn, attn = model.blocks[0], model.blocks[3]
    assert gdn.mixer_type == "gdn" and attn.mixer_type == "attn"

    def assert_std(p, expected):
        assert abs(p.std().item() - expected) < 0.05 * expected, (p.std().item(), expected)
        assert abs(p.mean().item()) < 0.05 * expected

    assert_std(model.embed.weight, std)
    assert_std(gdn.mixer.q_proj.weight, std)
    assert_std(gdn.ffn.up_proj.weight, std)
    assert_std(attn.mixer.gate_proj.weight, std)
    assert_std(model.mtp.proj.weight, std)
    for o in (gdn.mixer.o_proj, attn.mixer.o_proj, gdn.ffn.down_proj, model.mtp.block.mixer.o_proj,
              model.mtp.block.ffn.down_proj):
        assert_std(o.weight, scaled)

    # inits próprios preservados: conv da GDN U(±1/sqrt(K)), normas em 1, hyper-residual do T6
    bound = cfg.conv_kernel**-0.5
    assert gdn.mixer.conv_q.abs().max() <= bound and gdn.mixer.conv_q.std() > 0.5 * bound
    assert torch.equal(gdn.norm1.weight, torch.ones_like(gdn.norm1.weight))
    for m in model.modules():
        if isinstance(m, GatedHyperResidual):
            assert not m.up.weight.any() and not m.write_proj.weight.any()
            torch.testing.assert_close(m.up.bias, torch.full_like(m.up.bias, math.log(1 / 3)))


@pytest.mark.parametrize("residual", ["plain", "hyper"])
def test_packed_logits_equal_separate(residual):
    # Review Focus 1: conv, GDN, atenção, residual e MTP não vazam entre documentos empacotados
    model = _model(residual=residual)
    if residual == "hyper":
        _perturb_hyper(model)
    ids, cu = _tokens(30, seed=1), _cu(CU)

    with torch.no_grad():
        packed = model.logits(ids, cu)
        separate = torch.cat([model.logits(ids[:, a:b]) for a, b in DOCS], dim=1)
        leaky = model.logits(ids)  # sem cu_seqlens: o doc 2 enxerga o doc 1
    assert packed.shape == (1, 30, model.cfg.vocab_size)
    torch.testing.assert_close(packed, separate, atol=1e-5, rtol=0)
    assert not torch.allclose(leaky[:, 9:], separate[:, 9:], atol=1e-3)

    # MTP: hidden do módulo MTP empacotado == documentos sozinhos
    with torch.no_grad():
        emb = model.embed(ids)
        h = model.trunk(emb, cu)
        mtp_packed = model.mtp(h, next_token_embeddings(emb, cu), cu)
        mtp_separate = torch.cat(
            [model.mtp(model.trunk(emb[:, a:b]), next_token_embeddings(emb[:, a:b])) for a, b in DOCS], dim=1
        )
        # controles: emb_next sem a fronteira (última posição do doc 1 vê o 1º token do doc 2)
        # e Block do MTP sem cu_seqlens (o doc 2 atende ao doc 1)
        leaky_emb = model.mtp(h, next_token_embeddings(emb, None), cu)
        leaky_block = model.mtp(h, next_token_embeddings(emb, cu), None)
    torch.testing.assert_close(mtp_packed, mtp_separate, atol=1e-5, rtol=0)
    assert not torch.allclose(leaky_emb[:, 8], mtp_separate[:, 8], atol=1e-3)
    assert not torch.allclose(leaky_block[:, 9:], mtp_separate[:, 9:], atol=1e-3)


def test_packed_loss_equal_separate():
    # forward empacotado == soma ponderada dos documentos sozinhos, para a loss do LM e a do MTP
    model = _model()
    ids, cu, labels = _tokens(30, seed=1), _cu(CU), _labels()
    with torch.no_grad():
        out = model(ids, cu, labels=labels)
        parts = [model(ids[:, a:b], labels=labels[:, a:b]) for a, b in DOCS]

    assert out.n_tokens == sum(p.n_tokens for p in parts) == 27
    lm = sum(p.lm_loss * p.n_tokens for p in parts) / out.n_tokens
    torch.testing.assert_close(out.lm_loss, lm, atol=1e-5, rtol=0)

    def n_mtp(lab, cu_seqlens=None):
        return int((mtp_labels(lab, cu_seqlens) != -100).sum())

    counts = [n_mtp(labels[:, a:b]) for a, b in DOCS]
    assert n_mtp(labels, cu) == sum(counts)
    mtp = sum(p.mtp_loss * c for p, c in zip(parts, counts)) / sum(counts)
    torch.testing.assert_close(out.mtp_loss, mtp, atol=1e-5, rtol=0)
    torch.testing.assert_close(out.loss, out.lm_loss + model.cfg.mtp_weight * out.mtp_loss)


def test_mtp_labels_mask_boundaries():
    labels = torch.arange(10, 20).view(1, 10)
    labels[0, 6] = -100  # um alvo ignorado dentro do doc 3 vira out[5] = -100
    cu = _cu([0, 3, 4, 10])  # docs [0,3), [3,4) (1 token), [4,10)
    assert mtp_labels(labels, cu).tolist() == [[11, 12, -100, -100, 15, -100, 17, 18, 19, -100]]
    # sem cu_seqlens só a última posição da linha fica sem alvo (além do ignorado)
    assert mtp_labels(labels, None).tolist() == [[11, 12, 13, 14, 15, -100, 17, 18, 19, -100]]
    # batch sem cu_seqlens: cada linha é um documento
    rows = torch.arange(6).view(2, 3)
    assert mtp_labels(rows, None).tolist() == [[1, 2, -100], [4, 5, -100]]

    # emb_next: embedding do próximo token dentro do documento, zero na última posição de cada um
    emb = torch.randn(1, 10, 3)
    nxt = next_token_embeddings(emb, cu)
    for t in range(10):
        if t in (2, 3, 9):
            assert not nxt[0, t].any(), t
        else:
            assert torch.equal(nxt[0, t], emb[0, t + 1]), t


@pytest.mark.parametrize(
    "overrides", [{}, {"residual": "hyper"}, {"tie_embeddings": False}], ids=["plain", "hyper", "untied"]
)
def test_backward_all_params_get_grad(overrides):
    model = _model(**overrides)
    out = model(_tokens(30, seed=1), _cu(CU), labels=_labels())
    assert out.mtp_loss is not None
    out.loss.backward()
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    assert not missing, missing
    for name, p in model.named_parameters():
        assert torch.isfinite(p.grad).all(), name


def test_grad_checkpoint_matches(monkeypatch):
    calls = []
    real_checkpoint = lm_module.checkpoint

    def counting_checkpoint(fn, *args, **kwargs):
        calls.append(kwargs.get("use_reentrant"))
        return real_checkpoint(fn, *args, **kwargs)

    monkeypatch.setattr(lm_module, "checkpoint", counting_checkpoint)
    ids, cu, labels = _tokens(30, seed=1), _cu(CU), _labels()
    results = []
    for ckpt in (False, True):
        calls.clear()
        model = _model(residual="hyper", grad_checkpoint=ckpt)  # mesma seed, mesmos pesos
        _perturb_hyper(model)
        out = model(ids, cu, labels=labels)
        out.loss.backward()
        results.append((out.loss.detach(), {n: p.grad for n, p in model.named_parameters()}))
        # um checkpoint por Block do tronco, sempre não-reentrante
        assert calls == ([False] * model.cfg.n_layers if ckpt else [])

    (loss_ref, grads_ref), (loss_ckpt, grads_ckpt) = results
    torch.testing.assert_close(loss_ckpt, loss_ref, atol=1e-6, rtol=0)
    for name, g in grads_ref.items():
        torch.testing.assert_close(grads_ckpt[name], g, atol=1e-6, rtol=1e-5, msg=name)

    # sem grad (avaliação) o checkpoint é pulado
    calls.clear()
    with torch.no_grad():
        _model(grad_checkpoint=True)(ids, cu, labels=labels)
    assert calls == []


@pytest.mark.cuda
@pytest.mark.fla
def test_flex_block_mask_built_once_per_forward(monkeypatch):
    # R24: um BlockMask por forward, compartilhado pelas camadas de atenção do tronco e pelo MTP,
    # e o recompute do grad checkpoint não o refaz
    import torch.nn.attention.flex_attention as flex_module

    calls = []
    real = flex_module.create_block_mask

    def counting(*args, **kwargs):
        calls.append(kwargs["Q_LEN"])
        return real(*args, **kwargs)

    monkeypatch.setattr(flex_module, "create_block_mask", counting)
    model = _model(n_layers=8, gdn_backend="fla", attn_backend="flex", grad_checkpoint=True).cuda()
    ids, cu = _tokens(300, seed=1).cuda(), _cu([0, 120, 300]).cuda()
    out = model(ids, cu, labels=_tokens(300, seed=2).cuda())
    out.loss.backward()
    assert calls == [300]  # 2 camadas de atenção + MTP, forward e recompute: um só BlockMask

    # o BlockMask compartilhado dá o mesmo resultado que a máscara densa do sdpa
    ref = _model(n_layers=8, gdn_backend="fla", attn_backend="sdpa").cuda()
    ref.load_state_dict(model.state_dict())
    with torch.no_grad():
        torch.testing.assert_close(model.logits(ids, cu), ref.logits(ids, cu), atol=2e-3, rtol=0)


@pytest.mark.cuda
@pytest.mark.fla
def test_packed_isolation_bf16_fla_flex():
    # ponta a ponta no caminho de treino (FLA + flex, autocast bf16): documentos empacotados ==
    # documentos sozinhos. Os de 1 e 69 tokens caem no sdpa (T < 128); o empacotado usa o flex.
    model = _model(gdn_backend="fla", attn_backend="flex").cuda()
    cu_list = [0, 1, 70, 200, 333]
    ids, cu = _tokens(333, seed=1).cuda(), _cu(cu_list).cuda()

    def logits(x, cu_seqlens=None):
        # tronco em bf16 e projeção em fp32: compara o tronco, não o arredondamento bf16 dos logits
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model.trunk(model.embed(x), cu_seqlens)
        return torch.nn.functional.linear(h.float(), model.head_weight)

    with torch.no_grad():
        packed = logits(ids, cu)
        separate = torch.cat([logits(ids[:, a:b]) for a, b in zip(cu_list[:-1], cu_list[1:])], dim=1)
        leaky = logits(ids)  # controle sem cu_seqlens: os documentos seguintes enxergam os anteriores
    torch.testing.assert_close(packed, separate, atol=2e-2, rtol=0)
    assert (leaky[:, 70:] - separate[:, 70:]).abs().max() > 0.2

    labels = _tokens(333, seed=2).cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(ids, cu, labels=labels)
    out.loss.backward()
    bad = [n for n, p in model.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    assert not bad, bad


def test_mtp_stream_is_fp32_under_autocast():
    # o stream residual do MTP é fp32 como o do tronco: a saída bf16 do `proj` (autocast) é convertida
    model = _model()
    seen = []
    model.mtp.block.register_forward_pre_hook(lambda mod, args: seen.append(args[0].dtype))
    model.blocks[0].register_forward_pre_hook(lambda mod, args: seen.append(args[0].dtype))
    ids, cu = _tokens(30, seed=1), _cu(CU)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        model(ids, cu, labels=_labels())
    assert seen == [torch.float32, torch.float32]  # bloco 0 do tronco, depois o Block do MTP


def test_loss_chunk_size_from_config(monkeypatch):
    calls = []
    real = lm_module.chunked_lm_loss

    def recording(hidden, weight, labels, chunk_size=8192, **kwargs):
        calls.append(chunk_size)
        return real(hidden, weight, labels, chunk_size=chunk_size, **kwargs)

    ids, cu, labels = _tokens(30, seed=1), _cu(CU), _labels()
    with torch.no_grad():
        ref = _model()(ids, cu, labels=labels)
        monkeypatch.setattr(lm_module, "chunked_lm_loss", recording)
        out = _model(loss_chunk_size=7)(ids, cu, labels=labels)
    assert calls == [7, 7]  # loss do LM e do MTP
    torch.testing.assert_close(out.lm_loss, ref.lm_loss, atol=1e-6, rtol=0)
    torch.testing.assert_close(out.mtp_loss, ref.mtp_loss, atol=1e-6, rtol=0)


def test_forward_variants():
    model = _model()
    ids, cu, labels = _tokens(30, seed=1), _cu(CU), _labels()
    with torch.no_grad():
        ref = model(ids, cu, labels=labels)
        # inputs_embeds sem input_ids: o tronco é o mesmo, mas o MTP é pulado
        emb_only = model(inputs_embeds=model.embed(ids), cu_seqlens=cu, labels=labels)
        no_labels = model(ids, cu)
    assert emb_only.mtp_loss is None and emb_only.n_tokens == ref.n_tokens
    torch.testing.assert_close(emb_only.lm_loss, ref.lm_loss)
    torch.testing.assert_close(emb_only.loss, emb_only.lm_loss)
    assert no_labels == LMOutput(loss=None, lm_loss=None, mtp_loss=None, n_tokens=0)
    with pytest.raises(ValueError):
        model(cu_seqlens=cu, labels=labels)

    # sem MTP: nenhum módulo mtp e loss == lm_loss
    no_mtp = _model(mtp_layers=0)
    assert no_mtp.mtp is None
    with torch.no_grad():
        out = no_mtp(ids, cu, labels=labels)
    assert out.mtp_loss is None
    torch.testing.assert_close(out.loss, out.lm_loss)
    with pytest.raises(NotImplementedError):
        SoreLM(get_preset("tiny", mtp_layers=2))


def test_param_counts():
    with torch.device("meta"):
        s150m = SoreLM(get_preset("s150m"))
        b0_8 = SoreLM(get_preset("b0_8"))
        b0_8_untied = SoreLM(get_preset("b0_8", tie_embeddings=False))
    assert 0.13e9 <= s150m.num_params() <= 0.17e9, s150m.num_params()
    assert 0.75e9 <= b0_8.num_params() <= 0.85e9, b0_8.num_params()

    # embedding amarrado conta uma vez; exclude_embedding tira V*D (e a cabeça, se separada)
    vd = b0_8.cfg.vocab_size * b0_8.cfg.d_model
    assert b0_8.lm_head is None and b0_8.head_weight is b0_8.embed.weight
    assert b0_8.num_params() - b0_8.num_params(exclude_embedding=True) == vd
    assert b0_8_untied.num_params() == b0_8.num_params() + vd
    assert b0_8_untied.num_params(exclude_embedding=True) == b0_8.num_params(exclude_embedding=True)
