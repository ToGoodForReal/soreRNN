"""Testes dos residuais (plain e gated hyper-connection), do stream residual e do Block."""
import pytest
import torch

from sore.model.block import Block
from sore.model.config import get_preset
from sore.model.residual import (
    GatedHyperResidual,
    PlainResidual,
    StreamFinal,
    init_stream,
    make_residual,
)


def _cu(values):
    return torch.tensor(values, dtype=torch.long)


@pytest.mark.parametrize("mixer_type", ["gdn", "attn"])
def test_plain_block_shapes(mixer_type):
    torch.manual_seed(0)
    cfg = get_preset("tiny")
    block = Block(cfg, layer_idx=0, mixer_type=mixer_type)
    assert isinstance(block.res_mixer, PlainResidual) and isinstance(block.res_ffn, PlainResidual)
    assert list(block.res_mixer.parameters()) == list(block.res_ffn.parameters()) == []

    x = torch.randn(2, 10, cfg.d_model, requires_grad=True)
    H = init_stream(x, cfg)
    assert H is x  # plain: o stream é o próprio x
    out = block(H)
    assert out.shape == x.shape and out.dtype == x.dtype
    assert torch.isfinite(out).all()
    assert torch.equal(StreamFinal(cfg)(out), out)  # plain: StreamFinal é identidade

    # pré-norma padrão: h = x + mixer(norm1(x)); out = h + ffn(norm2(h))
    h = x + block.mixer(block.norm1(x))
    torch.testing.assert_close(out, h + block.ffn(block.norm2(h)), atol=1e-6, rtol=0)

    out.square().mean().backward()
    for name, p in block.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name

    # layout empacotado [1, N] com cu_seqlens
    xp = torch.randn(1, 30, cfg.d_model)
    assert block(xp, _cu([0, 9, 30])).shape == xp.shape


def test_hyper_reduces_to_mean_at_init():
    torch.manual_seed(0)
    cfg = get_preset("tiny", residual="hyper")
    n, d = cfg.hc_branches, cfg.d_model
    res = make_residual(cfg, layer_idx=0, sub=0)
    assert isinstance(res, GatedHyperResidual)

    x = torch.randn(2, 7, d)
    H = init_stream(x, cfg)
    assert H.shape == (2, 7, n, d) and H.is_contiguous()
    # ramos idênticos: read devolve o próprio ramo
    torch.testing.assert_close(res.read(H), x, atol=1e-6, rtol=0)
    # no init o gate de leitura vale 1/n em tudo, então read é a média dos ramos para qualquer H
    Hr = torch.randn(2, 7, n, d)
    torch.testing.assert_close(res.read(Hr), Hr.mean(2), atol=1e-6, rtol=0)

    # StreamFinal começa uniforme: média dos ramos (e devolve x a partir do stream inicial)
    final = StreamFinal(cfg)
    torch.testing.assert_close(final(H), x, atol=1e-6, rtol=0)
    torch.testing.assert_close(final(Hr), Hr.mean(2), atol=1e-6, rtol=0)


def test_hyper_matches_formula():
    # fora do init (pesos aleatórios), read/write seguem a fórmula do plano ao pé da letra
    torch.manual_seed(0)
    n, d, rank, eps = 4, 32, 8, 1e-6
    res = GatedHyperResidual(d, n, rank, layer_idx=2, eps=eps)
    assert {name for name, _ in res.named_parameters()} == {
        "down.weight", "up.weight", "up.bias", "write_proj.weight", "write_bias"
    }  # rms(·) não tem peso
    with torch.no_grad():
        for p in res.parameters():
            p.normal_(0, 0.5)
    H = torch.randn(2, 5, n, d)
    y = torch.randn(2, 5, d)

    m = H.mean(2)
    z = m / torch.sqrt(m.pow(2).mean(-1, keepdim=True) + eps)
    r = torch.sigmoid(res.up(torch.nn.functional.silu(z @ res.down.weight.T))).view(2, 5, n, d)
    torch.testing.assert_close(res.read(H), (r * H).sum(2))
    w = 2 * torch.sigmoid(res.write_bias + z @ res.write_proj.weight.T)  # [B,T,n]
    torch.testing.assert_close(res.write(H, y), H + w[..., None] * y[:, :, None])


def _write_proj_grads(block, cfg, x, target):
    block.zero_grad()
    out = StreamFinal(cfg)(block(init_stream(x, cfg)))
    (out * target).sum().backward()
    return [block.res_mixer.write_proj.weight.grad, block.res_ffn.write_proj.weight.grad]


@pytest.mark.parametrize("layer_idx", [0, 5])
def test_hyper_symmetry_breaks(layer_idx):
    torch.manual_seed(0)
    cfg = get_preset("tiny", residual="hyper")
    n, d = cfg.hc_branches, cfg.d_model
    hot = layer_idx % n
    block = Block(cfg, layer_idx=layer_idx, mixer_type="gdn")

    # gate de escrita no init: w = 1 no ramo layer_idx % n e 0.1 nos outros (nos dois sub-blocos)
    expected = torch.full((n,), 0.1)
    expected[hot] = 1.0
    for res in (block.res_mixer, block.res_ffn):
        torch.testing.assert_close(2 * torch.sigmoid(res.write_bias), expected)
        torch.testing.assert_close(
            res.write(torch.zeros(1, 3, n, d), torch.ones(1, 3, d)),
            expected[None, None, :, None].expand(1, 3, n, d),
        )

    x = torch.randn(2, 12, d)
    target = torch.randn(2, 12, d)
    for g in _write_proj_grads(block, cfg, x, target):
        assert g.shape == (n, d)
        assert g.abs().max() > 1e-3  # gradiente de verdade, não zero
        for i in range(n):
            if i != hot:
                assert not torch.allclose(g[hot], g[i], rtol=1e-2, atol=0), (hot, i)

    # controle: com o bias de escrita uniforme os ramos são intercambiáveis e os gradientes
    # das linhas de W_w ficam iguais, ou seja, a assimetria vem do init de write_bias
    with torch.no_grad():
        block.res_mixer.write_bias.zero_()
        block.res_ffn.write_bias.zero_()
    for g in _write_proj_grads(block, cfg, x, target):
        for i in range(1, n):
            torch.testing.assert_close(g[i], g[0])


@pytest.mark.parametrize("mixer_type", ["gdn", "attn"])
def test_hyper_packed_equals_separate(mixer_type):
    torch.manual_seed(0)
    cfg = get_preset("tiny", residual="hyper")
    block = Block(cfg, layer_idx=1, mixer_type=mixer_type)
    # Sai do init (leitura uniforme, W_w = 0) para que o caminho dependente de dados (z) pese
    # na saída: um vazamento no tempo dentro do residual apareceria aqui.
    with torch.no_grad():
        for res in (block.res_mixer, block.res_ffn):
            res.up.weight.normal_(0, 0.1)
            res.write_proj.weight.normal_(0, 0.1)

    H = torch.randn(1, 30, cfg.hc_branches, cfg.d_model)  # ramos distintos
    packed = block(H, _cu([0, 9, 30]))
    separate = torch.cat([block(H[:, :9]), block(H[:, 9:])], dim=1)
    assert packed.shape == H.shape
    torch.testing.assert_close(packed, separate, atol=1e-5, rtol=0)
    # sem cu_seqlens o segundo documento enxerga o primeiro
    assert not torch.allclose(block(H)[:, 9:], separate[:, 9:], atol=1e-3)


def test_rejects_unknown_options():
    cfg = get_preset("tiny")
    with pytest.raises(ValueError):
        Block(cfg, layer_idx=0, mixer_type="mamba")
    bad = get_preset("tiny", residual="nope")
    with pytest.raises(ValueError):
        make_residual(bad, layer_idx=0, sub=0)
    with pytest.raises(ValueError):
        init_stream(torch.randn(1, 3, cfg.d_model), bad)
    with pytest.raises(ValueError):
        StreamFinal(bad)
