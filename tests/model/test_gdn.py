"""Testes da camada GatedDeltaNet (backends `torch` de referência e `fla`)."""
import pytest
import torch

from sore.model.config import get_preset
from sore.model.gdn import GatedDeltaNet


def _layer(backend="torch", device="cpu", seed=0):
    torch.manual_seed(seed)
    return GatedDeltaNet(get_preset("tiny", gdn_backend=backend)).to(device)


def _cu(values, device="cpu"):
    return torch.tensor(values, dtype=torch.long, device=device)


def test_shapes_finite():
    cfg = get_preset("tiny")
    layer = _layer()
    hqk, hv, dh, k = cfg.gdn_qk_heads, cfg.gdn_v_heads, cfg.gdn_head_dim, cfg.conv_kernel
    assert layer.q_proj.weight.shape == layer.k_proj.weight.shape == (hqk * dh, cfg.d_model)
    assert layer.v_proj.weight.shape == layer.g_proj.weight.shape == (hv * dh, cfg.d_model)
    assert layer.a_proj.weight.shape == layer.b_proj.weight.shape == (hv, cfg.d_model)
    assert layer.conv_q.shape == layer.conv_k.shape == (hqk * dh, k)
    assert layer.conv_v.shape == (hv * dh, k)
    assert layer.A_log.shape == layer.dt_bias.shape == (hv,)
    assert layer.o_proj.weight.shape == (cfg.d_model, hv * dh)
    assert all(m.bias is None for m in layer.modules() if isinstance(m, torch.nn.Linear))

    x = torch.randn(2, 10, cfg.d_model, requires_grad=True)
    out = layer(x)
    assert out.shape == x.shape and out.dtype == x.dtype
    assert torch.isfinite(out).all()

    out.square().mean().backward()
    for name, p in layer.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


def test_packed_equals_separate():
    layer = _layer()
    x = torch.randn(1, 20, 128)
    packed = layer(x, cu_seqlens=_cu([0, 7, 20]))
    separate = torch.cat([layer(x[:, :7]), layer(x[:, 7:])], dim=1)
    torch.testing.assert_close(packed, separate, atol=1e-5, rtol=0)
    # sem cu_seqlens o segundo documento enxerga o estado e a conv do primeiro
    assert not torch.allclose(layer(x)[:, 7:], separate[:, 7:], atol=1e-3)


def test_batch_rows_are_independent_documents():
    # Sem cu_seqlens e com B > 1, cada linha do batch é um documento.
    layer = _layer()
    x = torch.randn(3, 9, 128)
    batched = layer(x)
    rows = torch.cat([layer(x[i : i + 1]) for i in range(3)], dim=0)
    torch.testing.assert_close(batched, rows, atol=1e-5, rtol=0)


@pytest.mark.parametrize(
    "backend",
    ["torch", pytest.param("fla", marks=pytest.mark.fla)],
)
def test_extreme_inputs_finite(backend):
    device = "cuda" if backend == "fla" else "cpu"
    layer = _layer(backend, device)
    x = (torch.randn(1, 64, 128, device=device) * 100).requires_grad_(True)
    out = layer(x, cu_seqlens=_cu([0, 30, 64], device))
    assert torch.isfinite(out).all()
    out.square().mean().backward()
    assert torch.isfinite(x.grad).all()


@pytest.mark.fla
def test_fla_matches_torch():
    # Mesmos pesos nos dois backends; fp32 com documentos de 1 token e um maior que o chunk (64) do FLA.
    fla_layer = _layer("fla", "cuda")
    ref_layer = _layer("torch", "cuda", seed=1)
    ref_layer.load_state_dict(fla_layer.state_dict())

    gen = torch.Generator(device="cuda").manual_seed(2)
    x = torch.randn(1, 257, 128, device="cuda", generator=gen)
    cotangent = torch.randn(1, 257, 128, device="cuda", generator=gen)
    cu = _cu([0, 1, 2, 100, 257], "cuda")

    results = {}
    for name, layer in (("fla", fla_layer), ("torch", ref_layer)):
        xi = x.clone().requires_grad_(True)
        out = layer(xi, cu_seqlens=cu)
        (out * cotangent).sum().backward()
        results[name] = (out.detach(), xi.grad, layer.q_proj.weight.grad)

    (out_fla, gx_fla, gq_fla), (out_ref, gx_ref, gq_ref) = results["fla"], results["torch"]
    torch.testing.assert_close(out_fla, out_ref, atol=1e-3, rtol=0)
    # Em fp32 os tl.dot do FLA usam TF32: erro máximo medido ~1.2e-3 * max|ref| nos gradientes
    # (com TRITON_F32_DEFAULT=ieee cai para ~3e-5 * max|ref|, ou seja, não é erro de convenção).
    # atol = 5e-3 * max|ref| cobre os elementos perto de zero, onde rtol não tem sentido;
    # um erro de convenção (scale, g, beta, reset) dá erro relativo O(1).
    for actual, expected in ((gx_fla, gx_ref), (gq_fla, gq_ref)):
        atol = 5e-3 * expected.abs().max().item()
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=atol)


# --- Task 5: caminho do GDN (GVA nativo, gate no kernel, conv do FLA, norma com gate fundida) ---

_OPTIONS = ("gdn_kernel_gate", "gdn_fla_conv", "fused_norm_gate")
_OPTION_SETS = {
    "none": (),  # só o GVA nativo, que no backend fla vale sempre (D4)
    "kernel_gate": ("gdn_kernel_gate",),
    "fla_conv": ("gdn_fla_conv",),
    "fused_norm_gate": ("fused_norm_gate",),
    "all": _OPTIONS,
}
# documentos de 1-3 tokens (menores que K=4 e que o chunk 64) e um documento na janela inteira
_CU_CASES = {"short_docs": [0, 1, 3, 4, 70, 200, 512], "one_doc": [0, 512]}


def _fla_twin_layers(options, rep, seed=0):
    """Camada `fla` com as `options` ligadas e o gêmeo `torch` (referência) com os mesmos pesos."""
    kw = dict(gdn_qk_heads=2, gdn_v_heads=2 * rep, **{o: True for o in options})
    torch.manual_seed(seed)
    fla_layer = GatedDeltaNet(get_preset("tiny", gdn_backend="fla", **kw)).cuda()
    ref_layer = GatedDeltaNet(get_preset("tiny", gdn_backend="torch", gdn_qk_heads=2, gdn_v_heads=2 * rep)).cuda()
    ref_layer.load_state_dict(fla_layer.state_dict())
    return fla_layer, ref_layer


@pytest.mark.cuda
@pytest.mark.fla
@pytest.mark.parametrize("cu_case", sorted(_CU_CASES))
@pytest.mark.parametrize("rep", [1, 2], ids=["no_gva", "gva2x"])
@pytest.mark.parametrize("option_set", sorted(_OPTION_SETS))
def test_fla_options_match_torch_backend(option_set, rep, cu_case):
    fla_layer, ref_layer = _fla_twin_layers(_OPTION_SETS[option_set], rep)
    cu = _cu(_CU_CASES[cu_case], "cuda")
    cu_cpu = cu.cpu()

    gen = torch.Generator(device="cuda").manual_seed(2)
    x = torch.randn(1, 512, 128, device="cuda", generator=gen)
    cotangent = torch.randn(1, 512, 128, device="cuda", generator=gen)

    results = {}
    for name, layer in (("fla", fla_layer), ("torch", ref_layer)):
        xi = x.clone().requires_grad_(True)
        out = layer(xi, cu_seqlens=cu, cu_seqlens_cpu=cu_cpu if name == "fla" else None)
        (out * cotangent).sum().backward()
        grads = {n: p.grad for n, p in layer.named_parameters()}
        results[name] = (out.detach(), xi.grad, grads)

    (out_fla, gx_fla, g_fla), (out_ref, gx_ref, g_ref) = results["fla"], results["torch"]
    # mesmas tolerâncias de `test_fla_matches_torch` (os tl.dot do FLA em fp32 usam TF32; um erro de
    # convenção dá erro relativo O(1)) para a saída, o grad de x e o de cada parâmetro, menos A_log e
    # dt_bias. Esses dois são escalares por head somados sobre os 512 tokens e passam pela cadeia do
    # decaimento (cumsum de g): o ruído do TF32 chega a 3e-2 * max|ref| nos 20 casos medidos (também na
    # opção "none", que usa o gate antigo), e a 6e-5 * max|ref| com TRITON_F32_DEFAULT=ieee em todos os
    # casos (as 15 grandezas comparadas ficam em <= 2e-4). Não é erro de convenção; para esses dois o
    # atol é 5e-2 * max|ref|.
    torch.testing.assert_close(out_fla, out_ref, atol=1e-3, rtol=0)
    pairs = [("x", gx_fla, gx_ref)] + [(n, g_fla[n], g_ref[n]) for n in g_ref]
    for name, actual, expected in pairs:
        atol = (5e-2 if name in ("A_log", "dt_bias") else 5e-3) * expected.abs().max().item()
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=atol, msg=lambda m, n=name: f"{n}: {m}")


@pytest.mark.cuda
@pytest.mark.fla
@pytest.mark.parametrize("option_set", ["none", "all"])
def test_fla_options_isolate_documents(option_set):
    # mudar os tokens do documento 4 ([70, 200)) não pode alterar a saída dos documentos 0-3 ([0, 70)):
    # a conv e o estado resetam em cada documento, inclusive nos de 1-3 tokens (< K=4 e < chunk 64)
    layer, _ = _fla_twin_layers(_OPTION_SETS[option_set], rep=2)
    cu = _cu(_CU_CASES["short_docs"], "cuda")
    gen = torch.Generator(device="cuda").manual_seed(3)
    x = torch.randn(1, 512, 128, device="cuda", generator=gen)
    x2 = x.clone()
    x2[:, 70:200] = torch.randn(1, 130, 128, device="cuda", generator=gen)

    with torch.no_grad():
        out = layer(x, cu_seqlens=cu, cu_seqlens_cpu=cu.cpu())
        out2 = layer(x2, cu_seqlens=cu, cu_seqlens_cpu=cu.cpu())
    torch.testing.assert_close(out[:, :70], out2[:, :70], atol=1e-6, rtol=0)
    assert not torch.allclose(out[:, 70:200], out2[:, 70:200], atol=1e-3)  # o teste enxerga a mudança
