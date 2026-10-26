"""Testes da loss de LM em chunks (`chunked_lm_loss`) e fundida do FLA (`fused_lm_loss`)."""
import pytest
import torch
import torch.nn.functional as F

from sore.model.loss import ChunkCrossEntropy, chunked_lm_loss, fused_lm_loss

N, D, V = 50, 16, 37


def _inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(N, D, generator=g).requires_grad_()
    weight = torch.randn(V, D, generator=g).requires_grad_()
    labels = torch.randint(0, V, (N,), generator=g)
    labels[::5] = -100  # 10 ignorados, 40 válidos
    return hidden, weight, labels


def test_chunked_matches_full():
    hidden, weight, labels = _inputs()
    # chunk_size=7: 7 chunks cheios e uma sobra de 1 linha
    loss_sum, n_valid = chunked_lm_loss(hidden, weight, labels, chunk_size=7)
    assert loss_sum.dtype == torch.float32 and loss_sum.dim() == 0
    # a contagem é um tensor int64 0-dim, sem `int(...)`: ler o número na CPU sincronizaria a GPU
    assert isinstance(n_valid, torch.Tensor) and n_valid.dtype == torch.long and n_valid.dim() == 0
    assert n_valid.device == hidden.device and int(n_valid) == 40
    grad_h, grad_w = torch.autograd.grad(loss_sum, (hidden, weight))

    ref = F.cross_entropy(hidden @ weight.T, labels, ignore_index=-100, reduction="sum")
    ref_h, ref_w = torch.autograd.grad(ref, (hidden, weight))
    torch.testing.assert_close(loss_sum, ref, rtol=1e-5, atol=0)
    torch.testing.assert_close(grad_h, ref_h, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(grad_w, ref_w, rtol=1e-5, atol=1e-7)
    assert not grad_h[::5].any()  # linhas ignoradas não recebem gradiente


def test_chunked_uses_given_ce_module():
    # o SoreLM passa o próprio ChunkCrossEntropy (o compile_model o compila in-place)
    class Recording(ChunkCrossEntropy):
        def forward(self, hidden, weight, labels):
            rows.append(hidden.shape[0])
            return super().forward(hidden, weight, labels)

    rows = []
    hidden, weight, labels = _inputs()
    loss_sum, n_valid = chunked_lm_loss(hidden, weight, labels, chunk_size=7, ce=Recording())
    ref, ref_n = chunked_lm_loss(hidden, weight, labels, chunk_size=7)
    assert rows == [7] * 7 + [1] and int(n_valid) == int(ref_n) == 40
    torch.testing.assert_close(loss_sum, ref, rtol=0, atol=0)


def test_chunked_all_ignored_is_zero_and_connected():
    hidden, weight, labels = _inputs()
    loss_sum, n_valid = chunked_lm_loss(hidden, weight, torch.full_like(labels, -100), chunk_size=7)
    assert int(n_valid) == 0
    assert loss_sum.item() == 0.0 and loss_sum.requires_grad
    loss_sum.backward()
    assert hidden.grad is not None and not hidden.grad.any()
    assert weight.grad is not None and not weight.grad.any()


def _saved_shapes(fn):
    """Shapes dos tensores que o autograd guarda (fora de regiões com checkpoint) ao rodar `fn`."""
    shapes = []

    def pack(t):
        shapes.append(tuple(t.shape))
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        fn()
    return shapes


def test_chunked_does_not_save_logits():
    hidden, weight, labels = _inputs()
    # controle: a CE direta guarda tensores [N, V] (logits / log_softmax) para o backward
    full = _saved_shapes(lambda: F.cross_entropy(hidden @ weight.T, labels, reduction="sum"))
    assert (N, V) in full
    # em chunks nada do tamanho de logits fica guardado: o backward recalcula cada chunk
    chunked = _saved_shapes(lambda: chunked_lm_loss(hidden, weight, labels, chunk_size=7))
    assert not [s for s in chunked if len(s) == 2 and s[1] == V], chunked


@pytest.mark.cuda
@pytest.mark.fla
def test_fused_ce_matches_chunked():
    # hidden [1000, 64], weight [512, 64], 13% dos labels -100
    g = torch.Generator(device="cuda").manual_seed(42)
    h_fp32 = torch.randn(1000, 64, device="cuda", generator=g)
    w_fp32 = torch.randn(512, 64, device="cuda", generator=g)
    labels = torch.randint(0, 512, (1000,), device="cuda", generator=g)
    labels[torch.rand(1000, generator=g, device="cuda") < 0.13] = -100

    # 1. Em fp32: soma com rel 1e-4, gradientes de hidden e weight com rel 2e-3
    h_chunk = h_fp32.clone().requires_grad_()
    w_chunk = w_fp32.clone().requires_grad_()
    h_fused = h_fp32.clone().requires_grad_()
    w_fused = w_fp32.clone().requires_grad_()

    loss_c, count_c = chunked_lm_loss(h_chunk, w_chunk, labels)
    loss_f, count_f = fused_lm_loss(h_fused, w_fused, labels)

    assert count_f.dim() == 0 and count_f.dtype == torch.long and count_f.device == labels.device
    assert torch.equal(count_f, count_c)
    assert loss_f.dim() == 0 and loss_f.dtype == torch.float32

    torch.testing.assert_close(loss_f, loss_c, rtol=1e-4, atol=0)

    loss_c.backward()
    loss_f.backward()

    # atol cobre valores muito próximos de zero nos gradientes onde erro relativo não se aplica
    atol_h = 1e-3 * h_chunk.grad.abs().max().item()
    atol_w = 1e-3 * w_chunk.grad.abs().max().item()
    torch.testing.assert_close(h_fused.grad, h_chunk.grad, rtol=2e-3, atol=atol_h)
    torch.testing.assert_close(w_fused.grad, w_chunk.grad, rtol=2e-3, atol=atol_w)

    # 2. Sob autocast bf16: rel 2e-2
    h_chunk = h_fp32.clone().requires_grad_()
    w_chunk = w_fp32.clone().requires_grad_()
    h_fused = h_fp32.clone().requires_grad_()
    w_fused = w_fp32.clone().requires_grad_()

    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss_c_bf16, count_c_bf16 = chunked_lm_loss(h_chunk, w_chunk, labels)
        loss_f_bf16, count_f_bf16 = fused_lm_loss(h_fused, w_fused, labels)

    assert torch.equal(count_f_bf16, count_c_bf16)
    torch.testing.assert_close(loss_f_bf16, loss_c_bf16, rtol=2e-2, atol=0)

    loss_c_bf16.backward()
    loss_f_bf16.backward()

    atol_h_bf16 = 2e-3 * h_chunk.grad.abs().max().item()
    atol_w_bf16 = 2e-3 * w_chunk.grad.abs().max().item()
    torch.testing.assert_close(h_fused.grad, h_chunk.grad, rtol=2e-2, atol=atol_h_bf16)
    torch.testing.assert_close(w_fused.grad, w_chunk.grad, rtol=2e-2, atol=atol_w_bf16)


@pytest.mark.cuda
@pytest.mark.fla
def test_fused_ce_all_ignored():
    # todos os labels -100 -> soma 0, contagem 0, gradientes zerados e finitos (Review Focus 2)
    h = torch.randn(50, 16, device="cuda", requires_grad=True)
    w = torch.randn(32, 16, device="cuda", requires_grad=True)
    labels = torch.full((50,), -100, dtype=torch.long, device="cuda")

    loss_sum, n_valid = fused_lm_loss(h, w, labels)
    assert isinstance(n_valid, torch.Tensor) and n_valid.dtype == torch.long and n_valid.dim() == 0
    assert n_valid.item() == 0
    assert loss_sum.dim() == 0 and loss_sum.dtype == torch.float32
    assert loss_sum.item() == 0.0 and loss_sum.requires_grad

    loss_sum.backward()
    assert h.grad is not None and torch.isfinite(h.grad).all() and not h.grad.any()
    assert w.grad is not None and torch.isfinite(w.grad).all() and not w.grad.any()

