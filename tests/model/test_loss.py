"""Testes da loss de LM em chunks (`chunked_lm_loss`)."""
import torch
import torch.nn.functional as F

from sore.model.loss import ChunkCrossEntropy, chunked_lm_loss

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
    assert isinstance(n_valid, int) and n_valid == 40
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
    assert rows == [7] * 7 + [1] and n_valid == ref_n == 40
    torch.testing.assert_close(loss_sum, ref, rtol=0, atol=0)


def test_chunked_all_ignored_is_zero_and_connected():
    hidden, weight, labels = _inputs()
    loss_sum, n_valid = chunked_lm_loss(hidden, weight, torch.full_like(labels, -100), chunk_size=7)
    assert n_valid == 0
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
