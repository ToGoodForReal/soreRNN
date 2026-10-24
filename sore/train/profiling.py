"""Divisão do tempo de GPU de um passo por categoria de kernel (`scripts/bench.py --profile`).

O `torch.profiler` devolve um evento por kernel com o nome do binário. Aqui cada nome cai em uma de
`CATEGORIES` por regras de regex em uma única tabela (`_RULES`), para que as tasks seguintes
estendam o mapa em um lugar só (por exemplo, o kernel de uma CE fundida entra em "loss").

A tabela foi montada com nomes reais capturados na RTX 3050 (sm_86), eager e compilado, com o `tiny`
(FLA + flex, FLA + sdpa) e o `s150m`:

- FLA: `chunk_*`, `recompute_w_u_*`, `prepare_wy_repr_*`, `l2norm_*` (sem prefixo `void`);
- flex: `triton_tem_*flex_attention*` e o `triton_per_*flex_attention_backward*` que o acompanha;
  sdpa: `fmha_cutlass*`;
- cuBLAS (`ampere_*gemm*`, `cublasLt::*`, `gemmSN_*`) e CUTLASS (`cutlass::Kernel2<..gemm..>`);
- Inductor: `triton_poi_*`, `triton_red_*`, `triton_per_*` são pontuais/reduções fundidas, ou seja,
  "elementwise", salvo os do flex (atenção) e os da CE (`log_softmax`/`nll_loss`, "loss");
- ATen: `*elementwise_kernel`, `reduce_kernel`, `multi_tensor_apply` (Muon/AdamW) e `Cat*` também são
  "elementwise"; cópias e `memset` (que o Muon e o checkpoint geram em volume), scan/sort/searchsorted
  da montagem do `cu_seqlens` ficam em "other".

A ordem das regras importa: atenção antes de matmul (o `fmha_cutlass*` contém "cutlass"), loss antes
de elementwise (o `triton_red_*` da CE contém "triton_red_"). Nome sem regra cai em "other".
"""
from __future__ import annotations

import re

import torch

CATEGORIES = ("gdn", "attention", "matmul", "loss", "elementwise", "other")

# (categoria, regex) na ordem de precedência; `re.search` sobre o nome completo do kernel
_RULES: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (cat, re.compile(pattern))
    for cat, pattern in (
        ("gdn", r"^(chunk_|recompute_w_u|prepare_wy_repr|l2norm_|fused_recurrent|fused_gdn|solve_tril|merge_16x16|causal_conv1d)|gated_delta"),
        ("attention", r"flex_attention|fmha_|flash_(fwd|bwd)|pytorch_flash|efficient_attention"),
        ("loss", r"nll_loss|log_softmax|LogSoftMax|SoftMax(Forward|Backward)|softmax_warp|cross_entropy"),
        ("matmul", r"gemm|gemv|cutlass|cublas|xmma|nvjet|^triton_tem_"),
        ("elementwise", r"^triton_(poi|red|per)_|elementwise_kernel|reduce_kernel|multi_tensor_apply|CatArrayBatchedCopy|index_elementwise|gather_kernel|indexSelect|lpnorm|embedding_backward"),
    )
)

_GPU_ACTIVITIES = ("kernel", "gpu_memcpy", "gpu_memset")


def categorize_kernel(name: str) -> str:
    """Categoria (item de `CATEGORIES`) de um kernel pelo nome; sem regra, "other"."""
    for category, pattern in _RULES:
        if pattern.search(name):
            return category
    return "other"


def kernel_intervals(prof: torch.profiler.profile) -> list[tuple[str, float, float]]:
    """`(nome, início_s, fim_s)` de cada kernel/memcpy/memset da GPU no perfil (anotações ficam de fora).

    Os `gpu_user_annotation` (por exemplo `Optimizer.step#Muon.step`) cobrem os kernels de dentro e
    contariam o tempo em dobro.
    """
    return [
        (e.name, e.time_range.start * 1e-6, e.time_range.end * 1e-6)
        for e in prof.events()
        if e.device_type == torch.autograd.DeviceType.CUDA and str(e.activity_type) in _GPU_ACTIVITIES
    ]


def union_seconds(intervals: list[tuple[str, float, float]]) -> float:
    """Tempo coberto pela união dos intervalos: kernels simultâneos em streams diferentes contam uma vez."""
    busy, end = 0.0, float("-inf")
    for _, start, stop in sorted(intervals, key=lambda iv: iv[1]):
        if stop <= end:
            continue
        busy += stop - max(start, end)
        end = stop
    return busy


def kernel_breakdown(prof: torch.profiler.profile) -> dict[str, float]:
    """Segundos de GPU por categoria (soma das durações; todas as `CATEGORIES` presentes, mesmo com 0)."""
    out = dict.fromkeys(CATEGORIES, 0.0)
    for name, start, stop in kernel_intervals(prof):
        out[categorize_kernel(name)] += stop - start
    return out


def idle_seconds(prof: torch.profiler.profile, wall_s: float) -> float:
    """Parede do passo menos a união dos intervalos de kernels: a GPU parada esperando a CPU (>= 0)."""
    return max(wall_s - union_seconds(kernel_intervals(prof)), 0.0)
