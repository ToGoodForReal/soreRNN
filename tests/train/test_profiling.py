import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sore.train.profiling import (
    CATEGORIES,
    categorize_kernel,
    idle_seconds,
    kernel_breakdown,
    kernel_intervals,
    union_seconds,
)

# Nomes reais de kernels, coletados com torch.profiler na RTX 3050 (sm_86), eager e compilado, no
# `tiny` (FLA + flex, FLA + sdpa) e no `s150m`. Os nomes longos estão truncados em 130 caracteres:
# as regras só olham o começo do nome (o resto são parâmetros de template).
REAL_KERNELS = [
    ('chunk_bwd_kernel_dqkwg', 'gdn'),
    ('chunk_gated_delta_rule_fwd_kernel_h_blockdim64', 'gdn'),
    ('recompute_w_u_fwd_kernel', 'gdn'),
    ('chunk_fwd_kernel_o', 'gdn'),
    ('prepare_wy_repr_bwd_kernel', 'gdn'),
    ('l2norm_fwd_kernel', 'gdn'),
    ('l2norm_bwd_kernel', 'gdn'),
    ('chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64', 'gdn'),
    ('chunk_gated_delta_rule_fwd_kkt_solve_kernel', 'gdn'),
    ('chunk_local_cumsum_scalar_kernel', 'gdn'),
    ('chunk_bwd_kernel_dv_local', 'gdn'),
    ('triton_tem_fused_flex_attention_0', 'attention'),
    ('triton_tem_fused_flex_attention_backward_1', 'attention'),
    ('triton_per_fused_flex_attention_backward_0', 'attention'),
    ('triton_tem_fused__unsafe_view_flex_attention_transpose_view_10', 'attention'),
    ('triton_tem_fused__unsafe_view_flex_attention_backward_mul_sigmoid_transpose_view_5', 'attention'),
    ('triton_per_fused__unsafe_view_flex_attention_backward_mul_sigmoid_sigmoid_backward_transpose_view_4', 'attention'),
    ('fmha_cutlassF_bf16_aligned_64x64_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 6', 'attention'),
    ('fmha_cutlassB_bf16_aligned_64x64_k64_seqaligned_sm80(PyTorchMemEffAttention::AttentionBackwardKernel<cutlass::arch::Sm80, cutlass:', 'attention'),
    ('ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_tn', 'matmul'),
    ('ampere_bf16_s16816gemm_bf16_256x128_ldg8_f2f_stages_32x3_nt', 'matmul'),
    ('void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8>(cutlass_80_tensorop_bf16_s16816gemm_re', 'matmul'),
    ('void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_64x1_nn_align8>(cutlass_80_wmma_tensorop_bf16_s161616ge', 'matmul'),
    ('void cublasLt::splitKreduce_kernel<32, 16, int, __nv_bfloat16, __nv_bfloat16, float, __nv_bfloat16, false, __nv_bfloat16, __nv_bfl', 'matmul'),
    ('void gemmSN_TN_kernel<float, 128, 16, 2, 4, 2, 2, true, cublasGemvTensorStridedBatched<__nv_bfloat16 const>, cublasGemvTensorStrid', 'matmul'),
    ('triton_red_fused__log_softmax__to_copy_prepare_softmax_online_2', 'loss'),
    ('triton_red_fused__log_softmax__to_copy_nll_loss_forward_3', 'loss'),
    ('triton_red_fused__log_softmax__log_softmax_backward_data__to_copy_arange_eq_expand_nll_loss_backward_scalar_tensor_view_0', 'loss'),
    ('void at::native::(anonymous namespace)::cunn_SoftMaxForward<4, float, float, float, at::native::(anonymous namespace)::LogSoftMaxF', 'loss'),
    ('void at::native::(anonymous namespace)::cunn_SoftMaxBackward<4, float, float, float, at::native::(anonymous namespace)::LogSoftMax', 'loss'),
    ('void at::native::(anonymous namespace)::nll_loss_forward_reduce_cuda_kernel_2d<float, float, long>(float*, float*, float const*, l', 'loss'),
    ('void at::native::(anonymous namespace)::nll_loss_backward_reduce_cuda_kernel_2d<float, long>(float*, float const*, long const*, fl', 'loss'),
    ('void (anonymous namespace)::softmax_warp_forward<float, float, float, 9, true, false, 32>(float*, float const*, int, int, int, boo', 'loss'),
    ('triton_poi_fused__to_copy_permute_0', 'elementwise'),
    ('triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_0', 'elementwise'),
    ('triton_red_fused__to_copy_add_mul_sum_view_3', 'elementwise'),
    ('triton_poi_fused__unsafe_view_mul_silu_silu_backward_view_2', 'elementwise'),
    ('void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>, false>(int, at::nati', 'elementwise'),
    ('void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::lerp_scalar_kernel(at::TensorIteratorBase&, c', 'elementwise'),
    ('void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::silu_kernel(at::TensorIteratorBase&)::{lambda', 'elementwise'),
    ('void at::native::(anonymous namespace)::multi_tensor_apply_kernel<at::native::(anonymous namespace)::FusedOptimizerTensorListMetad', 'elementwise'),
    ('void at::native::reduce_kernel<512, 1, at::native::ReduceOp<c10::BFloat16, at::native::NormTwoOps<c10::BFloat16, float, c10::BFloa', 'elementwise'),
    ('void at::native::(anonymous namespace)::CatArrayBatchedCopy<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned int, 4, 64', 'elementwise'),
    ('void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1', 'elementwise'),
    ('void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::masked_fill_kern', 'elementwise'),
    ('Memcpy DtoD (Device -> Device)', 'other'),
    ('Memcpy DtoH (Device -> Pinned)', 'other'),
    ('Memset (Device)', 'other'),
    ('void at_cuda_detail::cub::detail::scan::DeviceScanKernel<at_cuda_detail::cub::detail::scan::policy_hub<long, long, long, unsigned ', 'other'),
    ('void at_cuda_detail::cub::detail::scan::DeviceScanInitKernel<at_cuda_detail::cub::ScanTileState<long, true> >(at_cuda_detail::cub:', 'other'),
    ('void compute_cuda_kernel<long>(long const*, long const*, long*, long, long)', 'other'),
    ('void at::native::(anonymous namespace)::searchsorted_cuda_kernel<long, long>(long*, long const*, long const*, long const*, long, l', 'other'),
    ('void at::native::warpMergeSortKVInPlace<2, -1, 128, 16, int, long, at::native::GTOp<int, true>, unsigned int, 32>(at::cuda::detail', 'other'),
]


@pytest.mark.parametrize("name, expected", REAL_KERNELS, ids=[n[:50] for n, _ in REAL_KERNELS])
def test_categorize_kernel(name, expected):
    assert categorize_kernel(name) == expected


def test_categorize_kernel_unknown_is_other():
    assert categorize_kernel("some_future_kernel_v9") == "other"
    assert set(c for _, c in REAL_KERNELS) == set(CATEGORIES)  # a tabela cobre todas as categorias


def _fake_event(name, start_us, end_us, kind="kernel"):
    return SimpleNamespace(
        name=name,
        device_type=torch.autograd.DeviceType.CUDA,
        activity_type=kind,
        time_range=SimpleNamespace(start=start_us, end=end_us),
    )


def _fake_prof(*events):
    cpu = SimpleNamespace(
        name="aten::mm", device_type=torch.autograd.DeviceType.CPU, activity_type="cpu_op",
        time_range=SimpleNamespace(start=0.0, end=9e6),
    )
    return SimpleNamespace(events=lambda: [cpu, *events])


def test_union_seconds_counts_overlap_once():
    ivs = [("a", 0.0, 2.0), ("b", 1.0, 3.0), ("c", 1.5, 2.5), ("d", 5.0, 6.0)]
    assert union_seconds(ivs) == pytest.approx(4.0)
    assert union_seconds([]) == 0.0


def test_breakdown_and_idle_from_events():
    prof = _fake_prof(
        _fake_event("chunk_fwd_kernel_o", 0, 1_000_000),
        _fake_event("ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_tn", 500_000, 2_000_000),  # sobrepõe
        _fake_event("Optimizer.step#Muon.step", 0, 3_000_000, kind="gpu_user_annotation"),  # ignorada
        _fake_event("Memcpy DtoD (Device -> Device)", 3_000_000, 3_500_000, kind="gpu_memcpy"),
    )
    kernels = kernel_breakdown(prof)
    assert tuple(kernels) == CATEGORIES
    assert kernels["gdn"] == pytest.approx(1.0)
    assert kernels["matmul"] == pytest.approx(1.5)
    assert kernels["other"] == pytest.approx(0.5)
    assert len(kernel_intervals(prof)) == 3
    # união: [0, 2] + [3, 3.5] = 2.5 s de GPU ocupada; parede 4 s -> 1.5 s ociosos
    assert idle_seconds(prof, 4.0) == pytest.approx(1.5)
    assert idle_seconds(prof, 1.0) == 0.0  # nunca negativo


def _load_bench():
    spec = importlib.util.spec_from_file_location("bench_script", Path(__file__).resolve().parents[2] / "scripts" / "bench.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_synthetic_batch_with_doc_lengths_varies_documents_inside_windows():
    bench = _load_bench()
    seq, mb = 64, 3
    gen, rng = torch.Generator().manual_seed(0), np.random.default_rng(0)
    n_docs = set()
    for _ in range(12):
        ids, labels, cu = bench.synthetic_batch(seq, mb, 100, gen, [5, 9, 20, 40, 70], rng)
        assert ids.shape == labels.shape == (1, seq * mb)
        assert cu.dtype == torch.long and cu[0] == 0 and cu[-1] == seq * mb
        assert set(range(0, seq * mb + 1, seq)) <= set(cu.tolist())  # nenhum documento atravessa a janela
        assert (cu.diff() > 0).all()
        assert (labels[0, cu[1:] - 1] == -100).all()
        n_docs.add(len(cu) - 1)
    assert len(n_docs) > 1


def test_synthetic_batch_default_keeps_four_documents_per_window():
    bench = _load_bench()
    _, _, cu = bench.synthetic_batch(64, 2, 100, torch.Generator().manual_seed(0))
    assert len(cu) - 1 == 2 * bench.DOCS_PER_WINDOW


@pytest.mark.cuda
def test_kernel_breakdown_on_real_profile():
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    (x @ x).sum().item()  # aquece cuBLAS fora do perfil
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        (x @ x).sum().item()
    kernels = kernel_breakdown(prof)
    assert set(kernels) == set(CATEGORIES)
    assert kernels["matmul"] > 0


@pytest.mark.cuda
@pytest.mark.fla
def test_bench_smoke_profile(tmp_path):
    out = tmp_path / "bench.jsonl"
    root = Path(__file__).resolve().parents[2]
    subprocess.run(
        [
            sys.executable, str(root / "scripts" / "bench.py"),
            "--preset", "tiny", "--override", "gdn_backend=fla", "--override", "attn_backend=flex",
            "--seq-len", "256", "--micro-batch", "1", "--steps", "3", "--warmup", "1",
            "--profile", "--out", str(out),
        ],
        check=True, cwd=root, timeout=600,
    )
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["overrides"] == {"gdn_backend": "fla", "attn_backend": "flex"}
    assert set(row["phases"]) == {"data", "fwd_bwd", "clip", "optimizer"}
    assert all(v > 0 for v in row["phases"].values())
    assert set(row["kernels"]) == set(CATEGORIES) | {"idle"}
    assert row["kernels"]["gdn"] > 0 and row["kernels"]["attention"] > 0 and row["kernels"]["matmul"] > 0
    # as fases cobrem o passo inteiro (o passo medido inclui a montagem do batch)
    assert sum(row["phases"].values()) == pytest.approx(row["step_s"], rel=0.25)
