"""Testes das políticas de checkpoint de ativação, checkpoint_frac e compilação do Block."""
import pytest
import torch

from sore.model import SoreLM, get_preset
from sore.model.checkpointing import checkpointed_blocks
from sore.train.trainer import compile_model


def test_checkpointed_blocks():
    # frac=1.0 dá todos os blocos
    assert checkpointed_blocks(24, 1.0) == frozenset(range(24))
    assert checkpointed_blocks(4, 1.0) == frozenset(range(4))

    # (24, 0.5) dá 12 índices sem dois vizinhos
    blocks_half = checkpointed_blocks(24, 0.5)
    assert len(blocks_half) == 12
    assert all(i in range(24) for i in blocks_half)
    sorted_blocks = sorted(blocks_half)
    for a, b in zip(sorted_blocks, sorted_blocks[1:]):
        assert b - a > 1, f"vizinhos consecutivos encontrados: {a}, {b} em {sorted_blocks}"

    # frac fora de (0, 1] levanta ValueError
    for invalid_frac in (-0.5, 0.0, 1.05, 2.0):
        with pytest.raises(ValueError, match="checkpoint_frac"):
            checkpointed_blocks(24, invalid_frac)


@pytest.mark.cuda
@pytest.mark.fla
@pytest.mark.parametrize(
    "policy, frac, compile_blocks",
    [
        ("full", 1.0, False),
        ("full", 1.0, True),
        ("full", 0.5, False),
        ("full", 0.5, True),
        ("ffn", 1.0, False),
        ("ffn", 1.0, True),
        ("ffn", 0.5, False),
        ("ffn", 0.5, True),
        ("selective", 1.0, False),
        ("selective", 1.0, True),
        ("selective", 0.5, False),
        ("selective", 0.5, True),
    ],
    ids=[
        "full_1.0_eager", "full_1.0_compile",
        "full_0.5_eager", "full_0.5_compile",
        "ffn_1.0_eager", "ffn_1.0_compile",
        "ffn_0.5_eager", "ffn_0.5_compile",
        "selective_1.0_eager", "selective_1.0_compile",
        "selective_0.5_eager", "selective_0.5_compile",
    ],
)
def test_checkpoint_policies_match_no_checkpoint(policy, frac, compile_blocks):
    # tiny (fla/flex, MTP), em fp32: loss e gradientes iguais aos de grad_checkpoint=False com rel 2e-3.
    # com compile, todos os Blocks continuam compilados e nenhum Block sem checkpoint vira eager.
    torch._dynamo.reset()
    try:
        base_cfg = get_preset(
            "tiny",
            gdn_backend="fla",
            attn_backend="flex",
            grad_checkpoint=False,
        )
        torch.manual_seed(0)
        ref_model = SoreLM(base_cfg).cuda()
        input_ids = torch.randint(1, base_cfg.vocab_size, (1, 256), device="cuda")
        cu_seqlens = torch.tensor([0, 100, 256], device="cuda", dtype=torch.long)
        cu_cpu = cu_seqlens.cpu()

        # Execução de referência sem checkpoint
        out_ref = ref_model(input_ids, cu_seqlens=cu_seqlens, labels=input_ids, cu_seqlens_cpu=cu_cpu)
        out_ref.loss.backward()
        ref_loss = out_ref.loss.detach()
        ref_grads = {name: p.grad.clone() for name, p in ref_model.named_parameters() if p.grad is not None}

        # Modelo sob teste com política e fração
        test_cfg = get_preset(
            "tiny",
            gdn_backend="fla",
            attn_backend="flex",
            grad_checkpoint=True,
            checkpoint_policy=policy,
            checkpoint_frac=frac,
        )
        test_model = SoreLM(test_cfg).cuda()
        test_model.load_state_dict(ref_model.state_dict())
        if compile_blocks:
            compile_model(test_model)
            # Todos os blocks (do tronco e MTP) continuam compilados
            for b in test_model.blocks:
                assert b._compiled_call_impl is not None, f"Block {b.layer_idx} não está compilado"
            if test_model.mtp is not None:
                assert test_model.mtp.block._compiled_call_impl is not None

        out_test = test_model(input_ids, cu_seqlens=cu_seqlens, labels=input_ids, cu_seqlens_cpu=cu_cpu)
        out_test.loss.backward()

        torch.testing.assert_close(out_test.loss, ref_loss, rtol=2e-3, atol=1e-4)
        for name, g_ref in ref_grads.items():
            g_test = dict(test_model.named_parameters())[name].grad
            diff = (g_test - g_ref).abs().max().item()
            scale = max(g_ref.abs().max().item(), 1e-6)
            assert diff <= 2e-3 * scale, f"{name}: diff={diff:.3e} > 2e-3 * scale={scale:.3e}"
    finally:
        torch._dynamo.reset()


@pytest.mark.cuda
@pytest.mark.fla
@pytest.mark.parametrize("compile_blocks", [False, True], ids=["eager", "compile"])
def test_checkpoint_policy_ffn_hyper_parity(compile_blocks):
    # A política "ffn" também com residual="hyper", que é só de paridade (spec §4)
    torch._dynamo.reset()
    try:
        base_cfg = get_preset(
            "tiny",
            gdn_backend="fla",
            attn_backend="flex",
            residual="hyper",
            grad_checkpoint=False,
        )
        torch.manual_seed(0)
        ref_model = SoreLM(base_cfg).cuda()
        input_ids = torch.randint(1, base_cfg.vocab_size, (1, 256), device="cuda")
        cu_seqlens = torch.tensor([0, 100, 256], device="cuda", dtype=torch.long)
        cu_cpu = cu_seqlens.cpu()

        out_ref = ref_model(input_ids, cu_seqlens=cu_seqlens, labels=input_ids, cu_seqlens_cpu=cu_cpu)
        out_ref.loss.backward()
        ref_loss = out_ref.loss.detach()
        ref_grads = {name: p.grad.clone() for name, p in ref_model.named_parameters() if p.grad is not None}

        test_cfg = get_preset(
            "tiny",
            gdn_backend="fla",
            attn_backend="flex",
            residual="hyper",
            grad_checkpoint=True,
            checkpoint_policy="ffn",
            checkpoint_frac=1.0,
        )
        test_model = SoreLM(test_cfg).cuda()
        test_model.load_state_dict(ref_model.state_dict())
        if compile_blocks:
            compile_model(test_model)

        out_test = test_model(input_ids, cu_seqlens=cu_seqlens, labels=input_ids, cu_seqlens_cpu=cu_cpu)
        out_test.loss.backward()

        torch.testing.assert_close(out_test.loss, ref_loss, rtol=2e-3, atol=1e-4)
        for name, g_ref in ref_grads.items():
            g_test = dict(test_model.named_parameters())[name].grad
            diff = (g_test - g_ref).abs().max().item()
            scale = max(g_ref.abs().max().item(), 1e-6)
            assert diff <= 2e-3 * scale, f"{name}: diff={diff:.3e} > 2e-3 * scale={scale:.3e}"
    finally:
        torch._dynamo.reset()


@pytest.mark.cuda
@pytest.mark.fla
def test_checkpoint_policy_memory_order():
    # No s150m a 2048: pico com grad_checkpoint=False > "ffn" > "full". O pico de "selective" fica abaixo do sem checkpoint.
    seq_len = 2048
    peaks = {}

    configs = {
        "no_ckpt": dict(grad_checkpoint=False),
        "ffn": dict(grad_checkpoint=True, checkpoint_policy="ffn", checkpoint_frac=1.0),
        "full": dict(grad_checkpoint=True, checkpoint_policy="full", checkpoint_frac=1.0),
        "selective": dict(grad_checkpoint=True, checkpoint_policy="selective", checkpoint_frac=1.0),
    }

    input_ids = torch.randint(1, 1000, (1, seq_len), device="cuda")
    cu_seqlens = torch.tensor([0, seq_len], device="cuda", dtype=torch.long)
    cu_cpu = cu_seqlens.cpu()

    for name, overrides in configs.items():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        cfg = get_preset("s150m", **overrides)
        torch.manual_seed(42)
        model = SoreLM(cfg).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids, cu_seqlens=cu_seqlens, labels=input_ids, cu_seqlens_cpu=cu_cpu)
            out.loss.backward()
        peaks[name] = torch.cuda.max_memory_allocated()
        del model, out
        torch.cuda.empty_cache()

    # no_ckpt > ffn > full
    assert peaks["no_ckpt"] > peaks["ffn"], f"no_ckpt ({peaks['no_ckpt']}) <= ffn ({peaks['ffn']})"
    assert peaks["ffn"] > peaks["full"], f"ffn ({peaks['ffn']}) <= full ({peaks['full']})"
    # selective < no_ckpt
    assert peaks["selective"] < peaks["no_ckpt"], f"selective ({peaks['selective']}) >= no_ckpt ({peaks['no_ckpt']})"


@pytest.mark.cuda
@pytest.mark.fla
def test_attention_block_compiles_once():
    # Com compile_model, dois forwards de mesma forma e número de documentos diferente
    # não aumentam torch._dynamo.utils.counters["stats"]["unique_graphs"] depois do primeiro.
    from torch._dynamo.utils import counters

    torch._dynamo.reset()
    try:
        cfg = get_preset("tiny", gdn_backend="fla", attn_backend="flex", grad_checkpoint=True)
        torch.manual_seed(0)
        model = SoreLM(cfg).cuda()
        compile_model(model)

        N = 256
        gen = torch.Generator().manual_seed(1)
        ids = torch.randint(1, 512, (1, N), generator=gen).cuda()

        # 1o forward: 2 documentos
        cu1 = torch.tensor([0, 100, N], device="cuda", dtype=torch.long)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out1 = model(ids, cu_seqlens=cu1, labels=ids, cu_seqlens_cpu=cu1.cpu())
            out1.loss.backward()
        model.zero_grad(set_to_none=True)
        g1 = counters["stats"]["unique_graphs"]
        assert g1 > 0

        # 2o forward: 4 documentos (mesmo N=256)
        cu2 = torch.tensor([0, 50, 120, 200, N], device="cuda", dtype=torch.long)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out2 = model(ids, cu_seqlens=cu2, labels=ids, cu_seqlens_cpu=cu2.cpu())
            out2.loss.backward()
        model.zero_grad(set_to_none=True)
        g2 = counters["stats"]["unique_graphs"]

        assert g2 == g1, f"Dynamo recompilou grafos com documento diferente: g1={g1}, g2={g2}"
    finally:
        torch._dynamo.reset()
