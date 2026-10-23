"""Testes do trainer: overfit, retomada exata através da troca de estágio, passo não finito e config."""
import dataclasses
import json
import math
import os
import signal
import time
from pathlib import Path

import numpy as np
import pytest
import torch

import sore.model.lm as lm_module
import sore.train.trainer as trainer_module
from sore.model import get_preset
from sore.model.flops import flops_per_token
from sore.train.config import Stage, TrainConfig, load_train_config, resolve_data_paths
from sore.train.data import write_token_shard
from sore.train.trainer import LossSpikeDetector, TrainSummary, compile_model, train

REPO = Path(__file__).resolve().parents[2]
CPU_BACKENDS = {"gdn_backend": "torch", "attn_backend": "sdpa"}


def _shard(path: Path, n_tokens: int, seed: int = 0, vocab: int = 512) -> Path:
    """Documentos curtos em progressão aritmética (aprendível), cada um terminado por EOS = 0."""
    rng = np.random.default_rng(seed)
    out: list[int] = []
    while len(out) < n_tokens:
        length = int(rng.integers(8, 60))
        start, step = int(rng.integers(0, vocab - 1)), int(rng.integers(1, 4))
        out.extend((1 + (start + step * np.arange(length)) % (vocab - 1)).tolist())
        out.append(0)
    write_token_shard(path, np.array(out[:n_tokens]))
    return path


def _cfg(out_dir: Path, shard: Path, stages: list[Stage], steps: int, **overrides) -> TrainConfig:
    """Config de teste em CPU: `total_tokens` = `steps` passos do formato do 1º estágio."""
    grad_accum = overrides.pop("grad_accum", 1)
    tokens_per_step = stages[0].seq_len * stages[0].micro_batch * grad_accum
    base = dict(
        model_preset="tiny",
        model_overrides=dict(CPU_BACKENDS),
        data_paths=[str(shard)],
        eos_id=0,
        grad_accum=grad_accum,
        stages=stages,
        total_tokens=steps * tokens_per_step,
        warmup_steps=2,
        decay_frac=0.2,
        ckpt_every_tokens=10**12,
        out_dir=str(out_dir),
        log_every=1000,
        device="cpu",
    )
    base.update(overrides)
    return TrainConfig(**base)


def _metrics(out_dir: str) -> list[dict]:
    lines = (Path(out_dir) / "metrics.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


def _ckpt_model_state(out_dir: str) -> dict:
    return torch.load(Path(out_dir) / "last.pt", map_location="cpu", weights_only=True)["model"]


# --- treino -----------------------------------------------------------------------------------


def test_overfit(tmp_path):
    # shard pequeno (31 janelas de 32) repetido por ~5 épocas em 40 passos; lr do AdamW (embedding
    # amarrado à cabeça) acima do default para memorizar rápido (razão final/inicial ~0.24)
    shard = _shard(tmp_path / "s.bin", 1000)
    cfg = _cfg(tmp_path / "run", shard, [Stage(40 * 128, 32, 4)], steps=40, lr_adamw=1e-2)
    summary = train(cfg)

    assert isinstance(summary, TrainSummary)
    assert summary.steps == 40 and summary.skipped_steps == 0
    assert summary.tokens == 40 * 128
    metrics = _metrics(cfg.out_dir)
    assert [m["step"] for m in metrics] == list(range(1, 41))
    assert set(metrics[0]) >= {
        "step", "tokens", "loss", "lm_loss", "mtp_loss", "grad_norm", "lr_muon", "tok_s", "mem_gb",
        "seq_len", "skipped", "mfu", "mtp_weight", "spike",
    }
    # λ do MTP: o mtp_weight do modelo (0.3) no 1º passo até mtp_weight_end (0.1) no último
    assert metrics[0]["mtp_weight"] == pytest.approx(0.3) and metrics[-1]["mtp_weight"] == pytest.approx(0.1)
    assert not any(m["spike"] for m in metrics)
    losses = [m["loss"] for m in metrics]
    assert summary.last_loss == pytest.approx(losses[-1])
    assert losses[-1] < 0.5 * losses[0], losses
    assert (Path(cfg.out_dir) / "last.pt").exists()


# grad_accum=2 e 256 tokens por passo nos dois estágios: a troca 32x4 -> 64x2 cai no fim do passo 3
RESUME_STAGES = [Stage(3 * 256, 32, 4), Stage(6 * 256, 64, 2)]


@pytest.fixture(scope="module")
def straight_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("straight")
    shard = _shard(tmp / "s.bin", 4000)
    cfg = _cfg(tmp / "run", shard, RESUME_STAGES, steps=6, grad_accum=2)
    return cfg, train(cfg)


@pytest.mark.parametrize("split", [3, 4], ids=["at_stage_switch", "mid_stage"])
def test_resume_equivalence(tmp_path, straight_run, split):
    straight_cfg, straight = straight_run
    straight_metrics = _metrics(straight_cfg.out_dir)
    assert [m["seq_len"] for m in straight_metrics] == [32, 32, 32, 64, 64, 64]

    cfg = dataclasses.replace(straight_cfg, out_dir=str(tmp_path / "run"))
    first = train(cfg, max_steps=split)
    assert first.steps == split
    # processo novo: retoma do last.pt e vai até total_steps
    resumed = train(cfg, resume=Path(cfg.out_dir) / "last.pt")

    assert (resumed.steps, resumed.tokens, resumed.skipped_steps) == (6, straight.tokens, 0)
    metrics = _metrics(cfg.out_dir)
    assert [m["step"] for m in metrics] == list(range(1, 7))
    assert [m["seq_len"] for m in metrics] == [32, 32, 32, 64, 64, 64]
    after = slice(split, None)
    np.testing.assert_allclose(
        [m["loss"] for m in metrics[after]], [m["loss"] for m in straight_metrics[after]], rtol=0, atol=1e-5
    )
    assert [m["tokens"] for m in metrics] == [m["tokens"] for m in straight_metrics]
    final, final_straight = _ckpt_model_state(cfg.out_dir), _ckpt_model_state(straight_cfg.out_dir)
    for name, value in final_straight.items():
        torch.testing.assert_close(final[name], value, rtol=0, atol=1e-5, msg=name)
    # R22: unidades do tamanho do maior seq_len, então a troca 32 -> 64 não reamostra o corpus
    ckpt = torch.load(Path(cfg.out_dir) / "last.pt", weights_only=True)
    assert ckpt["data_state"]["unit_len"] == 64
    # a EMA do alerta de spike também é retomada
    ema_straight = torch.load(Path(straight_cfg.out_dir) / "last.pt", weights_only=True)["train_state"]["loss_ema"]
    assert ckpt["train_state"]["loss_ema"] == pytest.approx(ema_straight, rel=0, abs=1e-5)


def test_nonfinite_step_skipped(tmp_path, monkeypatch):
    shard = _shard(tmp_path / "s.bin", 2000)
    # warmup 0: o passo 1 já anda com lr > 0, então "parâmetros do passo 1" != init
    cfg = _cfg(tmp_path / "run", shard, [Stage(10 * 128, 32, 4)], steps=10, warmup_steps=0)

    original = lm_module.SoreLM.forward
    snapshots: list[dict[str, torch.Tensor]] = []  # parâmetros no início de cada forward (= de cada passo)

    def forward_nan_on_step_2(self, *args, **kwargs):
        snapshots.append({n: p.detach().clone() for n, p in self.named_parameters()})
        out = original(self, *args, **kwargs)
        if len(snapshots) == 2:
            out.lm_loss = out.lm_loss * float("nan")  # o trainer monta o loss de lm_loss e mtp_loss (R23)
        return out

    monkeypatch.setattr(lm_module.SoreLM, "forward", forward_nan_on_step_2)
    summary = train(cfg, max_steps=3)

    assert summary.steps == 3 and summary.skipped_steps == 1
    metrics = _metrics(cfg.out_dir)
    assert [m["skipped"] for m in metrics] == [False, True, False]
    assert metrics[1]["loss"] is None and metrics[1]["grad_norm"] is None  # não finito -> null
    assert math.isfinite(metrics[2]["loss"]) and math.isfinite(metrics[2]["grad_norm"])
    initial, after_step_1, after_step_2 = snapshots
    assert any(not torch.equal(initial[n], after_step_1[n]) for n in initial)
    for name, value in after_step_1.items():
        assert torch.equal(after_step_2[name], value), name


def test_step_loss_is_token_weighted_mean(tmp_path, monkeypatch):
    # 8 janelas de 32 e grad_accum=8 (mb 1): o passo 1 vê todas. Janela 4 só tem EOS na entrada
    # (0 tokens válidos: o micro-batch é pulado) e a janela 5 alterna EOS (16 válidos).
    tokens = np.random.default_rng(0).integers(1, 512, size=8 * 32 + 1)
    tokens[100] = 0
    tokens[128:160] = 0
    tokens[161:192:2] = 0
    shard = tmp_path / "s.bin"
    write_token_shard(shard, tokens)
    cfg = _cfg(tmp_path / "run", shard, [Stage(10 * 256, 32, 1)], steps=10, grad_accum=8, warmup_steps=0)

    original = lm_module.SoreLM.forward
    calls, initial = [], {}

    def recording_forward(self, input_ids, cu_seqlens=None, labels=None, **kwargs):
        if not initial:
            initial.update({k: v.detach().clone() for k, v in self.state_dict().items()})
        out = original(self, input_ids, cu_seqlens=cu_seqlens, labels=labels, **kwargs)
        calls.append((input_ids.clone(), cu_seqlens.clone(), labels.clone(), out))
        return out

    monkeypatch.setattr(lm_module.SoreLM, "forward", recording_forward)
    train(cfg, max_steps=1)
    monkeypatch.undo()

    counts = [out.n_tokens for *_, out in calls]
    assert len(calls) == 7 and 16 in counts  # o micro-batch sem tokens válidos não roda
    total = sum(counts)
    (metrics,) = _metrics(cfg.out_dir)
    for key in ("loss", "lm_loss", "mtp_loss"):
        expected = sum(getattr(out, key).item() * out.n_tokens for *_, out in calls) / total
        assert metrics[key] == pytest.approx(expected, rel=1e-6), key

    # gradiente de referência: backward de sum(n_i * loss_i) / sum(n_i) nos mesmos micro-batches
    ref = lm_module.SoreLM(get_preset("tiny", **CPU_BACKENDS))
    ref.load_state_dict(initial)
    for ids, cu, labels, out in calls:
        (ref(ids, cu_seqlens=cu, labels=labels).loss * out.n_tokens / total).backward()
    ref_norm = torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in ref.parameters() if p.grad is not None]))
    assert metrics["grad_norm"] == pytest.approx(ref_norm.item(), rel=1e-5)


def test_step_loss_uses_mtp_weight_schedule(tmp_path, monkeypatch):
    # R23: o trainer faz o backward de lm + λ(passo)·mtp (não do out.loss, que usa o mtp_weight do modelo)
    shard = _shard(tmp_path / "s.bin", 4000)
    cfg = _cfg(
        tmp_path / "run", shard, [Stage(5 * 128, 32, 4)], steps=5, warmup_steps=0,
        mtp_weight_start=0.9, mtp_weight_end=0.1,
    )
    original = lm_module.SoreLM.forward
    calls, initial = [], {}

    def recording_forward(self, input_ids, cu_seqlens=None, labels=None, **kwargs):
        if not initial:
            initial.update({k: v.detach().clone() for k, v in self.state_dict().items()})
        calls.append((input_ids.clone(), cu_seqlens.clone(), labels.clone()))
        return original(self, input_ids, cu_seqlens=cu_seqlens, labels=labels, **kwargs)

    monkeypatch.setattr(lm_module.SoreLM, "forward", recording_forward)
    train(cfg)
    monkeypatch.undo()

    metrics = _metrics(cfg.out_dir)
    np.testing.assert_allclose([m["mtp_weight"] for m in metrics], [0.9, 0.7, 0.5, 0.3, 0.1], rtol=1e-12)
    for m in metrics:
        assert m["loss"] == pytest.approx(m["lm_loss"] + m["mtp_weight"] * m["mtp_loss"], rel=1e-6)

    # gradiente do passo 1 = o de lm + 0.9·mtp nos mesmos pesos e dados
    ref = lm_module.SoreLM(get_preset("tiny", **CPU_BACKENDS))
    ref.load_state_dict(initial)
    ids, cu, labels = calls[0]
    out = ref(ids, cu_seqlens=cu, labels=labels)
    (out.lm_loss + 0.9 * out.mtp_loss).backward()
    ref_norm = torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in ref.parameters() if p.grad is not None]))
    assert metrics[0]["grad_norm"] == pytest.approx(ref_norm.item(), rel=1e-5)


def test_spike_detector_ema():
    det = LossSpikeDetector(threshold=0.3, beta=0.5)
    assert det(2.0) is False and det.ema == 2.0  # a 1a chamada só inicializa a EMA
    assert det(1.0) is False and det.ema == 1.5
    assert det(1.96) is True and det.ema == 1.5  # > 1.5 * 1.3 = 1.95: spike não atualiza a EMA
    assert det(1.94) is False and det.ema == pytest.approx(1.72)
    assert det(math.nan) is False and det.ema == pytest.approx(1.72)  # não finito: ignorado


def test_loss_spike_flag(tmp_path, monkeypatch, capsys):
    shard = _shard(tmp_path / "s.bin", 4000)
    cfg = _cfg(tmp_path / "run", shard, [Stage(12 * 128, 32, 4)], steps=12, warmup_steps=2)
    original = lm_module.SoreLM.forward
    calls = []

    def forward_jump_on_step_8(self, *args, **kwargs):
        out = original(self, *args, **kwargs)
        calls.append(None)
        if len(calls) == 8:  # grad_accum 1: o forward 8 é o passo 8
            out.lm_loss, out.mtp_loss = out.lm_loss * 3, out.mtp_loss * 3
        return out

    monkeypatch.setattr(lm_module.SoreLM, "forward", forward_jump_on_step_8)
    train(cfg)

    assert [m["spike"] for m in _metrics(cfg.out_dir)] == [False] * 7 + [True] + [False] * 4
    assert "[spike] passo 8" in capsys.readouterr().out


def test_mtp_weight_can_change_on_resume(tmp_path):
    # mtp_weight é campo de execução: não muda os pesos, então a retomada aceita outro valor
    shard = _shard(tmp_path / "s.bin", 1000)
    cfg = _cfg(tmp_path / "run", shard, [Stage(10 * 128, 32, 4)], steps=10)
    train(cfg, max_steps=1)
    changed = dataclasses.replace(cfg, model_overrides={**CPU_BACKENDS, "mtp_weight": 0.5})
    assert train(changed, resume=Path(cfg.out_dir) / "last.pt", max_steps=1).steps == 2


def test_metrics_include_mfu(tmp_path):
    # mfu = flops_per_token(cfg do modelo, seq_len do passo) * tok_s / (peak_tflops * 1e12)
    shard = _shard(tmp_path / "s.bin", 4000)
    cfg = _cfg(tmp_path / "run", shard, RESUME_STAGES, steps=6, grad_accum=2, peak_tflops=0.5)
    train(cfg, max_steps=5)  # cruza a troca de estágio 32 -> 64

    model_cfg = get_preset("tiny", **CPU_BACKENDS)
    metrics = _metrics(cfg.out_dir)
    assert [m["seq_len"] for m in metrics] == [32, 32, 32, 64, 64]
    for m in metrics:
        expected = flops_per_token(model_cfg, m["seq_len"]) * m["tok_s"] / (0.5 * 1e12)
        assert m["mfu"] == pytest.approx(expected, rel=1e-12) and m["mfu"] > 0


# --- checkpoints e sinais (R27) ---------------------------------------------------------------


def _record_saves(monkeypatch) -> list[tuple[str, int]]:
    """Grava `(nome do arquivo, train_state["step"])` de cada `save_checkpoint` do trainer."""
    saves = []
    real = trainer_module.save_checkpoint

    def recording(path, model, optimizers, data_state, train_state):
        saves.append((Path(path).name, train_state["step"]))
        real(path, model, optimizers, data_state, train_state)

    monkeypatch.setattr(trainer_module, "save_checkpoint", recording)
    return saves


def test_refuses_to_overwrite_existing_run(tmp_path):
    shard = _shard(tmp_path / "s.bin", 1000)
    cfg = _cfg(tmp_path / "run", shard, [Stage(10 * 128, 32, 4)], steps=10)
    train(cfg, max_steps=1)
    with pytest.raises(FileExistsError, match="--resume .*last.pt"):
        train(cfg)
    assert len(_metrics(cfg.out_dir)) == 1  # a 2a chamada não treinou nem escreveu nada


def test_periodic_checkpoint(tmp_path, monkeypatch):
    saves = _record_saves(monkeypatch)
    shard = _shard(tmp_path / "s.bin", 1000)
    # 128 tokens por passo: ckpt a cada 2 passos e o final no passo 5
    cfg = _cfg(tmp_path / "run", shard, [Stage(10 * 128, 32, 4)], steps=10, ckpt_every_tokens=2 * 128)
    train(cfg, max_steps=5)
    assert saves == [("last.pt", 2), ("last.pt", 4), ("last.pt", 5)]


def test_snapshots_keep_the_newest(tmp_path, monkeypatch):
    saves = _record_saves(monkeypatch)
    shard = _shard(tmp_path / "s.bin", 1000)
    cfg = _cfg(
        tmp_path / "run", shard, [Stage(10 * 128, 32, 4)], steps=10, snapshot_every_tokens=128, keep_snapshots=2
    )
    train(cfg, max_steps=5)

    # cada snapshot sai junto com um last.pt do mesmo passo
    assert saves == [(name, s) for s in range(1, 6) for name in ("last.pt", f"snapshot_step{s:08d}.pt")]
    out = Path(cfg.out_dir)
    snaps = sorted(p.name for p in out.glob("snapshot_step*.pt"))
    assert snaps == ["snapshot_step00000004.pt", "snapshot_step00000005.pt"]  # só os 2 mais novos
    assert not list(out.glob("*.tmp"))
    last = torch.load(out / "last.pt", weights_only=True)
    newest = torch.load(out / snaps[-1], weights_only=True)
    assert newest["train_state"] == last["train_state"] and newest["train_state"]["step"] == 5
    for name, value in last["model"].items():
        assert torch.equal(newest["model"][name], value), name


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT], ids=["SIGTERM", "SIGINT"])
def test_signal_saves_at_step_boundary(tmp_path, monkeypatch, capsys, signum):
    shard = _shard(tmp_path / "s.bin", 1000)
    cfg = _cfg(tmp_path / "run", shard, [Stage(10 * 128, 32, 4)], steps=10)
    before = signal.getsignal(signum)
    original = lm_module.SoreLM.forward
    calls = []

    def forward_signal_on_step_2(self, *args, **kwargs):
        calls.append(signal.getsignal(signum))
        if len(calls) == 2:
            assert calls[-1] is not before  # o handler do trainer está instalado (senão não envia)
            # Ctrl-C sob `uv run`: o Python recebe o SIGINT do terminal e o repassado pelo uv
            os.kill(os.getpid(), signum)
            os.kill(os.getpid(), signum)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(lm_module.SoreLM, "forward", forward_signal_on_step_2)
    summary = train(cfg)

    assert summary.steps == 2 and len(calls) == 2  # o passo 2 terminou e o treino parou na fronteira
    assert torch.load(Path(cfg.out_dir) / "last.pt", weights_only=True)["train_state"]["step"] == 2
    assert [m["step"] for m in _metrics(cfg.out_dir)] == [1, 2]
    assert signal.getsignal(signum) is before  # restaurado na saída do train()
    out = capsys.readouterr().out
    assert signal.Signals(signum).name in out and "last.pt" in out


def test_second_signal_has_default_effect(tmp_path, monkeypatch):
    # um 2º SIGINT depois da janela de repetição volta ao efeito padrão: KeyboardInterrupt na hora
    monkeypatch.setattr(trainer_module, "_SIGNAL_REPEAT_S", 0.0)
    shard = _shard(tmp_path / "s.bin", 1000)
    cfg = _cfg(tmp_path / "run", shard, [Stage(10 * 128, 32, 4)], steps=10)
    before = signal.getsignal(signal.SIGINT)
    original = lm_module.SoreLM.forward
    calls = []

    def forward_two_signals_on_step_2(self, *args, **kwargs):
        calls.append(None)
        if len(calls) == 2:
            assert signal.getsignal(signal.SIGINT) is not before
            os.kill(os.getpid(), signal.SIGINT)  # 1º: pede a parada
            time.sleep(0.01)
            os.kill(os.getpid(), signal.SIGINT)  # 2º: efeito padrão
        return original(self, *args, **kwargs)

    monkeypatch.setattr(lm_module.SoreLM, "forward", forward_two_signals_on_step_2)
    with pytest.raises(KeyboardInterrupt):
        train(cfg)
    assert len(calls) == 2
    assert signal.getsignal(signal.SIGINT) is before


def test_compile_model_compiles_each_block_in_place(monkeypatch):
    # helper compartilhado pelo trainer e pelo scripts/bench.py: Module.compile() em cada Block do
    # tronco, no Block do MTP e na cross-entropy dos chunks da loss (N estático), com
    # mix_order_reduction já desligado; sem wrapper
    import torch._inductor.config as inductor_config

    monkeypatch.setattr(inductor_config.triton, "mix_order_reduction", True)  # restaurado ao fim
    seen = []

    def fake_compile(fn, *args, **kwargs):
        seen.append((fn.__self__, kwargs.get("dynamic"), inductor_config.triton.mix_order_reduction))
        return fn

    monkeypatch.setattr(torch, "compile", fake_compile)
    model = lm_module.SoreLM(get_preset("tiny", **CPU_BACKENDS))
    names = [n for n, _ in model.named_parameters()]
    assert compile_model(model) is model
    assert [m for m, *_ in seen] == [*model.blocks, model.mtp.block, model.chunk_ce]
    assert all(dynamic is False and mix is False for _, dynamic, mix in seen)
    assert [n for n, _ in model.named_parameters()] == names  # divisão Muon/AdamW intacta


def test_compile_disables_mix_order_reduction(tmp_path, monkeypatch):
    # torch 2.14.1: o Inductor quebra no backward com mix_order_reduction ligado; o trainer desliga
    # antes de compilar. torch.compile é trocado por identidade (teste barato, sem compilar nada).
    import torch._inductor.config as inductor_config

    monkeypatch.setattr(inductor_config.triton, "mix_order_reduction", True)  # restaurado ao fim
    compiled = []

    def fake_compile(fn, *args, **kwargs):
        compiled.append(inductor_config.triton.mix_order_reduction)
        return fn

    monkeypatch.setattr(torch, "compile", fake_compile)
    shard = _shard(tmp_path / "s.bin", 1000)
    cfg = _cfg(tmp_path / "run", shard, [Stage(10 * 128, 32, 4)], steps=10, compile=True)
    assert train(cfg, max_steps=1).steps == 1
    n_modules = get_preset("tiny").n_layers + 2  # Blocks do tronco + Block do MTP + cross-entropy da loss
    assert compiled == [False] * n_modules  # flag já desligada quando torch.compile é chamado


@pytest.mark.cuda
@pytest.mark.fla
def test_compile_model_matches_eager_with_grad_checkpoint():
    # R25: com grad checkpoint, compilar o modelo inteiro deixava o tronco eager (o checkpoint
    # desliga o Dynamo). Por bloco, loss e gradientes batem com o eager (fp32) e o S (nº de
    # documentos) varia sem quebrar o recompute do checkpoint.
    from torch._dynamo.utils import counters

    torch._dynamo.reset()
    try:
        cfg = get_preset("tiny", gdn_backend="fla", attn_backend="flex", grad_checkpoint=True)
        torch.manual_seed(0)
        eager = lm_module.SoreLM(cfg).cuda()
        compiled = lm_module.SoreLM(cfg).cuda()
        compiled.load_state_dict(eager.state_dict())
        assert compile_model(compiled) is compiled
        targets = [*compiled.blocks, compiled.mtp.block, compiled.chunk_ce]
        assert all(m._compiled_call_impl is not None for m in targets)
        frames_before = counters["frames"]["ok"]

        gen = torch.Generator().manual_seed(1)
        graphs = []  # grafos compilados após cada iteração
        for cu in ([0, 5, 140, 300], [0, 60, 300], [0, 1, 2, 100, 300], [0, 300]):  # S = 3, 2, 4, 1
            ids = torch.randint(1, 512, (1, 300), generator=gen).cuda()
            cu_seqlens = torch.tensor(cu, device="cuda")
            results = []
            for model in (eager, compiled):
                model.zero_grad(set_to_none=True)
                out = model(ids, cu_seqlens=cu_seqlens, labels=ids)
                out.loss.backward()
                results.append((out.loss.detach(), {n: p.grad for n, p in model.named_parameters()}))
            (loss_ref, grads_ref), (loss, grads) = results
            torch.testing.assert_close(loss, loss_ref, rtol=1e-5, atol=0)
            for name, g_ref in grads_ref.items():
                diff = (grads[name] - g_ref).abs().max().item()
                assert diff <= 2e-3 * g_ref.abs().max().item(), (cu, name, diff)
            graphs.append(counters["stats"]["unique_graphs"])
        assert counters["frames"]["ok"] > frames_before  # o Dynamo compilou de fato
        # S dinâmico desde o 1º compile: S = 2 e 4 reusam os grafos (S = 1 especializa à parte)
        assert graphs[1] == graphs[2] == graphs[0], graphs
    finally:
        torch._dynamo.reset()


# --- config -----------------------------------------------------------------------------------

BASE_YAML = """\
model_preset: tiny
data_paths: ["data/x.bin"]
eos_id: 0
grad_accum: 64
stages:
{stages}
total_tokens: {total}
ckpt_every_tokens: 2.5e8
out_dir: runs/x
"""
GOOD_STAGES = """\
  - {until_tokens: 35e9, seq_len: 4096, micro_batch: 2}
  - {until_tokens: 50e9, seq_len: 8192, micro_batch: 1}"""


def _yaml(tmp_path: Path, stages: str = GOOD_STAGES, total: str = "50e9", extra: str = "") -> Path:
    path = tmp_path / "cfg.yaml"
    path.write_text(BASE_YAML.format(stages=stages, total=total) + extra)
    return path


def test_load_train_config_coerces_scientific_notation(tmp_path):
    cfg = load_train_config(_yaml(tmp_path, extra="lr_adamw: 3e-3\n"))  # PyYAML lê "35e9"/"3e-3" como str
    assert cfg.total_tokens == 50_000_000_000 and type(cfg.total_tokens) is int
    assert cfg.stages == [Stage(35_000_000_000, 4096, 2), Stage(50_000_000_000, 8192, 1)]
    assert all(type(s.until_tokens) is int for s in cfg.stages)
    assert cfg.ckpt_every_tokens == 250_000_000 and type(cfg.ckpt_every_tokens) is int
    assert cfg.lr_adamw == 3e-3 and type(cfg.lr_adamw) is float
    # defaults do brief; lr_muon = lr_adamw porque o Muon usa match_rms_adamw (R21)
    assert cfg.model_overrides == {}
    assert (cfg.lr_muon, cfg.warmup_steps, cfg.decay_frac, cfg.weight_decay, cfg.grad_clip) == (
        3e-3, 500, 0.2, 0.1, 1.0
    )
    assert TrainConfig.lr_muon == TrainConfig.lr_adamw
    assert (cfg.compile, cfg.log_every, cfg.seed, cfg.device, cfg.peak_tflops) == (False, 10, 0, "auto", 165.0)
    assert (cfg.snapshot_every_tokens, cfg.keep_snapshots) == (0, 3)  # snapshots desligados por padrão
    assert (cfg.mtp_weight_start, cfg.mtp_weight_end, cfg.spike_threshold) == (None, 0.1, 0.3)


def test_load_train_config_peak_tflops(tmp_path):
    cfg = load_train_config(_yaml(tmp_path, extra="peak_tflops: 3.3e2\n"))  # str no PyYAML
    assert cfg.peak_tflops == 330.0 and type(cfg.peak_tflops) is float


@pytest.mark.parametrize(
    "stages, total, extra, match",
    [
        (
            "  - {until_tokens: 35e9, seq_len: 4096, micro_batch: 2}\n"
            "  - {until_tokens: 50e9, seq_len: 8192, micro_batch: 2}",
            "50e9", "", "seq_len \\* micro_batch",
        ),
        (
            "  - {until_tokens: 50e9, seq_len: 4096, micro_batch: 2}\n"
            "  - {until_tokens: 50e9, seq_len: 8192, micro_batch: 1}",
            "50e9", "", "crescente",
        ),
        (
            "  - {until_tokens: 35e9, seq_len: 3072, micro_batch: 4}\n"
            "  - {until_tokens: 50e9, seq_len: 4096, micro_batch: 3}",
            "50e9", "", "dividir",
        ),
        (GOOD_STAGES, "60e9", "", "total_tokens"),
        ("  []", "50e9", "", "estágio"),
        (GOOD_STAGES, "50e9", "seq_length: 4096\n", "desconhecid"),
        (GOOD_STAGES, "50e9", "warmup_steps: 2.5\n", "warmup_steps"),
        (GOOD_STAGES, "50e9", "peak_tflops: 0\n", "peak_tflops"),
        (GOOD_STAGES, "50e9", "snapshot_every_tokens: -1\n", "snapshot_every_tokens deve ser >= 0"),
        (GOOD_STAGES, "50e9", "keep_snapshots: 0\n", "keep_snapshots deve ser >= 1"),
        (GOOD_STAGES, "50e9", "spike_threshold: 0\n", "spike_threshold deve ser > 0"),
        (GOOD_STAGES, "50e9", "mtp_weight_end: -0.1\n", "mtp_weight_end deve ser >= 0"),
    ],
    ids=[
        "tokens_per_microbatch", "not_ascending", "seq_len_not_divisor", "last_below_total", "empty",
        "unknown_key", "fractional",
        "peak_tflops_zero", "snapshot_negative", "keep_zero", "spike_threshold_zero", "mtp_weight_negative",
    ],
)
def test_load_train_config_rejects_invalid(tmp_path, stages, total, extra, match):
    with pytest.raises(ValueError, match=match):
        load_train_config(_yaml(tmp_path, stages=stages, total=total, extra=extra))


@pytest.mark.parametrize(
    "name, preset, tokens_per_step, n_stages",
    [
        ("tiny_local.yaml", "tiny", 512, 2),
        ("s150m_ablation.yaml", "s150m", 131_072, 2),
        ("b0_8_pretrain.yaml", "b0_8", 524_288, 2),
    ],
)
def test_shipped_configs_load(name, preset, tokens_per_step, n_stages):
    cfg = load_train_config(REPO / "configs" / name)
    assert cfg.model_preset == preset and len(cfg.stages) == n_stages
    s0 = cfg.stages[0]
    assert s0.seq_len * s0.micro_batch * cfg.grad_accum == tokens_per_step
    get_preset(cfg.model_preset, **cfg.model_overrides)  # overrides são campos válidos


def test_b0_8_config_values():
    cfg = load_train_config(REPO / "configs" / "b0_8_pretrain.yaml")
    assert cfg.stages == [Stage(35_000_000_000, 4096, 2), Stage(50_000_000_000, 8192, 1)]
    assert (cfg.total_tokens, cfg.grad_accum, cfg.compile) == (50_000_000_000, 64, True)
    assert cfg.model_overrides == {"grad_checkpoint": True}
    # R27: last.pt a cada 5e7 tokens (~95 passos) e 4 snapshots numerados, um a cada 2,5e9
    assert (cfg.ckpt_every_tokens, cfg.snapshot_every_tokens, cfg.keep_snapshots) == (50_000_000, 2_500_000_000, 4)


def test_resolve_data_paths(tmp_path):
    for name in ("b.bin", "a.bin"):
        _shard(tmp_path / name, 100)
    assert resolve_data_paths([str(tmp_path / "*.bin")]) == [tmp_path / "a.bin", tmp_path / "b.bin"]
    with pytest.raises(FileNotFoundError, match="nada"):
        resolve_data_paths([str(tmp_path / "*.bin"), str(tmp_path / "missing*.bin")])
