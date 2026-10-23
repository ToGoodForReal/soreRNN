"""Testes do trainer: overfit, retomada exata através da troca de estágio, passo não finito e config."""
import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

import sore.model.lm as lm_module
from sore.model import get_preset
from sore.train.config import Stage, TrainConfig, load_train_config, resolve_data_paths
from sore.train.data import write_token_shard
from sore.train.trainer import TrainSummary, train

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
        "seq_len", "skipped",
    }
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
            out.loss = out.loss * float("nan")
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


def test_compile_disables_mix_order_reduction(tmp_path, monkeypatch):
    # torch 2.14.1: o Inductor quebra no backward com mix_order_reduction ligado; o trainer desliga
    # antes de compilar. torch.compile é trocado por identidade (teste barato, sem compilar nada).
    import torch._inductor.config as inductor_config

    monkeypatch.setattr(inductor_config.triton, "mix_order_reduction", True)  # restaurado ao fim
    compiled = []

    def fake_compile(model, *args, **kwargs):
        compiled.append(inductor_config.triton.mix_order_reduction)
        return model

    monkeypatch.setattr(torch, "compile", fake_compile)
    shard = _shard(tmp_path / "s.bin", 1000)
    cfg = _cfg(tmp_path / "run", shard, [Stage(10 * 128, 32, 4)], steps=10, compile=True)
    assert train(cfg, max_steps=1).steps == 1
    assert compiled == [False]  # flag já desligada quando torch.compile é chamado


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
    # defaults do brief
    assert cfg.model_overrides == {}
    assert (cfg.lr_muon, cfg.warmup_steps, cfg.decay_frac, cfg.weight_decay, cfg.grad_clip) == (
        0.02, 500, 0.2, 0.1, 1.0
    )
    assert (cfg.compile, cfg.log_every, cfg.seed, cfg.device) == (False, 10, 0, "auto")


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
        (GOOD_STAGES, "60e9", "", "total_tokens"),
        ("  []", "50e9", "", "estágio"),
        (GOOD_STAGES, "50e9", "seq_length: 4096\n", "desconhecid"),
        (GOOD_STAGES, "50e9", "warmup_steps: 2.5\n", "warmup_steps"),
    ],
    ids=["tokens_per_microbatch", "not_ascending", "last_below_total", "empty", "unknown_key", "fractional"],
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


def test_resolve_data_paths(tmp_path):
    for name in ("b.bin", "a.bin"):
        _shard(tmp_path / name, 100)
    assert resolve_data_paths([str(tmp_path / "*.bin")]) == [tmp_path / "a.bin", tmp_path / "b.bin"]
    with pytest.raises(FileNotFoundError, match="nada"):
        resolve_data_paths([str(tmp_path / "*.bin"), str(tmp_path / "missing*.bin")])
