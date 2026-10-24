"""Configuração de treino (`TrainConfig`, `Stage`) e leitura/validação do YAML.

O PyYAML (YAML 1.1) lê `35e9`, `2.5e8` e `3e-3` como string; os campos numéricos são
convertidos aqui (contagens inteiras via `float`, rejeitando valores fracionários).
"""
from __future__ import annotations

import glob
from dataclasses import MISSING, asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


def _as_int(name: str, x: Any) -> int:
    """Inteiro a partir de int, float integral ou string (`"35e9"`); ValueError se fracionário."""
    if isinstance(x, bool):
        raise ValueError(f"{name}: esperado inteiro, recebeu {x!r}")
    if isinstance(x, int):
        return x
    try:
        value = float(x)
    except (TypeError, ValueError):
        raise ValueError(f"{name}: esperado inteiro, recebeu {x!r}") from None
    if not value.is_integer():
        raise ValueError(f"{name}: esperado inteiro, recebeu {x!r}")
    return int(value)


def _as_float(name: str, x: Any) -> float:
    if isinstance(x, bool):
        raise ValueError(f"{name}: esperado número, recebeu {x!r}")
    try:
        return float(x)
    except (TypeError, ValueError):
        raise ValueError(f"{name}: esperado número, recebeu {x!r}") from None


@dataclass
class Stage:
    """Estágio do currículo de contexto: vale enquanto `tokens_consumed < until_tokens`."""

    until_tokens: int
    seq_len: int
    micro_batch: int

    def __post_init__(self) -> None:
        for f in fields(self):
            setattr(self, f.name, _as_int(f"stages.{f.name}", getattr(self, f.name)))
        if self.seq_len < 1 or self.micro_batch < 1:
            raise ValueError(f"estágio com seq_len/micro_batch < 1: {self}")


_INT_FIELDS = (
    "eos_id", "grad_accum", "total_tokens", "warmup_steps", "log_every", "ckpt_every_tokens",
    "snapshot_every_tokens", "keep_snapshots", "seed",
)
_FLOAT_FIELDS = (
    "lr_muon", "lr_adamw", "decay_frac", "weight_decay", "grad_clip", "mtp_weight_end", "spike_threshold",
    "peak_tflops",
)


@dataclass(kw_only=True)
class TrainConfig:
    # Modelo e dados (`data_paths` são padrões glob, expandidos por `resolve_data_paths`)
    model_preset: str
    model_overrides: dict = field(default_factory=dict)
    data_paths: list[str]
    eos_id: int

    # Batch e estágios
    grad_accum: int
    stages: list[Stage]
    total_tokens: int

    # Otimização (com match_rms_adamw o lr do Muon tem a escala do AdamW: os dois começam iguais)
    lr_muon: float = 3e-3
    lr_adamw: float = 3e-3
    warmup_steps: int = 500
    decay_frac: float = 0.2
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    # λ do MTP no loss do passo (R23): linear de `mtp_weight_start` (None = o `mtp_weight` do
    # modelo) no passo 0 até `mtp_weight_end` no último passo
    mtp_weight_start: float | None = None
    mtp_weight_end: float = 0.1
    # alerta de spike: loss do passo > EMA (beta 0.98, depois do warmup) × (1 + spike_threshold)
    spike_threshold: float = 0.3

    # Execução (`out_dir` é o diretório do run: metrics.jsonl, last.pt e os snapshots ficam nele)
    compile: bool = False
    log_every: int = 10
    ckpt_every_tokens: int
    snapshot_every_tokens: int = 0  # snapshot_step{passo}.pt numerado a cada N tokens; 0 = desligado
    keep_snapshots: int = 3  # só os N snapshots mais novos ficam no disco
    out_dir: str
    seed: int = 0
    device: str = "auto"  # "auto" -> "cuda" se disponível, senão "cpu"
    peak_tflops: float = 165.0  # pico BF16 denso da GPU (4090) para o MFU do metrics.jsonl

    def __post_init__(self) -> None:
        for name in _INT_FIELDS:
            setattr(self, name, _as_int(name, getattr(self, name)))
        for name in _FLOAT_FIELDS:
            setattr(self, name, _as_float(name, getattr(self, name)))
        if self.mtp_weight_start is not None:
            self.mtp_weight_start = _as_float("mtp_weight_start", self.mtp_weight_start)
        self.stages = [s if isinstance(s, Stage) else Stage(**s) for s in self.stages]
        self._validate()

    def _validate(self) -> None:
        if not self.stages:
            raise ValueError("é preciso ao menos um estágio em `stages`")
        if self.grad_accum < 1 or self.log_every < 1 or self.ckpt_every_tokens < 1:
            raise ValueError("grad_accum, log_every e ckpt_every_tokens devem ser >= 1")
        if self.peak_tflops <= 0:
            raise ValueError(f"peak_tflops deve ser > 0, recebeu {self.peak_tflops}")
        if self.snapshot_every_tokens < 0:
            raise ValueError(f"snapshot_every_tokens deve ser >= 0 (0 desliga), recebeu {self.snapshot_every_tokens}")
        if self.keep_snapshots < 1:
            raise ValueError(f"keep_snapshots deve ser >= 1, recebeu {self.keep_snapshots}")
        if self.spike_threshold <= 0:
            raise ValueError(f"spike_threshold deve ser > 0, recebeu {self.spike_threshold}")
        for name in ("mtp_weight_start", "mtp_weight_end"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} deve ser >= 0, recebeu {value}")
        per_micro = {s.seq_len * s.micro_batch for s in self.stages}
        if len(per_micro) != 1:
            # tokens por passo constantes => o schedule por passo (WSD) é exato entre estágios
            raise ValueError(f"seq_len * micro_batch deve ser igual em todos os estágios, recebeu {sorted(per_micro)}")
        longest = max(s.seq_len for s in self.stages)
        if any(longest % s.seq_len for s in self.stages):
            # o stream corta o corpus em unidades do maior seq_len (R22)
            raise ValueError(
                f"todo seq_len deve dividir o maior ({longest}), recebeu {[s.seq_len for s in self.stages]}"
            )
        until = [s.until_tokens for s in self.stages]
        if any(b <= a for a, b in zip(until, until[1:])):
            raise ValueError(f"until_tokens dos estágios deve ser estritamente crescente, recebeu {until}")
        if until[-1] < self.total_tokens:
            raise ValueError(f"o último estágio (until_tokens={until[-1]}) não cobre total_tokens={self.total_tokens}")

    @property
    def tokens_per_step(self) -> int:
        s = self.stages[0]
        return s.seq_len * s.micro_batch * self.grad_accum

    @property
    def total_steps(self) -> int:
        return self.total_tokens // self.tokens_per_step

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Aplica `overrides` sobre `base`: dict com dict faz merge recursivo, o resto substitui.

    O merge é o que deixa `model_overrides.fused_ce=true` somar ao `model_overrides` do YAML em vez
    de apagá-lo; listas (como `stages`) são substituídas inteiras.
    """
    out = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_train_config(path: Path, overrides: dict[str, Any] | None = None) -> TrainConfig:
    """Lê o YAML, aplica `overrides` (de `parse_overrides`) e valida.

    As overrides entram no dict cru, antes da validação: chave desconhecida continua dando
    ValueError. ValueError também em estágios inconsistentes.
    """
    raw = yaml.safe_load(Path(path).read_text()) or {}
    if overrides:
        raw = _merge(raw, overrides)
    known = {f.name for f in fields(TrainConfig)}
    required = {f.name for f in fields(TrainConfig) if f.default is MISSING and f.default_factory is MISSING}
    unknown, missing = set(raw) - known, required - set(raw)
    if unknown:
        raise ValueError(f"chaves desconhecidas em {path}: {sorted(unknown)}")
    if missing:
        raise ValueError(f"chaves obrigatórias ausentes em {path}: {sorted(missing)}")
    return TrainConfig(**raw)


def _parse_value(text: str) -> Any:
    """`yaml.safe_load` do valor; string que o `float()` aceita como literal numérico vira float.

    O PyYAML (YAML 1.1) devolve `3e-3` e `35e9` como string; aqui eles viram número, como nos
    campos do `TrainConfig`. "nan"/"inf"/"infinity" (qualquer caixa) continuam strings: `float()`
    os aceita, mas ninguém quer um override numérico que seja NaN sem querer.
    """
    value = yaml.safe_load(text)
    if isinstance(value, str) and value.strip().lower().lstrip("+-") not in ("nan", "inf", "infinity"):
        try:
            return float(value)
        except ValueError:
            pass
    return value


def parse_overrides(pairs: list[str]) -> dict[str, Any]:
    """`["a=1", "b.c=true"]` -> `{"a": 1, "b": {"c": True}}` (flags `--override KEY=VALUE`).

    O valor é lido como YAML (`true`, `4`, `[1, 2]`, `null`); chave com ponto vira dict aninhado,
    o formato do `model_overrides`. Par sem `=` levanta ValueError.
    """
    out: dict[str, Any] = {}
    for pair in pairs:
        key, sep, text = pair.partition("=")
        if not sep or not key:
            raise ValueError(f"override inválido {pair!r}: esperado KEY=VALUE")
        node = out
        *parents, leaf = key.split(".")
        for part in parents:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(f"override {pair!r} conflita com um valor escalar já dado para {part!r}")
        node[leaf] = _parse_value(text)
    return out


def resolve_data_paths(patterns: list[str]) -> list[Path]:
    """Expande cada padrão com `sorted(glob)` (ordem determinística dos shards), na ordem dada."""
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"o padrão de dados {pattern!r} não casou com nada")
        paths.extend(Path(m) for m in matches)
    return paths
