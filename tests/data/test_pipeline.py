"""Testes unitários e de integração para o pipeline de dados em streaming e CLI."""

import json
from pathlib import Path
import numpy as np
import pytest

from sore.data.pipeline import WeightedStreamIterator, run_pipeline


class MockTokenizer:
    """Tokenizador simplificado para testes rápidos sem download de modelos."""

    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size

    def encode(self, text: str) -> list[int]:
        words = text.split()
        return [abs(hash(w)) % (self.vocab_size - 1) for w in words]


class MockHFTokenizer:
    """Mock simulando interface Hugging Face com atributo .ids."""

    class Encoding:
        def __init__(self, ids: list[int]):
            self.ids = ids

    def encode(self, text: str) -> Encoding:
        words = text.split()
        return self.Encoding([abs(hash(w)) % 50000 for w in words])


def test_weighted_stream_multiplexer():
    src1 = iter([{"text": f"doc1_{i}"} for i in range(20)])
    src2 = iter([{"text": f"doc2_{i}"} for i in range(20)])
    mux = WeightedStreamIterator({"s1": src1, "s2": src2}, {"s1": 0.5, "s2": 0.5}, seed=42)

    items = [next(mux)["text"] for _ in range(20)]
    assert any("doc1" in x for x in items)
    assert any("doc2" in x for x in items)


def test_weighted_stream_iterator_exhaustion():
    # Fonte curta de 2 elementos e fonte longa de 10 elementos
    src_short = iter([{"text": f"short_{i}"} for i in range(2)])
    src_long = iter([{"text": f"long_{i}"} for i in range(10)])

    mux = WeightedStreamIterator({"short": src_short, "long": src_long}, {"short": 0.5, "long": 0.5}, seed=42)
    collected = [item["text"] for item in mux]

    assert len(collected) == 12
    assert sum(1 for x in collected if "short" in x) == 2
    assert sum(1 for x in collected if "long" in x) == 10


def test_weighted_stream_iterator_invalid_weights():
    with pytest.raises(ValueError):
        WeightedStreamIterator({"s1": iter([])}, {"s1": -1.0})

    with pytest.raises(ValueError):
        WeightedStreamIterator({}, {})


def test_run_pipeline_integration(tmp_path: Path):
    valid_pt_doc_1 = (
        "A inteligência artificial tem avançado rapidamente nos últimos anos em todo o mundo. "
        "Pesquisadores desenvolvem novos modelos de linguagem com arquiteturas eficientes e compactas. "
        "O objetivo é permitir que sistemas inteligentes sejam executados em dispositivos móveis sem perda "
        "de qualidade ou acurácia. A engenharia de software moderna exige soluções práticas e robustas."
    )
    # Duplicata quase idêntica de doc 1
    valid_pt_doc_dup = (
        "A inteligência artificial tem avançado rapidamente nos últimos anos em todo o mundo. "
        "Pesquisadores desenvolvem novos modelos de linguagem com arquiteturas eficientes e compactas. "
        "O objetivo é permitir que sistemas inteligentes sejam executados em dispositivos móveis sem perda "
        "de qualidade ou acurácia. A engenharia de software moderna exige soluções práticas e robustas!"
    )
    bad_short_doc = "Olá mundo, este é um teste muito curto."
    valid_en_doc = (
        "Deep learning and recurrent neural networks represent an important foundation for sequence modeling. "
        "Modern architectures combine state space representations with linear attention mechanisms to achieve "
        "high throughput and low latency on parallel computing hardware. These methods scale effectively across "
        "billions of tokens during language pretraining."
    )

    sources = {
        "pt_web": [
            {"text": valid_pt_doc_1},
            {"text": bad_short_doc},
            {"text": valid_pt_doc_dup},
        ],
        "en_edu": [
            {"text": valid_en_doc},
        ],
    }
    weights = {"pt_web": 0.7, "en_edu": 0.3}
    tokenizer = MockHFTokenizer()

    out_dir = tmp_path / "shards"
    shard_size = 64  # Shard pequeno para teste
    unit_len = 8

    shards = run_pipeline(
        sources=sources,
        weights=weights,
        tokenizer=tokenizer,
        out_dir=out_dir,
        max_tokens=200,
        shard_size=shard_size,
        unit_len=unit_len,
        eos_id=98303,
        seed=42,
    )

    assert len(shards) >= 1
    for s in shards:
        assert s.exists()
        sidecar = s.with_suffix(".json")
        assert sidecar.exists()

        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        assert meta["eos_id"] == 98303
        assert meta["unit_len"] == unit_len
        assert "sha256" in meta
        assert "sources" in meta
        assert meta["n_tokens"] <= shard_size

        # Verifica conteúdo binário
        tokens = np.fromfile(s, dtype=np.uint32)
        assert len(tokens) == meta["n_tokens"]
        assert 98303 in tokens


def test_run_pipeline_stats_and_limits(tmp_path: Path):
    doc_pt = (
        "A inteligência artificial tem avançado rapidamente nos últimos anos em todo o mundo. "
        "Pesquisadores desenvolvem novos modelos de linguagem com arquiteturas eficientes e compactas. "
        "O objetivo é permitir que sistemas inteligentes sejam executados em dispositivos móveis sem perda "
        "de qualidade ou acurácia. A engenharia de software moderna exige soluções práticas e robustas."
    )
    sources = {
        "pt": [{"text": doc_pt} for _ in range(10)]
    }
    tokenizer = MockTokenizer()

    shards, stats = run_pipeline(
        sources=sources,
        weights={"pt": 1.0},
        tokenizer=tokenizer,
        out_dir=tmp_path / "stats_shards",
        max_tokens=1000,
        shard_size=128,
        unit_len=8,
        eos_id=98303,
        return_stats=True,
    )

    assert stats["documents_seen"] > 0
    # Como todos os documentos são idênticos, apenas o primeiro deve ser aceito e os seguintes deduplicados
    assert stats["documents_accepted"] == 1
    assert stats["documents_duplicated"] >= 1
    assert stats["tokens_written"] > 0


def test_prepare_data_cli(tmp_path: Path):
    import subprocess
    import sys

    out_dir = tmp_path / "cli_shards"
    cmd = [
        sys.executable,
        "scripts/prepare_data.py",
        "--mock",
        "--output-dir",
        str(out_dir),
        "--max-tokens",
        "200",
        "--shard-size",
        "64",
        "--unit-len",
        "8",
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"CLI falhou: {res.stderr}\n{res.stdout}"

    bin_files = list(out_dir.glob("shard_*.bin"))
    assert len(bin_files) >= 1
    json_files = list(out_dir.glob("shard_*.json"))
    assert len(json_files) == len(bin_files)
