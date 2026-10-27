"""Testes unitários para o gravador de shards binários uint32 (BinarySharder)."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from sore.data.sharder import BinarySharder


def test_binary_sharder_exact_blocks(tmp_path: Path):
    """Verifica gravação de blocos exatos e metadados básicos."""
    # Shard miniatura de 32 tokens com unit_len=8 (4 unidades inteiras)
    sharder = BinarySharder(tmp_path, shard_size=32, eos_id=999, unit_len=8)
    # Adiciona documentos totalizando 32 tokens
    doc1 = [1, 2, 3]  # vira [1, 2, 3, 999] (4 tokens)
    doc2 = [4] * 27  # vira [4]*27 + [999] (28 tokens) -> fecha 32
    closed = sharder.add_document(doc1)
    assert len(closed) == 0
    closed = sharder.add_document(doc2)
    assert len(closed) == 1

    shard_bin = closed[0]
    shard_json = shard_bin.with_suffix(".json")
    assert shard_bin.exists() and shard_json.exists()
    assert shard_bin.stat().st_size == 32 * 4  # 32 uint32 = 128 bytes

    meta = json.loads(shard_json.read_text(encoding="utf-8"))
    assert meta["n_tokens"] == 32
    assert meta["dtype"] == "uint32"
    assert meta["eos_id"] == 999
    assert meta["unit_len"] == 8

    # Leitura como uint32
    tokens = np.fromfile(shard_bin, dtype=np.uint32)
    assert len(tokens) == 32
    assert tokens[3] == 999
    assert tokens[-1] == 999


def test_binary_sharder_multi_document_spillover(tmp_path: Path):
    """Verifica transbordamento de documentos entre múltiplos shards sem perda de tokens."""
    sharder = BinarySharder(tmp_path, shard_size=16, eos_id=100, unit_len=4)

    # doc1: 10 tokens + 1 eos = 11 tokens
    doc1 = list(range(1, 11))
    # doc2: 9 tokens + 1 eos = 10 tokens (11 + 10 = 21 tokens: fecha shard 0 com 16, sobram 5 no shard 1)
    doc2 = list(range(11, 20))

    closed1 = sharder.add_document(doc1, source_name="srcA")
    assert len(closed1) == 0

    closed2 = sharder.add_document(doc2, source_name="srcB")
    assert len(closed2) == 1
    assert closed2[0].name == "shard_00000.bin"

    # Fecha o sharder para descarregar o shard parcial restante
    final_shards = sharder.close()
    assert len(final_shards) == 2
    assert final_shards[0].name == "shard_00000.bin"
    assert final_shards[1].name == "shard_00001.bin"

    # Validar conteúdo reconstruído
    shard0_tokens = np.fromfile(final_shards[0], dtype=np.uint32)
    shard1_tokens = np.fromfile(final_shards[1], dtype=np.uint32)
    assert len(shard0_tokens) == 16
    assert len(shard1_tokens) == 5

    all_tokens = np.concatenate([shard0_tokens, shard1_tokens])
    expected = doc1 + [100] + doc2 + [100]
    assert np.array_equal(all_tokens, expected)

    # Validar distribuição de fontes no JSON sidecar
    meta0 = json.loads(final_shards[0].with_suffix(".json").read_text(encoding="utf-8"))
    meta1 = json.loads(final_shards[1].with_suffix(".json").read_text(encoding="utf-8"))

    assert meta0["sources"] == {"srcA": 11, "srcB": 5}
    assert meta1["sources"] == {"srcB": 5}
    assert meta0["n_tokens"] == 16
    assert meta1["n_tokens"] == 5


def test_binary_sharder_huge_document_spanning_multiple_shards(tmp_path: Path):
    """Verifica que um único documento maior que múltiplos shards preenche e fecha sucessivamente."""
    sharder = BinarySharder(tmp_path, shard_size=10, eos_id=99, unit_len=2)

    # Documento com 29 tokens + 1 eos = 30 tokens -> exatamente 3 shards de 10
    doc = list(range(29))
    closed = sharder.add_document(doc, source_name="big_doc")
    assert len(closed) == 3
    assert [p.name for p in closed] == ["shard_00000.bin", "shard_00001.bin", "shard_00002.bin"]

    for i, p in enumerate(closed):
        assert p.stat().st_size == 10 * 4
        tokens = np.fromfile(p, dtype=np.uint32)
        assert len(tokens) == 10

    # Fechar não deve criar shard vazio
    all_shards = sharder.close()
    assert len(all_shards) == 3


def test_binary_sharder_sidecar_metadata_and_sha256(tmp_path: Path):
    """Verifica cálculo correto do sha256 no sidecar comparado com o hash real do arquivo."""
    sharder = BinarySharder(tmp_path, shard_size=8, eos_id=77, unit_len=4)
    sharder.add_document([1, 2, 3], source_name="test_sha")  # 4 tokens
    sharder.add_document([4, 5, 6], source_name="test_sha")  # 4 tokens -> fecha 8

    shards = sharder.close()
    assert len(shards) == 1

    bin_path = shards[0]
    json_path = bin_path.with_suffix(".json")

    bin_bytes = bin_path.read_bytes()
    expected_sha256 = hashlib.sha256(bin_bytes).hexdigest()

    meta = json.loads(json_path.read_text(encoding="utf-8"))
    assert meta["sha256"] == expected_sha256
    assert meta["n_tokens"] == 8
    assert meta["dtype"] == "uint32"
    assert meta["sources"] == {"test_sha": 8}


def test_binary_sharder_validation_and_errors(tmp_path: Path):
    """Verifica validações de invariantes de inicialização e estado."""
    # shard_size não divisível por unit_len
    with pytest.raises(ValueError, match="divisible by unit_len"):
        BinarySharder(tmp_path, shard_size=10, unit_len=3)

    # valores não positivos
    with pytest.raises(ValueError, match="shard_size must be positive"):
        BinarySharder(tmp_path, shard_size=0, unit_len=8)

    with pytest.raises(ValueError, match="unit_len must be positive"):
        BinarySharder(tmp_path, shard_size=32, unit_len=0)

    with pytest.raises(ValueError, match="eos_id must be non-negative"):
        BinarySharder(tmp_path, shard_size=32, eos_id=-1, unit_len=8)

    # Operação após fechamento
    sharder = BinarySharder(tmp_path, shard_size=16, eos_id=1, unit_len=4)
    sharder.close()
    with pytest.raises(RuntimeError, match="closed"):
        sharder.add_document([1, 2, 3])


def test_binary_sharder_empty_documents_and_empty_close(tmp_path: Path):
    """Verifica comportamento com documento vazio e close() sem dados."""
    sharder = BinarySharder(tmp_path, shard_size=8, eos_id=5, unit_len=4)
    # close sem adicionar nada retorna lista vazia
    assert sharder.close() == []

    # Sharder com documento vazio adiciona apenas o eos_id
    sharder2 = BinarySharder(tmp_path / "sub", shard_size=8, eos_id=5, unit_len=4)
    sharder2.add_document([])  # apenas [5]
    shards = sharder2.close()
    assert len(shards) == 1
    tokens = np.fromfile(shards[0], dtype=np.uint32)
    assert len(tokens) == 1
    assert tokens[0] == 5


def test_binary_sharder_context_manager(tmp_path: Path):
    """Verifica uso idiomático com bloco with."""
    with BinarySharder(tmp_path, shard_size=8, eos_id=99, unit_len=4) as sharder:
        sharder.add_document([1, 2, 3])

    # Ao sair do with, shard_00000.bin deve existir e estar fechado
    shard_bin = tmp_path / "shard_00000.bin"
    assert shard_bin.exists()
    assert len(sharder.written_shards) == 1
