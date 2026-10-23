"""Testes dos shards uint32 e do stream empacotado retomável (`PackedTokenStream`)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from sore.train.data import Batch, PackedTokenStream, write_token_shard

REPO = Path(__file__).resolve().parents[2]


def _shard(tmp_path: Path, tokens, name: str = "s0.bin") -> Path:
    path = tmp_path / name
    write_token_shard(path, np.asarray(tokens))
    return path


def _reference_cu_seqlens(input_ids: torch.Tensor, seq_len: int, eos_id: int) -> list[int]:
    """Implementação ingênua (laço em Python) da regra de fronteiras, só para comparar."""
    ids = input_ids[0].tolist()
    n = len(ids)
    cuts = {0, n}
    cuts.update(range(seq_len, n, seq_len))
    cuts.update(t + 1 for t, tok in enumerate(ids) if tok == eos_id)
    return sorted(cuts)


def _periodic_shard(tmp_path: Path, n_tokens: int, period: int = 7) -> Path:
    """tokens[i] = i % period: com eos_id=0 há um EOS a cada `period` tokens."""
    return _shard(tmp_path, np.arange(n_tokens) % period)


def test_write_token_shard_roundtrip_and_sidecar(tmp_path):
    path = tmp_path / "a.bin"
    tokens = np.array([0, 1, 4_000_000_000, 7], dtype=np.int64)
    write_token_shard(path, tokens)
    raw = np.fromfile(path, dtype="<u4")
    assert raw.tolist() == tokens.tolist()
    assert path.stat().st_size == 4 * len(tokens)
    assert json.loads(path.with_suffix(".json").read_text()) == {"n_tokens": 4, "dtype": "uint32"}


def test_invalid_sidecar_raises(tmp_path):
    path = _periodic_shard(tmp_path, 100)
    sidecar = path.with_suffix(".json")

    sidecar.write_text(json.dumps({"n_tokens": 99, "dtype": "uint32"}))
    with pytest.raises(ValueError, match="n_tokens"):
        PackedTokenStream([path], seq_len=8, micro_batch=1, eos_id=0, seed=0)

    sidecar.write_text(json.dumps({"n_tokens": 100, "dtype": "uint16"}))
    with pytest.raises(ValueError, match="dtype"):
        PackedTokenStream([path], seq_len=8, micro_batch=1, eos_id=0, seed=0)


def test_cu_seqlens_handcrafted(tmp_path):
    # seq_len=8, 17 tokens -> 2 janelas ([0:9] e [8:17]) e micro_batch=2 usa as duas.
    # EOS nas posições 2, 6, 10 e 14 => nas entradas das janelas, offsets 2 e 6 em ambas as
    # linhas; o resultado independe da ordem da permutação.
    tokens = [1, 2, 0, 3, 4, 5, 0, 6, 7, 8, 0, 9, 1, 2, 0, 3, 4]
    path = _shard(tmp_path, tokens)
    stream = PackedTokenStream([path], seq_len=8, micro_batch=2, eos_id=0, seed=0)
    batch = next(stream)
    assert isinstance(batch, Batch)
    assert batch.input_ids.shape == (1, 16)
    assert batch.cu_seqlens.dtype == torch.long
    assert batch.cu_seqlens.tolist() == [0, 3, 7, 8, 11, 15, 16]


def test_labels_masked_at_eos(tmp_path):
    # tokens[i] = i % 7: o alvo de t é (input[t] + 1) % 7, exceto onde a entrada é EOS.
    path = _periodic_shard(tmp_path, 200)
    stream = PackedTokenStream([path], seq_len=10, micro_batch=3, eos_id=0, seed=1)
    for _ in range(4):
        batch = next(stream)
        assert batch.labels.shape == batch.input_ids.shape == (1, 30)
        assert batch.labels.dtype == batch.input_ids.dtype == torch.long
        is_eos = batch.input_ids == 0
        assert is_eos.any()
        assert (batch.labels[is_eos] == -100).all()
        expected = (batch.input_ids + 1) % 7
        torch.testing.assert_close(batch.labels[~is_eos], expected[~is_eos])


def test_consecutive_eos_no_empty_segments(tmp_path):
    # seq_len=8: EOS consecutivos no meio e no início de uma linha, e EOS na última posição de
    # cada linha (t+1 coincide com a fronteira de linha e, na última, com N).
    tokens = [0, 0, 5, 0, 0, 0, 6, 0, 0, 0, 0, 3, 4, 5, 6, 0, 1, 2]
    path = _shard(tmp_path, tokens)
    stream = PackedTokenStream([path], seq_len=8, micro_batch=2, eos_id=0, seed=3)
    for _ in range(5):
        batch = next(stream)
        cu = batch.cu_seqlens
        assert (cu[1:] - cu[:-1] >= 1).all()
        assert cu[0] == 0 and cu[-1] == batch.input_ids.shape[1]
        assert cu.tolist() == _reference_cu_seqlens(batch.input_ids, 8, 0)


def test_cu_seqlens_random_data_is_valid(tmp_path):
    rng = np.random.default_rng(0)
    path = _shard(tmp_path, rng.integers(0, 5, size=5000))  # eos_id=0 -> ~20% EOS
    stream = PackedTokenStream([path], seq_len=32, micro_batch=4, eos_id=0, seed=5)
    n = 4 * 32
    for _ in range(10):
        batch = next(stream)
        cu = batch.cu_seqlens
        assert cu.dtype == torch.long and cu.device.type == "cpu" and cu.dim() == 1
        assert cu[0].item() == 0
        assert cu[-1].item() == n == batch.input_ids.shape[1]
        assert (torch.diff(cu) > 0).all()
        assert cu.tolist() == _reference_cu_seqlens(batch.input_ids, 32, 0)


def test_resume_determinism(tmp_path):
    path = _periodic_shard(tmp_path, 3000)
    stream = PackedTokenStream([path], seq_len=16, micro_batch=2, eos_id=0, seed=7)
    for _ in range(3):
        next(stream)
    state = stream.state_dict()
    assert all(isinstance(v, int) for v in state.values())
    expected = next(stream)

    resumed = PackedTokenStream([path], seq_len=16, micro_batch=2, eos_id=0, seed=7)
    resumed.load_state_dict(state)
    got = next(resumed)
    assert torch.equal(got.input_ids, expected.input_ids)
    assert torch.equal(got.labels, expected.labels)
    assert torch.equal(got.cu_seqlens, expected.cu_seqlens)
    assert resumed.tokens_consumed == stream.tokens_consumed == 4 * 2 * 16


def test_resume_across_epoch_rollover(tmp_path):
    # 105 tokens, seq_len=8 -> 13 janelas; micro_batch=4 -> 3 batches por época (sobra 1 descartada).
    path = _periodic_shard(tmp_path, 105)
    stream = PackedTokenStream([path], seq_len=8, micro_batch=4, eos_id=0, seed=11)
    assert stream.num_windows == 13
    for _ in range(3):
        next(stream)
    assert stream.state_dict()["epoch"] == 0
    state = stream.state_dict()
    tail = [next(stream) for _ in range(4)]  # 3 batches da época 1 e o primeiro da época 2
    assert stream.state_dict()["epoch"] == 2

    resumed = PackedTokenStream([path], seq_len=8, micro_batch=4, eos_id=0, seed=11)
    resumed.load_state_dict(state)
    for want in tail:
        got = next(resumed)
        assert torch.equal(got.input_ids, want.input_ids)
        assert torch.equal(got.cu_seqlens, want.cu_seqlens)


def test_epoch_is_a_permutation_without_repeats(tmp_path):
    # 41 tokens (valores 1..41), seq_len=8 -> 5 janelas: cada época usa cada janela uma vez.
    path = _shard(tmp_path, np.arange(1, 42))
    stream = PackedTokenStream([path], seq_len=8, micro_batch=1, eos_id=0, seed=2)
    assert stream.num_windows == 5
    orders = []
    for _ in range(2):
        orders.append([next(stream).input_ids[0, 0].item() for _ in range(5)])
        assert sorted(orders[-1]) == [1, 9, 17, 25, 33]
    assert orders[0] != orders[1]  # a permutação muda de uma época para a outra


def test_seed_controls_order(tmp_path):
    path = _periodic_shard(tmp_path, 3000)

    def first_batches(seed):
        s = PackedTokenStream([path], seq_len=16, micro_batch=2, eos_id=0, seed=seed)
        return torch.cat([next(s).input_ids for _ in range(5)], dim=1)

    assert torch.equal(first_batches(0), first_batches(0))
    assert not torch.equal(first_batches(0), first_batches(1))


def test_set_shape_changes_shape(tmp_path):
    path = _periodic_shard(tmp_path, 4000)
    stream = PackedTokenStream([path], seq_len=16, micro_batch=2, eos_id=0, seed=0, unit_len=32)
    first = next(stream)
    assert first.input_ids.shape == (1, 32)
    assert stream.tokens_consumed == 32

    stream.set_shape(seq_len=32, micro_batch=4)
    state = stream.state_dict()
    # as 2 janelas de 16 consumiram a 1a unidade de 32: o novo formato começa na 2a
    assert (state["seq_len"], state["micro_batch"], state["position"], state["unit_len"]) == (32, 4, 1, 32)
    second = next(stream)
    assert second.input_ids.shape == second.labels.shape == (1, 128)
    assert second.cu_seqlens[-1].item() == 128
    assert stream.tokens_consumed == 32 + 128

    # o estado do novo formato também é retomável
    resumed = PackedTokenStream([path], seq_len=16, micro_batch=2, eos_id=0, seed=0, unit_len=32)
    resumed.load_state_dict(stream.state_dict())
    assert torch.equal(next(resumed).input_ids, next(stream).input_ids)


# R22: 97 tokens com valores 1..97 (sem EOS) -> 6 unidades de 16; o 1º token de cada linha
# identifica a janela (offset no shard = valor - 1)
UNIT, N_UNITS = 16, 6


def _unit_stream(tmp_path, seq_len, micro_batch, seed=0):
    path = tmp_path / "units.bin"
    if not path.exists():
        write_token_shard(path, np.arange(1, N_UNITS * UNIT + 2))
    return PackedTokenStream([path], seq_len, micro_batch, eos_id=0, seed=seed, unit_len=UNIT)


def _window_starts(batch, seq_len):
    return (batch.input_ids[0, ::seq_len] - 1).tolist()


def test_windows_are_consecutive_within_a_unit(tmp_path):
    stream = _unit_stream(tmp_path, seq_len=4, micro_batch=2)
    assert (stream.num_units, stream.num_windows) == (N_UNITS, N_UNITS * 4)
    starts = [s for _ in range(12) for s in _window_starts(next(stream), 4)]  # uma época inteira
    assert stream.epoch == 0
    # as 4 janelas de cada unidade saem em sequência e cobrem a unidade inteira, uma vez
    units = [starts[i : i + 4] for i in range(0, 24, 4)]
    assert all(u == [u[0], u[0] + 4, u[0] + 8, u[0] + 12] and u[0] % UNIT == 0 for u in units)
    assert sorted(u[0] for u in units) == [UNIT * k for k in range(N_UNITS)]
    # a ordem das unidades não depende do seq_len: é a mesma de um stream com seq_len = unit_len
    whole = _unit_stream(tmp_path, seq_len=UNIT, micro_batch=1)
    assert [u[0] for u in units] == [_window_starts(next(whole), UNIT)[0] for _ in range(N_UNITS)]


def test_stage_switch_continues_the_epoch_without_repeats(tmp_path):
    stream = _unit_stream(tmp_path, seq_len=4, micro_batch=3)
    before = _window_starts(next(stream), 4)  # 3 das 4 janelas da 1a unidade
    u0 = before[0]
    assert before == [u0, u0 + 4, u0 + 8]

    stream.set_shape(seq_len=UNIT, micro_batch=1)
    assert (stream.epoch, stream.position) == (0, 1)  # pula o resto da unidade começada
    after = [_window_starts(next(stream), UNIT)[0] for _ in range(N_UNITS - 1)]
    assert stream.epoch == 0  # a mesma época continua, sem recomeçar a permutação
    # nenhuma janela repetida: as 5 unidades restantes, cada uma uma vez, nenhuma é a começada
    assert u0 not in after and sorted(after + [u0]) == [UNIT * k for k in range(N_UNITS)]
    # é a continuação da permutação da época: a ordem de um stream que começou com seq_len = unit_len
    whole = _unit_stream(tmp_path, seq_len=UNIT, micro_batch=1)
    assert [u0, *after] == [_window_starts(next(whole), UNIT)[0] for _ in range(N_UNITS)]
    next(stream)
    assert stream.epoch == 1  # só depois de esgotar as unidades


def test_resume_after_set_shape(tmp_path):
    stream = _unit_stream(tmp_path, seq_len=4, micro_batch=3)
    next(stream)
    stream.set_shape(seq_len=8, micro_batch=2)
    next(stream)
    state = stream.state_dict()
    expected = [next(stream) for _ in range(8)]  # atravessa a virada de época

    resumed = _unit_stream(tmp_path, seq_len=4, micro_batch=3)
    resumed.load_state_dict(state)
    for want in expected:
        got = next(resumed)
        assert torch.equal(got.input_ids, want.input_ids)
        assert torch.equal(got.cu_seqlens, want.cu_seqlens)
    assert resumed.state_dict() == stream.state_dict()


def test_unit_len_validation(tmp_path):
    with pytest.raises(ValueError, match="unit_len"):
        _unit_stream(tmp_path, seq_len=6, micro_batch=1)  # 16 % 6 != 0
    stream = _unit_stream(tmp_path, seq_len=4, micro_batch=1)
    with pytest.raises(ValueError, match="unit_len"):
        stream.set_shape(seq_len=32, micro_batch=1)
    assert next(stream).input_ids.shape == (1, 4)  # a falha não altera o stream
    state = stream.state_dict()
    other = PackedTokenStream(
        [tmp_path / "units.bin"], seq_len=4, micro_batch=1, eos_id=0, seed=0, unit_len=2 * UNIT
    )
    with pytest.raises(ValueError, match="unit_len"):
        other.load_state_dict(state)


def test_short_shard_contributes_no_units(tmp_path):
    # unit_len=16: um shard de 16 tokens (< unit_len + 1) não tem unidade; um de 40 tem 2
    short = _shard(tmp_path, np.arange(1, 17), "short.bin")
    long = _shard(tmp_path, np.arange(1000, 1040), "long.bin")
    stream = PackedTokenStream([short, long], seq_len=8, micro_batch=1, eos_id=0, seed=0, unit_len=16)
    assert (stream.num_units, stream.num_windows) == (2, 4)
    starts = sorted(next(stream).input_ids[0, 0].item() for _ in range(4))
    assert starts == [1000, 1008, 1016, 1024]


def test_windows_never_cross_shard_boundaries(tmp_path):
    # shard A: 21 tokens (valores 1..21) -> 2 janelas de seq_len=8 ([0:9], [8:17]);
    # shard B: 30 tokens (valores 1000..1029) -> 3 janelas. Uma janela que misturasse os
    # shards teria valores < 1000 e >= 1000, ou saltos de valor.
    a = _shard(tmp_path, np.arange(1, 22), "a.bin")
    b = _shard(tmp_path, np.arange(1000, 1030), "b.bin")
    stream = PackedTokenStream([a, b], seq_len=8, micro_batch=1, eos_id=0, seed=4)
    assert stream.num_windows == 5
    firsts = []
    for _ in range(5):
        batch = next(stream)
        ids, labels = batch.input_ids[0], batch.labels[0]
        assert (torch.diff(ids) == 1).all()  # linha contígua no mesmo shard
        assert (labels == ids + 1).all()  # o alvo da última posição também é do mesmo shard
        firsts.append(ids[0].item())
    assert sorted(firsts) == [1, 9, 1000, 1008, 1016]


def test_too_few_windows_raises(tmp_path):
    path = _periodic_shard(tmp_path, 40)  # (40 - 1) // 8 = 4 janelas de seq_len=8
    with pytest.raises(ValueError, match="janelas"):
        PackedTokenStream([path], seq_len=8, micro_batch=5, eos_id=0, seed=0)
    # unit_len=16: 2 unidades = 4 janelas de 8, mas só 2 de 16
    stream = PackedTokenStream([path], seq_len=8, micro_batch=4, eos_id=0, seed=0, unit_len=16)
    with pytest.raises(ValueError, match="janelas"):
        stream.set_shape(seq_len=16, micro_batch=4)
    # a falha não deixa o stream num estado quebrado
    assert next(stream).input_ids.shape == (1, 32)


def test_load_state_dict_rejects_other_shard_set(tmp_path):
    p1 = _periodic_shard(tmp_path, 400)
    other = _shard(tmp_path, np.arange(500) % 7, "other.bin")
    state = PackedTokenStream([p1], seq_len=8, micro_batch=2, eos_id=0, seed=0).state_dict()
    with pytest.raises(ValueError, match="shards"):
        PackedTokenStream([other], seq_len=8, micro_batch=2, eos_id=0, seed=0).load_state_dict(state)


def test_make_synthetic_shard_cli(tmp_path):
    out = tmp_path / "synth.bin"
    subprocess.run(
        [sys.executable, str(REPO / "scripts" / "make_synthetic_shard.py"), "--out", str(out),
         "--n-tokens", "20000", "--vocab", "64", "--eos", "0", "--seed", "3"],
        check=True, cwd=tmp_path,
    )
    tokens = np.memmap(out, dtype="<u4", mode="r")
    assert tokens.shape == (20000,)
    assert json.loads(out.with_suffix(".json").read_text()) == {"n_tokens": 20000, "dtype": "uint32"}
    assert tokens[-1] == 0 and tokens.max() < 64
    ends = np.flatnonzero(tokens == 0)
    lengths = np.diff(np.concatenate([[-1], ends]))  # inclui o EOS de cada documento
    assert lengths[:-1].min() >= 9 and lengths.max() <= 601  # 8..600 tokens de conteúdo + EOS
    stream = PackedTokenStream([out], seq_len=128, micro_batch=2, eos_id=0, seed=0)
    batch = next(stream)
    assert batch.cu_seqlens[-1].item() == 256
