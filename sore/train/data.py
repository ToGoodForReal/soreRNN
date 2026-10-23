"""Shards de tokens uint32 e stream empacotado determinístico e retomável.

Formato em disco: `uint32` little-endian cru (`<u4`) mais um sidecar JSON
`{"n_tokens": int, "dtype": "uint32"}` ao lado (`path.with_suffix(".json")`).

O stream sorteia janelas de `seq_len + 1` tokens (entrada = `[:-1]`, alvo = `[1:]`) por uma
permutação determinística de `(seed, epoch, seq_len)` e empacota `micro_batch` janelas numa
única linha `[1, N]`, com `cu_seqlens` marcando fronteiras de documento (EOS) e de linha.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from sore.model.loss import IGNORE_INDEX


@dataclass
class Batch:
    """Micro-batch empacotado: `input_ids`/`labels` `[1, N]` e `cu_seqlens` `[S+1]` (long, CPU)."""

    input_ids: torch.Tensor
    labels: torch.Tensor
    cu_seqlens: torch.Tensor


def write_token_shard(path: Path, tokens: np.ndarray) -> None:
    """Grava `tokens` como uint32 little-endian cru e o sidecar `.json` ao lado."""
    path = Path(path)
    arr = np.asarray(tokens, dtype="<u4")
    arr.tofile(path)
    path.with_suffix(".json").write_text(json.dumps({"n_tokens": int(arr.size), "dtype": "uint32"}))


def _open_shard(path: Path) -> np.ndarray:
    """Abre o shard como memmap `<u4` somente leitura, validando o sidecar."""
    meta = json.loads(path.with_suffix(".json").read_text())
    if meta.get("dtype") != "uint32":
        raise ValueError(f"{path}: dtype do sidecar deve ser 'uint32', recebeu {meta.get('dtype')!r}")
    size = path.stat().st_size
    if size % 4 != 0 or meta.get("n_tokens") != size // 4:
        raise ValueError(
            f"{path}: n_tokens do sidecar ({meta.get('n_tokens')}) não confere com o arquivo "
            f"({size} bytes = {size / 4} tokens uint32)"
        )
    return np.memmap(path, dtype="<u4", mode="r")


class PackedTokenStream:
    """Stream infinito de `Batch` sobre shards, determinístico e retomável via `state_dict()`.

    Regras (ruling R14): janelas nunca cruzam shards. Um shard de tamanho L tem
    `(L - 1) // seq_len` janelas não sobrepostas; a janela w cobre os tokens
    `[w * seq_len, w * seq_len + seq_len + 1)` (janelas vizinhas dividem um token: o alvo da
    última entrada). A cada época as janelas globais são visitadas na ordem de
    `default_rng([seed, epoch, seq_len]).permutation(num_windows)`; quando restam menos que
    `micro_batch` janelas, a sobra é descartada e começa a época seguinte.
    """

    def __init__(self, paths: list[Path], seq_len: int, micro_batch: int, eos_id: int, seed: int):
        if not paths:
            raise ValueError("é preciso ao menos um shard")
        self._shards = [_open_shard(Path(p)) for p in paths]
        self._n_tokens = sum(len(s) for s in self._shards)
        self.eos_id = eos_id
        self.seed = seed
        self.epoch = 0
        self.position = 0
        self.tokens_consumed = 0
        self._apply_shape(seq_len, micro_batch)
        self._perm = self._permutation()

    # --- janelas e permutação -------------------------------------------------------------

    def _window_counts(self, seq_len: int, micro_batch: int) -> np.ndarray:
        """Janelas por shard para o formato dado; valida o formato (ValueError)."""
        if seq_len < 1 or micro_batch < 1:
            raise ValueError(f"seq_len e micro_batch devem ser >= 1, recebeu {seq_len}, {micro_batch}")
        counts = np.array([max(0, (len(s) - 1) // seq_len) for s in self._shards], dtype=np.int64)
        if counts.sum() < micro_batch:
            raise ValueError(
                f"{int(counts.sum())} janelas de seq_len={seq_len} nos shards, "
                f"insuficiente para micro_batch={micro_batch}"
            )
        return counts

    def _apply_shape(self, seq_len: int, micro_batch: int) -> None:
        counts = self._window_counts(seq_len, micro_batch)  # valida antes de mexer no estado
        self.seq_len = seq_len
        self.micro_batch = micro_batch
        self._cum = np.cumsum(counts)  # janelas acumuladas até o fim de cada shard
        self._first = self._cum - counts  # índice global da 1a janela de cada shard
        self.num_windows = int(self._cum[-1])

    def _permutation(self) -> np.ndarray:
        rng = np.random.default_rng([self.seed, self.epoch, self.seq_len])
        return rng.permutation(self.num_windows)

    # --- iteração ---------------------------------------------------------------------------

    def __iter__(self) -> PackedTokenStream:
        return self

    def __next__(self) -> Batch:
        if self.position + self.micro_batch > self.num_windows:  # descarta a sobra da época
            self.epoch += 1
            self.position = 0
            self._perm = self._permutation()
        windows = self._perm[self.position : self.position + self.micro_batch]
        self.position += self.micro_batch

        shard_ids = np.searchsorted(self._cum, windows, side="right")
        starts = (windows - self._first[shard_ids]) * self.seq_len
        rows = np.empty((self.micro_batch, self.seq_len + 1), dtype=np.int64)  # int64 só aqui
        for r, (shard, start) in enumerate(zip(shard_ids, starts)):
            rows[r] = self._shards[shard][start : start + self.seq_len + 1]

        n = self.micro_batch * self.seq_len
        ids = rows[:, :-1].reshape(1, n)
        labels = rows[:, 1:].reshape(1, n).copy()
        is_eos = ids == self.eos_id
        labels[is_eos] = IGNORE_INDEX

        # Fronteiras: 0, N, cada linha (múltiplos de seq_len) e o fim de cada documento (t+1).
        cuts = np.concatenate([np.arange(0, n + 1, self.seq_len), np.flatnonzero(is_eos[0]) + 1])
        cu_seqlens = np.unique(cuts)  # ordenado, sem duplicatas (EOS no fim de linha, EOS seguidos)

        self.tokens_consumed += n
        return Batch(
            input_ids=torch.from_numpy(np.ascontiguousarray(ids)),
            labels=torch.from_numpy(labels),
            cu_seqlens=torch.from_numpy(cu_seqlens.astype(np.int64)),
        )

    def set_shape(self, seq_len: int, micro_batch: int) -> None:
        """Novo formato (estágio de currículo): nova grade de janelas e nova permutação.

        `epoch` e `tokens_consumed` são mantidos; `position` volta a 0. ValueError (sem alterar
        o stream) se o formato for inválido ou os shards não tiverem `micro_batch` janelas.
        """
        self._apply_shape(seq_len, micro_batch)
        self.position = 0
        self._perm = self._permutation()

    # --- checkpoint -----------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Estado retomável: inteiros simples, incluindo a identidade do conjunto de shards."""
        return {
            "seed": self.seed,
            "epoch": self.epoch,
            "position": self.position,
            "seq_len": self.seq_len,
            "micro_batch": self.micro_batch,
            "tokens_consumed": self.tokens_consumed,
            "eos_id": self.eos_id,
            "n_shards": len(self._shards),
            "n_tokens": self._n_tokens,
        }

    def load_state_dict(self, d: dict) -> None:
        """Restaura o estado (inclusive o formato) e regenera a permutação da época."""
        if d["n_shards"] != len(self._shards) or d["n_tokens"] != self._n_tokens:
            raise ValueError(
                f"conjunto de shards diferente do checkpoint: estado tem {d['n_shards']} shards / "
                f"{d['n_tokens']} tokens, o stream tem {len(self._shards)} / {self._n_tokens}"
            )
        if d["eos_id"] != self.eos_id:
            raise ValueError(f"eos_id do checkpoint ({d['eos_id']}) difere do stream ({self.eos_id})")
        self._apply_shape(d["seq_len"], d["micro_batch"])
        self.seed = d["seed"]
        self.epoch = d["epoch"]
        self.position = d["position"]
        self.tokens_consumed = d["tokens_consumed"]
        self._perm = self._permutation()
