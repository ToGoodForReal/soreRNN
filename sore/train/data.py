"""Shards de tokens uint32 e stream empacotado determinístico e retomável.

Formato em disco: `uint32` little-endian cru (`<u4`) mais um sidecar JSON
`{"n_tokens": int, "dtype": "uint32"}` ao lado (`path.with_suffix(".json")`).

O stream sorteia unidades fixas de `unit_len` tokens por uma permutação determinística de
`(seed, epoch, unit_len)`, corta cada unidade em janelas consecutivas de `seq_len + 1` tokens
(entrada = `[:-1]`, alvo = `[1:]`) e empacota `micro_batch` janelas numa única linha `[1, N]`, com
`cu_seqlens` marcando fronteiras de documento (EOS) e de linha.
"""
from __future__ import annotations

import glob
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

    def __len__(self) -> int:
        return self.input_ids.shape[-1]


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

    Regras (rulings R14 e R22): o corpus é dividido em unidades fixas de `unit_len` tokens
    (padrão: `seq_len`), que nunca cruzam shards. Um shard de tamanho L tem `(L - 1) // unit_len`
    unidades; a unidade u cobre os tokens `[u * U, u * U + U + 1)`, e um shard com menos de
    `unit_len + 1` tokens não contribui com nenhuma. Com `seq_len = s` (exige `U % s == 0`) cada
    unidade rende `U // s` janelas consecutivas, `[u * U + k * s, u * U + (k + 1) * s + 1)` (janelas
    vizinhas dividem um token: o alvo da última entrada).

    A cada época as unidades são visitadas na ordem de
    `default_rng([seed, epoch, unit_len]).permutation(num_units)`, que não depende do `seq_len`.
    `position` conta janelas do formato atual dentro da época. Quando restam menos que
    `micro_batch` janelas, a sobra é descartada e começa a época seguinte. Uma troca de formato
    (`set_shape`) continua a mesma época a partir da 1a unidade ainda não tocada, então nenhuma
    janela se repete dentro da época (ruling R22).
    """

    def __init__(
        self,
        paths: list[Path],
        seq_len: int,
        micro_batch: int,
        eos_id: int,
        seed: int,
        unit_len: int | None = None,
    ):
        if not paths:
            raise ValueError("é preciso ao menos um shard")
        self._shards = [_open_shard(Path(p)) for p in paths]
        self._n_tokens = sum(len(s) for s in self._shards)
        self.unit_len = seq_len if unit_len is None else unit_len
        if self.unit_len < 1:
            raise ValueError(f"unit_len deve ser >= 1, recebeu {self.unit_len}")
        counts = np.array([max(0, (len(s) - 1) // self.unit_len) for s in self._shards], dtype=np.int64)
        self._cum = np.cumsum(counts)  # unidades acumuladas até o fim de cada shard
        self._first = self._cum - counts  # índice global da 1a unidade de cada shard
        self.num_units = int(self._cum[-1])
        self.eos_id = eos_id
        self.seed = seed
        self.epoch = 0
        self.position = 0
        self.tokens_consumed = 0
        self._apply_shape(seq_len, micro_batch)
        self._perm = self._permutation()

    # --- unidades, janelas e permutação ---------------------------------------------------

    def _windows_per_unit(self, seq_len: int, micro_batch: int) -> int:
        """Janelas por unidade para o formato dado; valida o formato (ValueError)."""
        if seq_len < 1 or micro_batch < 1:
            raise ValueError(f"seq_len e micro_batch devem ser >= 1, recebeu {seq_len}, {micro_batch}")
        if self.unit_len % seq_len != 0:
            raise ValueError(f"unit_len={self.unit_len} não é múltiplo de seq_len={seq_len}")
        per_unit = self.unit_len // seq_len
        if self.num_units * per_unit < micro_batch:
            raise ValueError(
                f"{self.num_units * per_unit} janelas de seq_len={seq_len} nos shards "
                f"({self.num_units} unidades de {self.unit_len}), insuficiente para micro_batch={micro_batch}"
            )
        return per_unit

    def _apply_shape(self, seq_len: int, micro_batch: int) -> None:
        per_unit = self._windows_per_unit(seq_len, micro_batch)  # valida antes de mexer no estado
        self.seq_len = seq_len
        self.micro_batch = micro_batch
        self._per_unit = per_unit
        self.num_windows = self.num_units * per_unit

    def _permutation(self) -> np.ndarray:
        rng = np.random.default_rng([self.seed, self.epoch, self.unit_len])
        return rng.permutation(self.num_units)

    # --- iteração ---------------------------------------------------------------------------

    def __iter__(self) -> PackedTokenStream:
        return self

    def __next__(self) -> Batch:
        if self.position + self.micro_batch > self.num_windows:  # descarta a sobra da época
            self.epoch += 1
            self.position = 0
            self._perm = self._permutation()
        windows = np.arange(self.position, self.position + self.micro_batch)
        self.position += self.micro_batch

        units = self._perm[windows // self._per_unit]
        shard_ids = np.searchsorted(self._cum, units, side="right")
        starts = (units - self._first[shard_ids]) * self.unit_len + (windows % self._per_unit) * self.seq_len
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
        """Novo formato (estágio de currículo) na mesma época e na mesma permutação de unidades.

        `position` passa para a 1a unidade que ainda não foi tocada, nem em parte (o resto de uma
        unidade começada é pulado), e `epoch` e `tokens_consumed` são mantidos. ValueError (sem
        alterar o stream) se o formato for inválido, se `unit_len % seq_len != 0` ou se não houver
        `micro_batch` janelas.
        """
        old_per_unit = self._per_unit
        self._apply_shape(seq_len, micro_batch)
        units_started = -(-self.position // old_per_unit)  # ceil
        self.position = units_started * self._per_unit

    # --- checkpoint -----------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Estado retomável: inteiros simples, incluindo a identidade do conjunto de shards."""
        return {
            "seed": self.seed,
            "epoch": self.epoch,
            "position": self.position,
            "seq_len": self.seq_len,
            "micro_batch": self.micro_batch,
            "unit_len": self.unit_len,
            "tokens_consumed": self.tokens_consumed,
            "eos_id": self.eos_id,
            "n_shards": len(self._shards),
            "n_tokens": self._n_tokens,
        }

    def load_state_dict(self, d: dict) -> None:
        """Restaura o estado (inclusive o formato) e regenera a permutação da época.

        ValueError se os shards, o `eos_id` ou o `unit_len` diferirem: o `unit_len` define a
        grade de unidades e a permutação, então a posição salva só vale com o mesmo valor.
        """
        if d["n_shards"] != len(self._shards) or d["n_tokens"] != self._n_tokens:
            raise ValueError(
                f"conjunto de shards diferente do checkpoint: estado tem {d['n_shards']} shards / "
                f"{d['n_tokens']} tokens, o stream tem {len(self._shards)} / {self._n_tokens}"
            )
        if d["eos_id"] != self.eos_id:
            raise ValueError(f"eos_id do checkpoint ({d['eos_id']}) difere do stream ({self.eos_id})")
        if d["unit_len"] != self.unit_len:
            raise ValueError(
                f"unit_len do checkpoint ({d['unit_len']}) difere do stream ({self.unit_len}); "
                "no trainer ele é o maior seq_len dos estágios"
            )
        self._apply_shape(d["seq_len"], d["micro_batch"])
        self.seed = d["seed"]
        self.epoch = d["epoch"]
        self.position = d["position"]
        self.tokens_consumed = d["tokens_consumed"]
        self._perm = self._permutation()


class StreamDataset:
    """Interface streaming de alto nível para carregar shards contíguos de tokens uint32.

    Compatível com shards produzidos por `BinarySharder` (`sore.data.sharder`),
    permitindo iteração sequencial e consumo por janelas (`next_window`).

    Parâmetros:
        paths: Lista de caminhos ou caminho individual dos arquivos binários de shard.
        seq_len: Comprimento da janela de tokens retornada (padrão: 4096).
        unit_len: Tamanho da unidade de empacotamento nos shards (padrão: 8192).
        seed: Semente determinística para o cursor do stream (padrão: 42).
        eos_id: ID do token de fim de documento; se None, deduz do sidecar JSON do shard (padrão: None).
        micro_batch: Quantidade de janelas por lote interno (padrão: 1).
    """

    def __init__(
        self,
        paths: list[str | Path] | str | Path,
        seq_len: int = 4096,
        unit_len: int = 8192,
        seed: int = 42,
        eos_id: int | None = None,
        micro_batch: int = 1,
    ) -> None:
        if isinstance(paths, (str, Path)):
            paths = [paths]
        resolved_paths: list[Path] = []
        for p in paths:
            p_str = str(p)
            if "*" in p_str or "?" in p_str:
                matches = sorted(glob.glob(p_str))
                resolved_paths.extend(Path(m) for m in matches)
            else:
                resolved_paths.append(Path(p))
        self.paths = resolved_paths
        if not self.paths:
            raise ValueError("é preciso ao menos um shard")

        if eos_id is None:
            sidecar = self.paths[0].with_suffix(".json")
            if sidecar.exists():
                try:
                    meta = json.loads(sidecar.read_text(encoding="utf-8"))
                    eos_id = meta.get("eos_id", 0)
                except Exception:
                    eos_id = 0
            else:
                eos_id = 0

        self.eos_id = int(eos_id)
        self.seq_len = int(seq_len)
        self.unit_len = int(unit_len)
        self.micro_batch = int(micro_batch)
        self.seed = int(seed)

        self._stream = PackedTokenStream(
            paths=self.paths,
            seq_len=self.seq_len,
            micro_batch=self.micro_batch,
            eos_id=self.eos_id,
            seed=self.seed,
            unit_len=self.unit_len,
        )

    @property
    def n_tokens(self) -> int:
        """Quantidade total de tokens agregada em todos os shards carregados."""
        return self._stream._n_tokens

    def next_window(self) -> torch.Tensor:
        """Avança o stream e retorna a próxima janela de `seq_len` tokens como tensor 1D."""
        batch = next(self._stream)
        return batch.input_ids[0]

    def __iter__(self) -> StreamDataset:
        return self

    def __next__(self) -> Batch:
        return next(self._stream)

    def state_dict(self) -> dict:
        return self._stream.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self._stream.load_state_dict(state)

