"""Gravador de shards binários contíguos uint32 com sidecar JSON determinístico."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

# Limite superior de representação em uint32
_UINT32_MAX = 4294967295


class BinarySharder:
    """Grava sequências de tokens em shards binários contíguos no formato `uint32` (<u4).

    Cada documento adicionado tem o `eos_id` anexado ao final. Se o documento ultrapassar
    a capacidade restante do shard ativo (`shard_size`), o shard atual é preenchido, fechado,
    o sidecar JSON correspondente é gravado atomicamente e o restante dos tokens transborda
    para os shards seguintes sem nenhuma perda de tokens.

    Parâmetros:
        out_dir: Diretório de saída onde os arquivos `shard_NNNNN.bin` e `shard_NNNNN.json` serão salvos.
        shard_size: Capacidade máxima de cada shard em número de tokens uint32.
                    Padrão: 2^28 = 268.435.456 tokens (~1 GiB).
        eos_id: ID do token especial de fim de documento (EOS). Padrão: 98303.
        unit_len: Comprimento da unidade de empacotamento em tokens. Deve dividir `shard_size`.
                  Padrão: 8192.
        start_shard_index: Índice inicial do primeiro shard gravado. Padrão: 0.
    """

    def __init__(
        self,
        out_dir: str | Path,
        shard_size: int = 268435456,
        eos_id: int = 98303,
        unit_len: int = 8192,
        start_shard_index: int = 0,
    ) -> None:
        if shard_size <= 0:
            raise ValueError(f"shard_size must be positive, got {shard_size}")
        if unit_len <= 0:
            raise ValueError(f"unit_len must be positive, got {unit_len}")
        if shard_size % unit_len != 0:
            raise ValueError(
                f"shard_size ({shard_size}) must be divisible by unit_len ({unit_len})"
            )
        if eos_id < 0 or eos_id > _UINT32_MAX:
            raise ValueError(
                f"eos_id must be non-negative and fit in uint32, got {eos_id}"
            )
        if start_shard_index < 0:
            raise ValueError(
                f"start_shard_index must be non-negative, got {start_shard_index}"
            )

        self.out_dir = Path(out_dir)
        self.shard_size = int(shard_size)
        self.eos_id = int(eos_id)
        self.unit_len = int(unit_len)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self._shard_index = int(start_shard_index)
        self._written_shards: list[Path] = []
        self._current_file: Any = None
        self._current_tmp_bin: Path | None = None
        self._current_hasher: Any = None
        self._current_tokens: int = 0
        self._current_sources: dict[str, int] = {}
        self._closed: bool = False

    @property
    def written_shards(self) -> list[Path]:
        """Lista com todos os shards binários gravados até o momento."""
        return list(self._written_shards)

    @property
    def current_tokens(self) -> int:
        """Quantidade de tokens acumulados no shard aberto atualmente."""
        return self._current_tokens

    @property
    def shard_index(self) -> int:
        """Índice do próximo shard a ser gravado."""
        return self._shard_index

    @property
    def is_closed(self) -> bool:
        """Indica se o sharder já foi encerrado."""
        return self._closed

    def _open_new_shard(self) -> None:
        """Abre arquivo temporário para escrita com buffer e inicializa hasher."""
        bin_name = f"shard_{self._shard_index:05d}.bin"
        self._current_tmp_bin = self.out_dir / f".tmp_{bin_name}"
        self._current_file = open(self._current_tmp_bin, "wb", buffering=1024 * 1024)
        self._current_hasher = hashlib.sha256()
        self._current_tokens = 0
        self._current_sources = {}

    def _close_current_shard(self) -> Path:
        """Fecha o arquivo ativo, realiza rename atômico e gera sidecar JSON."""
        if self._current_file is None:
            raise RuntimeError("No shard file is currently open to close.")

        self._current_file.flush()
        self._current_file.close()
        self._current_file = None

        bin_name = f"shard_{self._shard_index:05d}.bin"
        json_name = f"shard_{self._shard_index:05d}.json"
        final_bin = self.out_dir / bin_name
        final_json = self.out_dir / json_name
        tmp_json = self.out_dir / f".tmp_{json_name}"

        # Rename atômico do binário
        assert self._current_tmp_bin is not None
        self._current_tmp_bin.replace(final_bin)
        self._current_tmp_bin = None

        assert self._current_hasher is not None
        sha256_hex = self._current_hasher.hexdigest()
        self._current_hasher = None

        meta = {
            "n_tokens": self._current_tokens,
            "dtype": "uint32",
            "eos_id": self.eos_id,
            "unit_len": self.unit_len,
            "sources": dict(self._current_sources),
            "sha256": sha256_hex,
        }

        tmp_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        tmp_json.replace(final_json)

        self._written_shards.append(final_bin)
        self._shard_index += 1
        self._current_tokens = 0
        self._current_sources = {}

        return final_bin

    def add_document(
        self,
        token_ids: list[int] | np.ndarray,
        source_name: str | None = None,
    ) -> list[Path]:
        """Adiciona um documento de tokens ao sharder, anexando `eos_id`.

        Se os tokens completarem um ou mais shards, fecha-os e retorna seus caminhos.

        Parâmetros:
            token_ids: Lista ou array numpy 1D com os IDs de token inteiros.
            source_name: Nome opcional da fonte do documento para rastreamento no sidecar.

        Retorna:
            Lista de caminhos dos shards fechados nesta chamada.
        """
        if self._closed:
            raise RuntimeError("BinarySharder is closed.")
        if token_ids is None:
            raise ValueError("token_ids cannot be None.")

        # Converte para array contíguo uint32 com eos_id ao final
        if isinstance(token_ids, np.ndarray):
            if token_ids.ndim != 1:
                raise ValueError(
                    f"token_ids array must be 1D, got ndim={token_ids.ndim}"
                )
            if np.any(token_ids < 0) or np.any(token_ids > _UINT32_MAX):
                raise ValueError("All token_ids must be non-negative and fit in uint32.")
            arr = np.empty(len(token_ids) + 1, dtype="<u4")
            arr[:-1] = token_ids
            arr[-1] = self.eos_id
        else:
            # Lista ou tupla
            for tid in token_ids:
                if tid < 0 or tid > _UINT32_MAX:
                    raise ValueError(
                        f"Token ID {tid} out of uint32 bounds [0, {_UINT32_MAX}]."
                    )
            arr = np.empty(len(token_ids) + 1, dtype="<u4")
            arr[:-1] = token_ids
            arr[-1] = self.eos_id

        closed_shards: list[Path] = []
        total_len = len(arr)
        offset = 0

        while offset < total_len:
            if self._current_file is None:
                self._open_new_shard()

            remaining_in_shard = self.shard_size - self._current_tokens
            chunk_len = min(remaining_in_shard, total_len - offset)
            chunk = arr[offset : offset + chunk_len]
            chunk_bytes = chunk.tobytes()

            self._current_file.write(chunk_bytes)
            self._current_hasher.update(chunk_bytes)
            self._current_tokens += chunk_len

            if source_name is not None:
                self._current_sources[source_name] = (
                    self._current_sources.get(source_name, 0) + chunk_len
                )

            offset += chunk_len

            if self._current_tokens == self.shard_size:
                closed_path = self._close_current_shard()
                closed_shards.append(closed_path)

        return closed_shards

    def close(self) -> list[Path]:
        """Fecha o sharder e descarrega qualquer shard parcial pendente.

        Retorna:
            Lista completa de todos os shards gravados por esta instância.
        """
        if not self._closed:
            if self._current_tokens > 0:
                self._close_current_shard()
            elif self._current_file is not None:
                self._current_file.close()
                self._current_file = None
                if self._current_tmp_bin and self._current_tmp_bin.exists():
                    self._current_tmp_bin.unlink(missing_ok=True)
                self._current_tmp_bin = None
            self._closed = True

        return list(self._written_shards)

    def __enter__(self) -> BinarySharder:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
