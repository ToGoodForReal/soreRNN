"""Fontes de texto do pré-treino (spec §5) lidas em streaming do Hugging Face Hub.

Cada `Source` descreve um dataset (repo, prefixo dos arquivos, formato, coluna de texto e um filtro
opcional por linha). Parquet é lido grupo de linhas a grupo de linhas, só com as colunas usadas
(no FineWeb2-HQ a coluna de embeddings, ~3/4 dos bytes, nunca é baixada); JSONL é lido linha a
linha. Nada é gravado em disco: os bytes vêm por requisições HTTP com `Range` via `HfFileSystem`.

Downloads são dados não confiáveis: só são lidos como parquet/JSON, nunca executados.
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import pyarrow.parquet as pq

# github-code-clean traz a licença de cada arquivo: só as permissivas entram no pré-treino
PERMISSIVE_LICENSES = frozenset(
    {"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc", "cc0-1.0", "unlicense"}
)
_RETRIES = 6


@dataclass(frozen=True)
class Source:
    """Um dataset do Hub. `prefix` filtra os arquivos (`path_in_repo`), `fmt` é "parquet" ou "jsonl".

    `keep` recebe as colunas `extra_columns` de uma linha (dict) e decide se ela entra; `id_column`
    identifica o documento (para a dedup entre fontes que se sobrepõem). `block_size` é o tamanho de
    leitura do `HfFileSystem`: grande para arquivos lidos quase inteiros, pequeno quando a projeção
    de colunas pula a maior parte dos bytes.
    """

    name: str
    repo: str
    prefix: str
    fmt: str = "parquet"
    text_column: str = "text"
    id_column: str | None = None
    extra_columns: tuple[str, ...] = ()
    keep: Callable[[dict[str, Any]], bool] | None = field(default=None, compare=False)
    block_size: int = 32 << 20

    @property
    def columns(self) -> list[str]:
        return [self.text_column, *([self.id_column] if self.id_column else []), *self.extra_columns]


def _permissive(row: dict[str, Any]) -> bool:
    return row["license"] in PERMISSIVE_LICENSES


SOURCES: dict[str, Source] = {
    s.name: s
    for s in [
        Source("fineweb2_pt", "HuggingFaceFW/fineweb-2", "data/por_Latn/train/", id_column="id"),
        Source("fineweb2_hq_pt", "epfml/FineWeb2-HQ", "por_Latn/", id_column="id", block_size=4 << 20),
        Source("wikipedia_pt", "wikimedia/wikipedia", "20231101.pt/"),
        Source("fineweb_edu", "HuggingFaceFW/fineweb-edu", "sample/100BT/"),
        Source("finemath", "HuggingFaceTB/finemath", "finemath-4plus/"),
        Source(
            "github_code", "codeparrot/github-code-clean", "data/", text_column="code",
            extra_columns=("license",), keep=_permissive,
        ),
        Source("opc_annealing", "OpenCoder-LLM/opc-annealing-corpus", "", fmt="jsonl"),
    ]
}


# Mistura do pré-treino (spec §5), em fração dos tokens: 65% PT web, 10% PT de alta qualidade,
# 15% EN educacional e matemática, 10% código. A mesma proporção guia a amostra da poda do tokenizer.
MIXTURE: dict[str, float] = {
    "fineweb2_pt": 0.65,
    "fineweb2_hq_pt": 0.088,
    "wikipedia_pt": 0.012,
    "fineweb_edu": 0.11,
    "finemath": 0.04,
    "github_code": 0.07,
    "opc_annealing": 0.03,
}

CODE_SOURCES = ("github_code", "opc_annealing")


def calibration_mixture(code_frac: float, mixture: dict[str, float] = MIXTURE) -> dict[str, float]:
    """A `mixture` com as fontes de código somando `code_frac` e as demais reescaladas para `1 - code_frac`.

    Serve para a amostra da poda do tokenizer: com mais código na amostra, a poda guarda mais tokens
    de código (identificadores, indentação, operadores) do que a fração de código do treino guardaria,
    ao custo de um pouco de fertilidade em PT (o portão de `scripts/prune_tokenizer.py` mede os dois).
    As proporções dentro de cada grupo (código e não código) ficam as da `mixture`.
    """
    if not 0.0 <= code_frac < 1.0:
        raise ValueError(f"code_frac deve estar em [0, 1), recebeu {code_frac}")
    code = {n: w for n, w in mixture.items() if n in CODE_SOURCES}
    rest = {n: w for n, w in mixture.items() if n not in CODE_SOURCES}
    if code_frac > 0 and not code:
        raise ValueError("code_frac > 0 mas a mistura não tem fontes de código")
    out = {n: w / sum(rest.values()) * (1.0 - code_frac) for n, w in rest.items()}
    out.update({n: w / sum(code.values()) * code_frac for n, w in code.items()} if code_frac > 0 else {})
    return out


def list_files(source: Source, revision: str, api=None) -> list[str]:
    """Caminhos (no repo) dos arquivos da fonte na `revision`, ordenados."""
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    ext = ".parquet" if source.fmt == "parquet" else ".jsonl"
    tree = api.list_repo_tree(
        source.repo, path_in_repo=source.prefix.rstrip("/") or None, repo_type="dataset",
        revision=revision, recursive=True,
    )
    return sorted(f.path for f in tree if getattr(f, "size", None) is not None and f.path.endswith(ext))


def hub_path(source: Source, revision: str, path: str) -> str:
    return f"datasets/{source.repo}@{revision}/{path}"


def _with_retries(fn: Callable[[], Any], what: str) -> Any:
    for attempt in range(_RETRIES):
        try:
            return fn()
        except (OSError, TimeoutError) as e:  # rede: aiohttp/requests sobem como OSError
            if attempt == _RETRIES - 1:
                raise
            wait = 2**attempt * 5
            print(f"[sources] {what}: {type(e).__name__}: {e}; nova tentativa em {wait}s", flush=True)
            time.sleep(wait)
    raise AssertionError("inalcançável")


def iter_documents(source: Source, url: str, fs=None) -> Iterator[tuple[str | None, str]]:
    """`(id, texto)` de cada linha aceita do arquivo `url` (caminho do `fs`), na ordem do arquivo.

    Linhas com texto vazio ou recusadas por `source.keep` são puladas. Cada grupo de linhas do parquet
    é lido com novas tentativas (a conexão é reaberta), então uma queda de rede não perde o arquivo.
    """
    if fs is None:
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem()
    if source.fmt == "jsonl":
        yield from _iter_jsonl(source, url, fs)
        return
    n_groups = _with_retries(
        lambda: pq.ParquetFile(fs.open(url, block_size=source.block_size)).metadata.num_row_groups, url
    )
    handle: list[pq.ParquetFile | None] = [None]

    def read(i: int):
        try:
            if handle[0] is None:
                handle[0] = pq.ParquetFile(fs.open(url, block_size=source.block_size))
            return handle[0].read_row_group(i, columns=source.columns).to_pydict()
        except (OSError, TimeoutError):
            handle[0] = None  # reabre na próxima tentativa
            raise

    for i in range(n_groups):
        cols = _with_retries(lambda: read(i), f"{url} grupo {i}")
        yield from _rows(source, cols)


def _rows(source: Source, cols: dict[str, list]) -> Iterator[tuple[str | None, str]]:
    texts = cols[source.text_column]
    ids = cols[source.id_column] if source.id_column else [None] * len(texts)
    for j, (doc_id, text) in enumerate(zip(ids, texts)):
        if not text:
            continue
        if source.keep is not None and not source.keep({c: cols[c][j] for c in source.extra_columns}):
            continue
        yield doc_id, text


def _iter_jsonl(source: Source, url: str, fs) -> Iterator[tuple[str | None, str]]:
    # JSONL é lido inteiro de uma vez por arquivo (~140 MB no opc-annealing) para poder repetir
    # a leitura numa queda de rede sem duplicar documentos já entregues.
    def read() -> bytes:
        with fs.open(url, "rb", block_size=source.block_size) as f:
            return f.read()

    data = _with_retries(read, url)
    for line in data.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cols = {c: [row.get(c)] for c in source.columns}
        yield from _rows(source, cols)
