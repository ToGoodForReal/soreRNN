"""Poda o tokenizer Qwen3.5 para `--vocab-size` ids por frequência numa amostra da mistura (spec §4).

Amostra: ~`--sample-tokens` tokens repartidos entre as fontes pelos pesos de `calibration_mixture(--code-frac)`
(a `MIXTURE` do treino com o código levado a `--code-frac`, padrão 25%, para a poda guardar mais tokens de
código), lidos de
até `--files-per-source` arquivos espaçados de cada fonte (para não ficar num só dump/arquivo).
Avaliação (fora da amostra): fertilidade (tokens por palavra) do vocabulário completo e do podado
em PT (C4-pt local de `data/raw`, outra fonte que não a da amostra), EN e código (último arquivo
de cada fonte), e o round-trip `decode(encode(x)) == NFC(x)`. Critérios: PT no máximo 3% pior que o
vocabulário completo (spec) e código no máximo `--max-code-fertility-loss` (padrão 3%) pior.

Saída em `--out`: `tokenizer.json` (podado, com os tokens adicionados), `old_to_new.npy`,
`counts.npy` (contagens da amostra no vocabulário completo) e `meta.json`.

Uso: uv run python scripts/prune_tokenizer.py --out data/tokenizer [--vocab-size 98304] [--sample-tokens 2e8] [--code-frac 0.25]
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
import unicodedata
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi
from tokenizers import Tokenizer

from sore.train.sources import SOURCES, calibration_mixture, hub_path, iter_documents, list_files
from sore.train.text_sample import download, iter_jsonl_gz_texts, resolve_revision
from sore.train.tokenizer_prune import build_pruned, select_vocab, to_json_str, without_added_tokens

EVAL_DOCS = 2000


def sample_counts(
    tokenizer, n_ids: int, sample_tokens: int, files_per_source: int, revisions: dict, mixture: dict[str, float]
) -> tuple:
    counts = np.zeros(n_ids, dtype=np.int64)
    used: dict[str, dict] = {}
    for name, weight in mixture.items():
        if weight <= 0:
            continue
        src = SOURCES[name]
        files = list_files(src, revisions[name])
        step = max(1, len(files) // files_per_source)
        picked = files[: step * files_per_source : step][:files_per_source]
        quota = int(sample_tokens * weight)
        per_file = -(-quota // len(picked))
        got = 0
        t0 = time.time()
        for path in picked:
            n_file = 0
            docs = (text for _, text in iter_documents(src, hub_path(src, revisions[name], path)))
            while n_file < per_file:
                batch = list(itertools.islice(docs, 64))  # lote pequeno: fontes de docs longos não estouram a cota
                if not batch:
                    break
                ids = np.fromiter(
                    (i for e in tokenizer.encode_batch(batch, add_special_tokens=False) for i in e.ids), dtype=np.int64
                )
                counts += np.bincount(ids, minlength=n_ids)
                n_file += len(ids)
            got += n_file
        used[name] = {"files": picked, "tokens": got}
        print(f"[amostra] {name}: {got:,} tokens de {len(picked)} arquivo(s) ({time.time() - t0:.0f}s)", flush=True)
    return counts, used


def eval_texts(revisions: dict, raw_dir: Path) -> dict[str, list[str]]:
    """Textos fora da amostra: C4-pt local (PT) e o último arquivo das fontes EN e de código."""
    c4 = sorted(raw_dir.glob("c4-pt.tfrecord-*.json.gz"))
    out = {"pt": list(itertools.islice(iter_jsonl_gz_texts(c4[0]), EVAL_DOCS)) if c4 else []}
    for key, name in [("en", "fineweb_edu"), ("code", "github_code")]:
        src = SOURCES[name]
        last = list_files(src, revisions[name])[-1]
        out[key] = [t for _, t in itertools.islice(iter_documents(src, hub_path(src, revisions[name], last)), EVAL_DOCS)]
    return out


def fertility(tokenizer, texts: list[str]) -> float:
    n_tok = sum(len(e.ids) for e in tokenizer.encode_batch(texts, add_special_tokens=False))
    return n_tok / max(1, sum(len(t.split()) for t in texts))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=98304, help="ids no total, tokens adicionados inclusos")
    parser.add_argument("--sample-tokens", type=float, default=2e8)
    parser.add_argument("--files-per-source", type=int, default=8)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--max-pt-fertility-loss", type=float, default=0.03)
    parser.add_argument("--code-frac", type=float, default=0.25, help="fração de código na amostra de calibração")
    parser.add_argument("--max-code-fertility-loss", type=float, default=0.03)
    args = parser.parse_args()
    mixture = calibration_mixture(args.code_frac)
    # o portão de PT é o critério da spec: sem o C4-pt local ele passaria sem medir nada
    if not sorted(args.raw_dir.glob("c4-pt.tfrecord-*.json.gz")):
        parser.error(
            f"nenhum c4-pt.tfrecord-*.json.gz em {args.raw_dir} (texto PT fora da amostra para o portão de "
            "fertilidade); gere com scripts/make_text_sample.py, que baixa esses arquivos em data/raw"
        )

    revision = resolve_revision(args.tokenizer, args.revision)
    tok_path = download(
        f"https://huggingface.co/{args.tokenizer}/resolve/{revision}/tokenizer.json",
        args.raw_dir / f"tokenizer-{args.tokenizer.replace('/', '--')}-{revision[:12]}.json",
    )
    tok_json = json.loads(tok_path.read_text())
    full = Tokenizer.from_str(to_json_str(without_added_tokens(tok_json)))
    n_ids = max([*tok_json["model"]["vocab"].values(), *(a["id"] for a in tok_json["added_tokens"])]) + 1
    budget = args.vocab_size - len(tok_json["added_tokens"])

    api = HfApi()
    revisions = {name: api.dataset_info(SOURCES[name].repo).sha for name in SOURCES}
    counts, used = sample_counts(full, n_ids, int(args.sample_tokens), args.files_per_source, revisions, mixture)
    keep = select_vocab(tok_json, counts, budget)
    pruned_json, old_to_new = build_pruned(tok_json, keep)
    pruned = Tokenizer.from_str(to_json_str(without_added_tokens(pruned_json)))

    texts = eval_texts(revisions, args.raw_dir)
    fert = {}
    for key, docs in texts.items():
        if not docs:
            continue
        f_full, f_pruned = fertility(full, docs), fertility(pruned, docs)
        fert[key] = {"full": f_full, "pruned": f_pruned, "loss": f_pruned / f_full - 1}
        print(f"[fertilidade] {key}: {f_full:.4f} -> {f_pruned:.4f} tokens/palavra ({fert[key]['loss']:+.2%})", flush=True)
    bad = [t for docs in texts.values() for t in docs[:200] if pruned.decode(pruned.encode(t).ids) != unicodedata.normalize("NFC", t)]
    print(f"[round-trip] {sum(len(d[:200]) for d in texts.values()) - len(bad)} ok, {len(bad)} falha(s)", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "tokenizer.json").write_text(to_json_str(pruned_json))
    np.save(args.out / "old_to_new.npy", old_to_new)
    np.save(args.out / "counts.npy", counts)
    eos = pruned_json["added_tokens"][0]
    assert eos["content"] == "<|endoftext|>"
    meta = {
        "tokenizer_repo": args.tokenizer,
        "tokenizer_revision": revision,
        "vocab_size": len(keep) + len(pruned_json["added_tokens"]),
        "bpe_tokens": len(keep),
        "merges": len(pruned_json["model"]["merges"]),
        "eos_id": eos["id"],
        "sample": {
            "tokens": int(counts.sum()), "code_frac": args.code_frac, "mixture": mixture,
            "sources": used, "dataset_revisions": revisions,
        },
        "fertility": fert,
        "roundtrip_failures": len(bad),
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"{args.out}: vocab {meta['vocab_size']} (eos_id {meta['eos_id']}), {meta['merges']} merges", flush=True)
    pt_loss = fert["pt"]["loss"]
    code_loss = fert.get("code", {}).get("loss", 0.0)
    if bad or pt_loss > args.max_pt_fertility_loss or code_loss > args.max_code_fertility_loss:
        raise SystemExit(
            f"REPROVADO: round-trip {len(bad)} falha(s), fertilidade PT {pt_loss:+.2%} (máx. "
            f"{args.max_pt_fertility_loss:+.0%}), código {code_loss:+.2%} (máx. {args.max_code_fertility_loss:+.0%}). "
            "PT acima: baixe --code-frac ou suba --vocab-size; código acima: suba --code-frac ou --vocab-size"
        )
    print("APROVADO", flush=True)


if __name__ == "__main__":
    main()
