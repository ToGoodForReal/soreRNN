"""Testes do gerador de shards de pré-treino (`sore.train.pretrain_data`) com fontes falsas."""
import json

import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from sore.train.data import PackedTokenStream, _open_shard
from sore.train.pretrain_data import SourcePlan, load_ids, plan_order, write_source

EOS = 0
WORDS = ["[UNK]", "<eos>", "a", "b", "c", "d", "e"]  # "a" -> 2 ... "e" -> 6


@pytest.fixture(scope="module")
def tok():
    t = Tokenizer(WordLevel({w: i for i, w in enumerate(WORDS)}, unk_token="[UNK]"))
    t.pre_tokenizer = Whitespace()
    return t


def _files(n_files=3, docs_per_file=50):
    """arquivo -> lista de (id, texto); doc k do arquivo f tem 1 + (k % 4) palavras."""
    return {
        f"f{f}": [(f"id{f}_{k}", " ".join(WORDS[2 + (k + j) % 5] for j in range(1 + k % 4))) for k in range(docs_per_file)]
        for f in range(n_files)
    }


def _reader(files, fail_on=None):
    def read(name):
        for i, doc in enumerate(files[name]):
            if fail_on == (name, i):
                raise OSError("queda de rede simulada")
            yield doc

    return read


def _all_tokens(src_dir):
    return np.concatenate([np.asarray(_open_shard(p)) for p in sorted(src_dir.glob("*.bin"))])


def test_quota_cut_at_document_boundary_with_eos(tmp_path, tok):
    files = _files()
    got = write_source(SourcePlan("s", 100, list(files)), tmp_path, tok, EOS, "uint16", _reader(files), log=lambda m: None)
    toks = _all_tokens(tmp_path / "s")
    assert got == len(toks) <= 100 and len(toks) > 90  # o próximo doc (até 5 tokens) não coube
    assert toks[-1] == EOS and (toks == EOS).sum() > 0
    # cada documento é a tokenização do texto seguida do EOS, na ordem do arquivo
    docs = np.split(toks, np.flatnonzero(toks == EOS) + 1)[:-1]
    for (_, text), d in zip(files["f0"], docs):
        assert d[:-1].tolist() == tok.encode(text).ids
    assert sorted(p.name for p in (tmp_path / "s").glob("*.bin")) == ["s_00000.bin"]
    assert json.loads((tmp_path / "s" / "progress.json").read_text())["tokens"] == got


def test_full_source_is_not_reopened_on_resume(tmp_path, tok):
    files = _files()
    quota = sum(len(t.split()) + 1 for _, t in files["f0"][:20]) + 1  # 1 token além do 20º doc: o 21º não cabe
    got = write_source(SourcePlan("s", quota, list(files)), tmp_path, tok, EOS, "uint16", _reader(files), log=lambda m: None)
    assert got == quota - 1 and json.loads((tmp_path / "s" / "progress.json").read_text())["complete"]

    def no_read(name):
        raise AssertionError(f"não deveria abrir {name}")

    again = write_source(SourcePlan("s", quota, list(files)), tmp_path, tok, EOS, "uint16", no_read, log=lambda m: None)
    assert again == got and sorted(p.name for p in (tmp_path / "s").glob("*.bin")) == ["s_00000.bin"]


def test_resume_after_failure_matches_uninterrupted(tmp_path, tok):
    files = _files()
    ref = tmp_path / "ref"
    write_source(SourcePlan("s", 10_000, list(files)), ref, tok, EOS, "uint32", _reader(files), log=lambda m: None)

    out = tmp_path / "out"
    with pytest.raises(OSError):
        write_source(SourcePlan("s", 10_000, list(files)), out, tok, EOS, "uint32", _reader(files, ("f1", 30)), log=lambda m: None)
    assert (out / "s" / "s_00000.bin").exists() and not (out / "s" / "s_00001.bin").exists()
    write_source(SourcePlan("s", 10_000, list(files)), out, tok, EOS, "uint32", _reader(files), log=lambda m: None)
    assert not list((out / "s").glob("*.tmp"))
    assert np.array_equal(_all_tokens(out / "s"), _all_tokens(ref / "s"))
    stream = PackedTokenStream(sorted((out / "s").glob("*.bin")), seq_len=8, micro_batch=2, eos_id=EOS, seed=0)
    assert next(stream).input_ids.shape == (1, 16)


def test_dedup_skips_ids_of_the_other_source(tmp_path, tok):
    files = _files(n_files=1, docs_per_file=20)
    hq = {"h0": files["f0"][:5]}
    write_source(SourcePlan("hq", 10_000, ["h0"]), tmp_path, tok, EOS, "uint16", _reader(hq), save_ids=True, log=lambda m: None)
    ids = load_ids(tmp_path / "hq")
    assert ids == {f"id0_{k}" for k in range(5)}
    write_source(SourcePlan("web", 10_000, ["f0"]), tmp_path, tok, EOS, "uint16", _reader(files), skip_ids=ids, log=lambda m: None)
    n_docs = int((_all_tokens(tmp_path / "web") == EOS).sum())
    assert n_docs == 15


def test_exhausted_source_is_marked_and_not_redone(tmp_path, tok):
    files = _files(n_files=1, docs_per_file=10)
    msgs = []
    got = write_source(SourcePlan("s", 10_000, ["f0"]), tmp_path, tok, EOS, "uint16", _reader(files), log=msgs.append)
    assert got < 10_000 and any("esgotada" in m for m in msgs)
    prog = json.loads((tmp_path / "s" / "progress.json").read_text())
    assert prog["exhausted"] and prog["files_done"] == ["f0"]
    again = write_source(SourcePlan("s", 10_000, ["f0"]), tmp_path, tok, EOS, "uint16", _reader({}), log=lambda m: None)
    assert again == got


def test_ids_that_do_not_fit_dtype_raise(tmp_path, tok):
    files = _files(n_files=1)
    with pytest.raises(ValueError, match="cabe"):
        write_source(SourcePlan("s", 10_000, ["f0"]), tmp_path, tok, 70_000, "uint16", _reader(files), log=lambda m: None)


def test_plan_order_puts_hq_first():
    assert plan_order(["fineweb2_pt", "wikipedia_pt", "fineweb2_hq_pt"]) == ["fineweb2_hq_pt", "fineweb2_pt", "wikipedia_pt"]
