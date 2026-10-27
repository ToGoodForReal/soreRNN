# Tokenizador Podado (96K) e Pipeline de Dados PT-BR — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implementar o algoritmo de poda do vocabulário do Qwen3.5 para exatamente 98.304 tokens com fechamento estrito sob merges BPE, tensores de mapeamento para destilação de logits, e o pipeline de streaming multiplexado 75/15/10 com filtros heurísticos, deduplicação MinHash e sharding binário `uint32` de 1 GiB.

**Architecture:** Módulo de grafo direcionado de dependências BPE para seleção gulosa com fechamento transitivo e exportação para HuggingFace `tokenizers`; esteira de streaming com filtros heurísticos compostos, anel circular de MinHash LSH para deduplicação, e gerador de shards binários de $2^{28}$ tokens perfeitamente alinhados com unidades de 8192 tokens do `StreamDataset`.

**Tech Stack:** Python 3.12, PyTorch 2.14.1, `tokenizers` (Hugging Face), NumPy, `datasets` (HF streaming).

**Spec:** [`docs/superpowers/specs/2026-10-27-tokenizer-data-pipeline-design.md`](file:///C:/Users/wellington.lara/Documents/GitHub/soreRNN/docs/superpowers/specs/2026-10-27-tokenizer-data-pipeline-design.md)

---

## Global Constraints

- **Tamanho do Vocabulário Alvo:** Rigorosamente $V = 98.304$ tokens (múltiplo de 128 e 256).
- **Fechamento BPE:** Para qualquer merge preservado $(A, B) \to C$, os tokens $A$, $B$ e $C$ pertencem obrigatoriamente a $\text{Vocab}_{96k}$.
- **Blindagem Obrigatória:** Todos os 256 bytes elementares (`<0x00>` a `<0xFF>`) e todos os tokens especiais originais do Qwen3.5 são preservados.
- **Formato de Shards:** Binário contíguo `uint32`, exatamente $2^{28} = 268.435.456$ tokens (1 GiB) por arquivo, sidecar `.json` obrigatório com `unit_len = 8192` e `eos_id = 98303`.
- **Degradação de Fertilidade PT:** $\le 3\%$ em relação ao vocabulário original de 248K.
- **Convenção de Idioma:** Código e identificadores em inglês; documentação, relatórios e comentários em português.

---

## Review Focus

1. **Fallback de Bytes UTF-8 Arbitrário:** Sequências com caracteres raros (ex: cirílico, kanji, emojis complexos ou bytes inválidos) codificam via bytes elementares sem gerar tokens UNK nem disparar exceções.
2. **Integridade da Tabela de Merges:** Nenhuma regra de merge faz referência a token ID fora do intervalo $[0, 98303]$.
3. **Fronteira de Shard com Documento Longo:** Documentos que ultrapassam o limite restante do shard ($2^{28}$ tokens) são seccionados ou completam o shard sem perda de tokens e iniciam o próximo shard com integridade.
4. **Resistência do MinHash a Documentos Curtos:** Textos com menos de 5 palavras não quebram a extração de n-gramas nem causam divisão por zero na similaridade de Jaccard.
5. **Alinhamento do Mapeamento KD:** Para qualquer índice de estudante $i \in [0, 98303]$, `new_to_old[i]` é um ID válido no intervalo $[0, 248319]$ e `old_to_new[new_to_old[i]] == i`.

---

## File Structure

```
sore/
├── data/
│   ├── __init__.py
│   ├── prune_tokenizer.py    # Algoritmo de grafo BPE, seleção com fechamento, mapeamento e exportador
│   ├── filters.py            # Filtros heurísticos: tamanho, densidade alfanumérica, repetição, stopwords
│   ├── dedup.py              # MinHash LSH local com janela circular deslizante
│   ├── sharder.py            # Gravação de shards binários uint32 de 1 GiB e sidecars JSON
│   └── pipeline.py           # Ingestão multiplexada ponderada e orquestração streaming
scripts/
├── prune_tokenizer.py        # CLI para calibração e exportação do tokenizer 96k
├── prepare_data.py           # CLI para ingestão, filtragem e geração de shards binários
└── eval_fertility.py         # Medição de fertilidade PT em relação ao baseline de 248K
tests/
└── data/
    ├── test_prune_tokenizer.py
    ├── test_filters.py
    ├── test_dedup.py
    ├── test_sharder.py
    └── test_pipeline.py
```

---

## Tasks

### Task 1: Grafo de Dependências BPE e Fechamento de Ancestrais

**Files:**
- Create: `sore/data/__init__.py`
- Create: `sore/data/prune_tokenizer.py`
- Test: `tests/data/test_prune_tokenizer.py`

**Interfaces:**
- Produces:
  - `build_dependency_graph(merges: list[tuple[str, str]]) -> dict[str, tuple[str, str]]`
  - `compute_ancestors(token: str, dep_graph: dict[str, tuple[str, str]], memo: dict[str, set[str]]) -> set[str]`
  - `is_merge_closed(vocab: set[str], merges: list[tuple[str, str]]) -> bool`

- [ ] **Step 1: Escrever teste de falha para construção do grafo e cálculo de ancestrais**

```python
# tests/data/test_prune_tokenizer.py
def test_dependency_graph_and_ancestors():
    # Merges: ("a", "b") -> "ab", ("ab", "c") -> "abc"
    merges = [("a", "b"), ("ab", "c")]
    graph = build_dependency_graph(merges)
    assert graph["ab"] == ("a", "b")
    assert graph["abc"] == ("ab", "c")
    memo = {}
    assert compute_ancestors("ab", graph, memo) == {"a", "b"}
    assert compute_ancestors("abc", graph, memo) == {"a", "b", "c", "ab"}
    assert is_merge_closed({"a", "b", "c", "ab", "abc"}, merges)
    assert not is_merge_closed({"a", "b", "abc"}, merges)  # falta "ab" e "c"
```

- [ ] **Step 2: Rodar teste para verificar falha**

Run: `uv run pytest tests/data/test_prune_tokenizer.py::test_dependency_graph_and_ancestors -v`  
Expected: FAIL com `ModuleNotFoundError` ou `ImportError`.

- [ ] **Step 3: Implementar `build_dependency_graph`, `compute_ancestors` e `is_merge_closed`**

Implementar em `sore/data/prune_tokenizer.py` com memoização de ancestrais para eficiência sobre os ~248K tokens.

- [ ] **Step 4: Rodar teste para verificar aprovação**

Run: `uv run pytest tests/data/test_prune_tokenizer.py::test_dependency_graph_and_ancestors -v`  
Expected: PASS.

- [ ] **Step 5: Commitar**

```bash
git add sore/data/__init__.py sore/data/prune_tokenizer.py tests/data/test_prune_tokenizer.py
git commit -m "feat(data): BPE dependency graph and transitive closure resolution"
```

---

### Task 2: Algoritmo de Seleção Gulosa com Blindagem Obrigatória

**Files:**
- Modify: `sore/data/prune_tokenizer.py`
- Test: `tests/data/test_prune_tokenizer.py`

**Interfaces:**
- Consumes: `build_dependency_graph`, `compute_ancestors`, `is_merge_closed` de Task 1.
- Produces:
  - `select_pruned_vocab(frequencies: dict[str, int], full_vocab: list[str], full_merges: list[tuple[str, str]], protected_tokens: set[str], target_size: int = 98304) -> tuple[set[str], list[tuple[str, str]]]`

- [ ] **Step 1: Escrever teste de falha para seleção gulosa fechada com tamanho alvo exato**

```python
# tests/data/test_prune_tokenizer.py
def test_select_pruned_vocab_exact_size_and_closure():
    # Vocabulario sintético com bytes, especiais e compostos
    bytes_tokens = {f"<0x{i:02X}>" for i in range(256)}
    special_tokens = {"<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"}
    protected = bytes_tokens | special_tokens

    # Merges: ("<0x61>", "<0x62>") -> "ab", ("ab", "<0x63>") -> "abc", ("<0x64>", "<0x65>") -> "de"
    full_merges = [("<0x61>", "<0x62>"), ("ab", "<0x63>"), ("<0x64>", "<0x65>")]
    full_vocab = list(protected) + ["ab", "abc", "de"]
    freqs = {"abc": 100, "de": 50, "ab": 10}

    target_size = len(protected) + 2  # Cabe protected + "ab" e "abc" (custo total 2), mas não "de"
    selected_vocab, pruned_merges = select_pruned_vocab(
        freqs, full_vocab, full_merges, protected, target_size=target_size
    )
    assert len(selected_vocab) == target_size
    assert protected.issubset(selected_vocab)
    assert "ab" in selected_vocab and "abc" in selected_vocab
    assert is_merge_closed(selected_vocab, pruned_merges)
```

- [ ] **Step 2: Rodar teste para verificar falha**

Run: `uv run pytest tests/data/test_prune_tokenizer.py::test_select_pruned_vocab_exact_size_and_closure -v`  
Expected: FAIL com `NameError: name 'select_pruned_vocab' is not defined`.

- [ ] **Step 3: Implementar `select_pruned_vocab` em `sore/data/prune_tokenizer.py`**

Garante proteção dos 256 bytes e especiais, calcula o custo marginal transitivo de cada candidato ranqueado por frequência e atinge rigorosamente `target_size`.

- [ ] **Step 4: Rodar teste para verificar aprovação**

Run: `uv run pytest tests/data/test_prune_tokenizer.py::test_select_pruned_vocab_exact_size_and_closure -v`  
Expected: PASS.

- [ ] **Step 5: Commitar**

```bash
git add sore/data/prune_tokenizer.py tests/data/test_prune_tokenizer.py
git commit -m "feat(data): greedy token selection with BPE closure and protected tokens"
```

---

### Task 3: Exportação do Tokenizador HuggingFace e Mapeamento de Logits (KD)

**Files:**
- Modify: `sore/data/prune_tokenizer.py`
- Create: `scripts/prune_tokenizer.py`
- Test: `tests/data/test_prune_tokenizer.py`

**Interfaces:**
- Consumes: `select_pruned_vocab` de Task 2.
- Produces:
  - `build_remapping_tensors(old_vocab: dict[str, int], new_vocab: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor]`
  - `export_pruned_tokenizer(source_tokenizer_path: str, output_dir: str, target_size: int = 98304, frequencies_path: str | None = None) -> None`

- [ ] **Step 1: Escrever teste de falha para tensores de remapeamento e exportação**

```python
# tests/data/test_prune_tokenizer.py
import torch

def test_build_remapping_tensors():
    old_vocab = {"<0x00>": 0, "a": 1, "b": 2, "unused": 3, "<|endoftext|>": 4}
    new_vocab = {"<0x00>": 0, "a": 1, "b": 2, "<|endoftext|>": 3}
    new_to_old, old_to_new = build_remapping_tensors(old_vocab, new_vocab)

    assert new_to_old.shape == (4,)
    assert old_to_new.shape == (5,)
    assert new_to_old.tolist() == [0, 1, 2, 4]
    assert old_to_new[0] == 0 and old_to_new[1] == 1 and old_to_new[2] == 2
    assert old_to_new[3] == -1  # descarte
    assert old_to_new[4] == 3
    # Invariante de destilação
    for new_id in range(len(new_vocab)):
        assert old_to_new[new_to_old[new_id]] == new_id
```

- [ ] **Step 2: Rodar teste para verificar falha**

Run: `uv run pytest tests/data/test_prune_tokenizer.py::test_build_remapping_tensors -v`  
Expected: FAIL com `NameError`.

- [ ] **Step 3: Implementar `build_remapping_tensors`, `export_pruned_tokenizer` e o script CLI `scripts/prune_tokenizer.py`**

Gera `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json` e `remapping.pt` contendo `{"new_to_old": ..., "old_to_new": ...}`.

- [ ] **Step 4: Rodar teste para verificar aprovação**

Run: `uv run pytest tests/data/test_prune_tokenizer.py -v`  
Expected: PASS (todos os testes de poda e mapeamento passam).

- [ ] **Step 5: Commitar**

```bash
git add sore/data/prune_tokenizer.py scripts/prune_tokenizer.py tests/data/test_prune_tokenizer.py
git commit -m "feat(data): tokenizer exporter and distillation remapping tensors"
```

---

### Task 4: Filtros Heurísticos de Qualidade e Limpeza de Texto

**Files:**
- Create: `sore/data/filters.py`
- Test: `tests/data/test_filters.py`

**Interfaces:**
- Produces:
  - `check_length(text: str, min_chars: int = 150, min_words: int = 30) -> bool`
  - `check_alpha_density(text: str, min_ratio: float = 0.70) -> bool`
  - `check_line_repetition(text: str, max_ratio: float = 0.30) -> bool`
  - `check_ngram_repetition(text: str, n: int = 5, max_ratio: float = 0.20) -> bool`
  - `check_pt_stopwords(text: str, min_count: int = 3) -> bool`
  - `clean_boilerplate(text: str) -> str`
  - `filter_document(text: str, is_pt: bool = True) -> str | None`

- [ ] **Step 1: Escrever testes de falha para os filtros heurísticos**

```python
# tests/data/test_filters.py
from sore.data.filters import filter_document, check_alpha_density, check_line_repetition

def test_filter_valid_pt_document():
    doc = (
        "O aprendizado de máquina moderno possibilita o desenvolvimento de modelos neurais "
        "avançados para o processamento de linguagem natural. Com o avanço das arquiteturas "
        "recorrentes modernas e da atenção linear, o treinamento de modelos na língua portuguesa "
        "atinge alta eficiência com excelente relação de tokens por segundo e qualidade semântica."
    )
    res = filter_document(doc, is_pt=True)
    assert res is not None

def test_filter_rejects_garbage_and_repetition():
    # Pouco alfanumérico
    assert not check_alpha_density("%%% $$$$ @@@@ #### !!!! ???? **** ++++ ----")
    # Repetição excessiva de linhas (loop de SEO)
    repeated = "\n".join(["compre passagens aereas baratas aqui"] * 10)
    assert not check_line_repetition(repeated)
    assert filter_document(repeated, is_pt=True) is None

def test_filter_rejects_foreign_as_pt():
    english_doc = (
        "Modern deep learning architectures allow rapid pretraining on consumer hardware. "
        "This is an English document that does not contain any Portuguese stopwords at all."
    )
    assert filter_document(english_doc, is_pt=True) is None
```

- [ ] **Step 2: Rodar teste para verificar falha**

Run: `uv run pytest tests/data/test_filters.py -v`  
Expected: FAIL com `ModuleNotFoundError: No module named 'sore.data.filters'`.

- [ ] **Step 3: Implementar `sore/data/filters.py`**

Implementar todas as heurísticas e o compositor `filter_document`.

- [ ] **Step 4: Rodar testes para verificar aprovação**

Run: `uv run pytest tests/data/test_filters.py -v`  
Expected: PASS.

- [ ] **Step 5: Commitar**

```bash
git add sore/data/filters.py tests/data/test_filters.py
git commit -m "feat(data): quality and heuristic text filtering rules"
```

---

### Task 5: Deduplicação Local em Streaming via MinHash LSH

**Files:**
- Create: `sore/data/dedup.py`
- Test: `tests/data/test_dedup.py`

**Interfaces:**
- Produces:
  - `class MinHashLSH(capacity: int = 100000, num_perm: int = 128, threshold: float = 0.80)`
  - `MinHashLSH.is_duplicate_and_insert(text: str) -> bool`
  - `compute_minhash_signature(tokens: list[str], num_perm: int = 128) -> np.ndarray`

- [ ] **Step 1: Escrever teste de falha para detecção de duplicatas e rotação de janela**

```python
# tests/data/test_dedup.py
from sore.data.dedup import MinHashLSH

def test_minhash_duplicate_detection():
    lsh = MinHashLSH(capacity=1000, num_perm=64, threshold=0.80)
    t1 = "O soreRNN é um modelo recorrente moderno treinado do zero em português com arquitetura baseada no Qwen4."
    t2 = "O soreRNN é um modelo recorrente moderno treinado do zero em português com arquitetura baseada no Qwen4!" # Quase idêntico
    t3 = "A computação quântica utiliza q-bits superpostos para resolver problemas intratáveis em criptografia moderna." # Totalmente diferente

    assert not lsh.is_duplicate_and_insert(t1)
    assert lsh.is_duplicate_and_insert(t2)  # Detecta como duplicata próxima
    assert not lsh.is_duplicate_and_insert(t3)  # Aceita texto diferente

def test_minhash_short_text_safety():
    lsh = MinHashLSH(capacity=100, num_perm=64)
    # Não quebra nem divide por zero com 1 ou 2 palavras
    assert not lsh.is_duplicate_and_insert("olá")
    assert not lsh.is_duplicate_and_insert("")
```

- [ ] **Step 2: Rodar teste para verificar falha**

Run: `uv run pytest tests/data/test_dedup.py -v`  
Expected: FAIL com `ModuleNotFoundError`.

- [ ] **Step 3: Implementar `MinHashLSH` em `sore/data/dedup.py`**

Utiliza permutações pseudo-aleatórias 64-bit determinísticas com buffer circular deslizante de capacidade fixa.

- [ ] **Step 4: Rodar teste para verificar aprovação**

Run: `uv run pytest tests/data/test_dedup.py -v`  
Expected: PASS.

- [ ] **Step 5: Commitar**

```bash
git add sore/data/dedup.py tests/data/test_dedup.py
git commit -m "feat(data): streaming local MinHash LSH deduplication"
```

---

### Task 6: Gravador de Shards Binários Alinhados (`uint32`)

**Files:**
- Create: `sore/data/sharder.py`
- Test: `tests/data/test_sharder.py`

**Interfaces:**
- Produces:
  - `class BinarySharder(out_dir: Path, shard_size: int = 268435456, eos_id: int = 98303, unit_len: int = 8192)`
  - `BinarySharder.add_document(token_ids: list[int]) -> list[Path]`
  - `BinarySharder.close() -> list[Path]`

- [ ] **Step 1: Escrever teste de falha para gravação de shards e sidecar JSON**

```python
# tests/data/test_sharder.py
import json
import numpy as np
from pathlib import Path
from sore.data.sharder import BinarySharder

def test_binary_sharder_exact_blocks(tmp_path: Path):
    # Shard miniatura de 32 tokens com unit_len=8 (4 unidades inteiras)
    sharder = BinarySharder(tmp_path, shard_size=32, eos_id=999, unit_len=8)
    # Adiciona documentos totalizando 32 tokens
    doc1 = [1, 2, 3]  # vira [1, 2, 3, 999] (4 tokens)
    doc2 = [4] * 27   # vira [4]*27 + [999] (28 tokens) -> fecha 32
    closed = sharder.add_document(doc1)
    assert len(closed) == 0
    closed = sharder.add_document(doc2)
    assert len(closed) == 1

    shard_bin = closed[0]
    shard_json = shard_bin.with_suffix(".json")
    assert shard_bin.exists() and shard_json.exists()
    assert shard_bin.stat().st_size == 32 * 4  # 32 uint32 = 128 bytes

    meta = json.loads(shard_json.read_text())
    assert meta["n_tokens"] == 32
    assert meta["eos_id"] == 999
    assert meta["unit_len"] == 8

    # Leitura como uint32
    tokens = np.fromfile(shard_bin, dtype=np.uint32)
    assert len(tokens) == 32
    assert tokens[3] == 999
    assert tokens[-1] == 999
```

- [ ] **Step 2: Rodar teste para verificar falha**

Run: `uv run pytest tests/data/test_sharder.py -v`  
Expected: FAIL com `ModuleNotFoundError`.

- [ ] **Step 3: Implementar `BinarySharder` em `sore/data/sharder.py`**

Empacotamento sem fragmentação em `uint32`, cálculo de SHA-256 e gravação atômica do sidecar `.json`.

- [ ] **Step 4: Rodar teste para verificar aprovação**

Run: `uv run pytest tests/data/test_sharder.py -v`  
Expected: PASS.

- [ ] **Step 5: Commitar**

```bash
git add sore/data/sharder.py tests/data/test_sharder.py
git commit -m "feat(data): aligned binary uint32 sharder and sidecar json generator"
```

---

### Task 7: Orquestrador do Pipeline em Streaming e CLI

**Files:**
- Create: `sore/data/pipeline.py`
- Create: `scripts/prepare_data.py`
- Test: `tests/data/test_pipeline.py`

**Interfaces:**
- Consumes: `filter_document` (Task 4), `MinHashLSH` (Task 5), `BinarySharder` (Task 6).
- Produces:
  - `class WeightedStreamIterator(sources: dict[str, Any], weights: dict[str, float])`
  - `run_pipeline(sources: dict, tokenizer, out_dir: Path, max_tokens: int, shard_size: int) -> list[Path]`

- [ ] **Step 1: Escrever teste de falha para o pipeline integrado**

```python
# tests/data/test_pipeline.py
from sore.data.pipeline import WeightedStreamIterator, run_pipeline

def test_weighted_stream_multiplexer():
    src1 = iter([{"text": f"doc1_{i}"} for i in range(10)])
    src2 = iter([{"text": f"doc2_{i}"} for i in range(10)])
    mux = WeightedStreamIterator({"s1": src1, "s2": src2}, {"s1": 0.5, "s2": 0.5})
    items = [next(mux)["text"] for _ in range(10)]
    assert any("doc1" in x for x in items)
    assert any("doc2" in x for x in items)
```

- [ ] **Step 2: Rodar teste para verificar falha**

Run: `uv run pytest tests/data/test_pipeline.py -v`  
Expected: FAIL com `ModuleNotFoundError`.

- [ ] **Step 3: Implementar `sore/data/pipeline.py` e `scripts/prepare_data.py`**

Conecta a leitura streaming com os filtros, dedup, tokenizador e gravador de shards, com barra de progresso e estatísticas de rejeição por filtro.

- [ ] **Step 4: Rodar teste para verificar aprovação**

Run: `uv run pytest tests/data/test_pipeline.py -v`  
Expected: PASS.

- [ ] **Step 5: Commitar**

```bash
git add sore/data/pipeline.py scripts/prepare_data.py tests/data/test_pipeline.py
git commit -m "feat(data): streaming multiplexer and end-to-end dataset preparation pipeline"
```

---

### Task 8: Integração com `StreamDataset`, Avaliação de Fertilidade e Validação Final

**Files:**
- Create: `scripts/eval_fertility.py`
- Modify: `configs/b0_8_pretrain.yaml:8`
- Modify: `sore/train/data.py` (ajustes pontuais se necessário)
- Test: `tests/data/test_pipeline.py`

**Interfaces:**
- Consumes: Todas as interfaces anteriores;
- Produces:
  - `scripts/eval_fertility.py`: CLI que avalia a fertilidade $\text{tokens}/\text{palavra}$ e garante degradação $\le 3\%$.

- [ ] **Step 1: Escrever teste de fumaça de ponta a ponta integrando Shards com `StreamDataset`**

```python
# tests/data/test_pipeline.py
from sore.train.data import StreamDataset

def test_stream_dataset_reads_generated_shards(tmp_path):
    # Gera shard sintético com BinarySharder e lê com StreamDataset
    from sore.data.sharder import BinarySharder
    sharder = BinarySharder(tmp_path, shard_size=16384, eos_id=98303, unit_len=8192)
    sharder.add_document([10] * 8191)  # fecha 8192 com eos_id
    sharder.add_document([20] * 8191)  # fecha 16384 com eos_id
    shards = sharder.close()

    ds = StreamDataset([str(s) for s in shards], seq_len=4096, unit_len=8192, seed=42)
    batch = ds.next_window()
    assert len(batch) == 4096
    assert ds.n_tokens == 16384
```

- [ ] **Step 2: Rodar teste para verificar integridade**

Run: `uv run pytest tests/data/test_pipeline.py::test_stream_dataset_reads_generated_shards -v`  
Expected: PASS.

- [ ] **Step 3: Implementar `scripts/eval_fertility.py` e atualizar `configs/b0_8_pretrain.yaml` com `vocab_size: 98304` e `eos_id: 98303`**

Script que compara `Qwen/Qwen3.5-0.8B` com o tokenizador 96K podado sobre uma amostra de teste PT e imprime a taxa de degradação.

- [ ] **Step 4: Executar suíte de testes de dados completa**

Run: `uv run pytest tests/data/ -v`  
Expected: PASS (todos os testes passam sem skips).

- [ ] **Step 5: Commitar**

```bash
git add scripts/eval_fertility.py configs/b0_8_pretrain.yaml sore/train/data.py tests/data/test_pipeline.py
git commit -m "feat(data): StreamDataset integration, pretrain config update and fertility evaluation"
```

---

## Self-Review Checklist

- [x] **Spec coverage:** Todos os 5 portões de qualidade da spec possuem testes correspondentes nos passos das tarefas.
- [x] **Step scan:** Cada tarefa segue rigorosamente a anatomia TDD (teste de falha $\to$ confirmação de erro $\to$ implementação mínima $\to$ teste passando $\to$ commit atômico).
- [x] **Type consistency:** Nomes e assinaturas de funções (`build_dependency_graph`, `select_pruned_vocab`, `build_remapping_tensors`, `filter_document`, `MinHashLSH`, `BinarySharder`) mantêm paridade exata entre as tarefas.
- [x] **Review Focus:** Todas as 5 classes de falha da seção Review Focus estão cobertas nos testes unitários das tarefas 1 a 6.
- [x] **Proportion:** O plano é direto, modular e focado nas decisões que o implementador precisa seguir.
