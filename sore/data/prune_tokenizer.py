"""Algoritmo de grafo BPE, resolução de ancestrais transitivos e seleção gulosa com fechamento."""

import json
import os
from pathlib import Path
import re

from tokenizers import Tokenizer
import torch

_HEX_BYTE_RE = re.compile(r"^<0x([0-9a-fA-F]{2})>$")


def _piece_to_str(piece: str) -> str:
    """Converte representação hexadecimal de byte (<0xHH>) para caractere, ou retorna a peça original."""
    m = _HEX_BYTE_RE.match(piece)
    if m:
        byte_val = int(m.group(1), 16)
        try:
            return bytes([byte_val]).decode("utf-8")
        except UnicodeDecodeError:
            return bytes([byte_val]).decode("latin-1")
    return piece


def _merge_result(left: str, right: str) -> str:
    """Determina o token resultante da concatenação de duas peças BPE, considerando fallback de bytes."""
    m_left = _HEX_BYTE_RE.match(left)
    m_right = _HEX_BYTE_RE.match(right)
    if m_left and m_right:
        raw_bytes = bytes([int(m_left.group(1), 16), int(m_right.group(1), 16)])
        try:
            return raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return _piece_to_str(left) + _piece_to_str(right)
    elif m_left or m_right:
        s_left = _piece_to_str(left)
        s_right = _piece_to_str(right)
        return s_left + s_right
    return left + right


def build_dependency_graph(
    merges: list[tuple[str, str]],
) -> dict[str, tuple[str, str]]:
    """Constrói o grafo direcionado de dependências a partir das regras de merge BPE.

    No algoritmo BPE, cada merge unifica dois tokens existentes A e B para formar
    o token composto C = A + B. O grafo mapeia o token resultante para a tupla de
    seus dois constituintes imediatos:
        C -> (A, B)

    Args:
        merges: Lista de tuplas (A, B) representando regras de merge em ordem de aplicação.

    Returns:
        Dicionário mapeando cada token composto resultante para a tupla (A, B).
    """
    graph: dict[str, tuple[str, str]] = {}
    for left, right in merges:
        graph[left + right] = (left, right)
        res = _merge_result(left, right)
        if res != left + right:
            graph[res] = (left, right)
    return graph


def compute_ancestors(
    token: str,
    dep_graph: dict[str, tuple[str, str]],
    memo: dict[str, set[str]] | None = None,
) -> set[str]:
    """Calcula recursivamente o fechamento transitivo de ancestrais de um token no grafo BPE.

    Ancestrais(C) = {A, B} U Ancestrais(A) U Ancestrais(B).
    Se o token for elementar (não estiver em dep_graph), o conjunto de ancestrais é vazio.
    Utiliza memoização in-place no dicionário `memo` para complexidade O(1) amortizada
    em chamadas subsequentes sobre os ~248K tokens do vocabulário original.

    Args:
        token: Token cujo conjunto de ancestrais transitivos será calculado.
        dep_graph: Grafo de dependências mapeando token composto para (pai_esq, pai_dir).
        memo: Dicionário opcional de memoização (token -> conjunto de ancestrais transitivos).

    Returns:
        Conjunto contendo todos os ancestrais transitivos do token.
    """
    if memo is None:
        memo = {}

    if token in memo:
        return memo[token]

    if token not in dep_graph:
        ancestors: set[str] = set()
        memo[token] = ancestors
        return ancestors

    left, right = dep_graph[token]
    ancestors = {left, right}
    ancestors.update(compute_ancestors(left, dep_graph, memo))
    ancestors.update(compute_ancestors(right, dep_graph, memo))
    memo[token] = ancestors
    return ancestors


def is_merge_closed(
    vocab: set[str],
    merges: list[tuple[str, str]],
) -> bool:
    """Verifica se um determinado vocabulário é estritamente fechado sob uma lista de merges BPE.

    Um vocabulário é fechado sob merges se, para cada regra (A, B) cujo resultado
    C pertença ao vocabulário, ambos os predecessores A e B obrigatoriamente
    também pertençam ao vocabulário.

    Args:
        vocab: Conjunto de tokens que compõem o vocabulário (ex: Vocab_96k).
        merges: Lista de tuplas de merge (A, B).

    Returns:
        True se todas as dependências imediatas estiverem no vocabulário, False caso contrário.
    """
    for left, right in merges:
        candidates = {left + right, _merge_result(left, right)}
        for merged in candidates:
            if merged in vocab:
                if left not in vocab or right not in vocab:
                    return False
    return True


def select_pruned_vocab(
    frequencies: dict[str, int],
    full_vocab: list[str],
    full_merges: list[tuple[str, str]],
    protected_tokens: set[str],
    target_size: int = 98304,
) -> tuple[set[str], list[tuple[str, str]]]:
    """Executa a seleção gulosa de tokens com blindagem obrigatória e fechamento transitivo BPE.

    Blindagem obrigatória:
        - Todos os tokens em `protected_tokens` (256 bytes elementares e tokens especiais)
          são preservados incondicionalmente, junto com todo o grafo de seus ancestrais recursivos.

    Seleção gulosa:
        - Candidatos de `full_vocab` são ranqueados por contagem de ocorrência decrescente;
        - Para cada candidato, é computado o incremento transitivo de dependências:
          Delta(t) = ({t} U Ancestrais(t)) \\ S;
        - O token e seus ancestrais faltantes são adicionados se |S| + |Delta(t)| <= target_size;
        - Se a lista de frequências for percorrida e |S| < target_size, realiza preenchimento
          residual com os tokens de menor custo marginal transitivo até atingir rigorosamente target_size.

    Args:
        frequencies: Dicionário contendo a frequência de tokens observada no corpus de calibração.
        full_vocab: Lista ordenada de todos os tokens do vocabulário original (ex: Qwen3.5 248K).
        full_merges: Lista de regras de merge (left, right) do tokenizador original.
        protected_tokens: Conjunto de tokens obrigatórios que não podem ser podados.
        target_size: Quantidade exata de tokens no vocabulário final (padrão 98304).

    Returns:
        Tupla (selected_vocab, pruned_merges):
            - selected_vocab: Conjunto de exatamente `target_size` tokens fechado sob merges.
            - pruned_merges: Lista de regras de merge ativas cujos constituintes e resultado
              pertencem a selected_vocab, preservando a ordem original.

    Raises:
        ValueError: Se target_size for menor que os tokens protegidos (e seus ancestrais)
            ou maior que len(full_vocab).
        RuntimeError: Se não for possível atingir rigorosamente target_size.
    """
    if target_size > len(full_vocab):
        raise ValueError(
            f"target_size ({target_size}) é maior que o tamanho do vocabulário original ({len(full_vocab)})."
        )

    dep_graph = build_dependency_graph(full_merges)
    memo: dict[str, set[str]] = {}

    # Inicializa com conjunto protegido e todos os seus ancestrais transitivos
    selected_vocab: set[str] = set(protected_tokens)
    for token in list(protected_tokens):
        selected_vocab.update(compute_ancestors(token, dep_graph, memo))

    if len(selected_vocab) > target_size:
        raise ValueError(
            f"target_size ({target_size}) é menor que o conjunto obrigatório protegido "
            f"e seus ancestrais transitivos ({len(selected_vocab)})."
        )

    if len(selected_vocab) < target_size:
        # Fila de prioridade de candidatos observados: apenas tokens com frequência estritamente positiva (> 0)
        candidates = [
            t for t in full_vocab
            if t not in selected_vocab and frequencies.get(t, 0) > 0
        ]
        candidates.sort(key=lambda t: frequencies[t], reverse=True)

        for token in candidates:
            if len(selected_vocab) == target_size:
                break
            if token in selected_vocab:
                continue

            needed = (compute_ancestors(token, dep_graph, memo) | {token}) - selected_vocab
            if len(selected_vocab) + len(needed) <= target_size:
                selected_vocab.update(needed)

    # Preenchimento residual caso |S| < target_size
    if len(selected_vocab) < target_size:
        remaining = [t for t in full_vocab if t not in selected_vocab]
        while len(selected_vocab) < target_size and remaining:
            best_token = None
            best_needed: set[str] = set()
            best_cost = float("inf")

            for t in remaining:
                needed = (compute_ancestors(t, dep_graph, memo) | {t}) - selected_vocab
                cost = len(needed)
                if 0 < cost <= (target_size - len(selected_vocab)):
                    if cost < best_cost:
                        best_cost = cost
                        best_token = t
                        best_needed = needed
                        if cost == 1:
                            break

            if best_token is None:
                break

            selected_vocab.update(best_needed)
            remaining = [t for t in remaining if t not in selected_vocab]

    if len(selected_vocab) != target_size:
        raise RuntimeError(
            f"Não foi possível atingir o target_size exato de {target_size}; "
            f"selecionados {len(selected_vocab)} tokens."
        )

    # Filtragem das regras de merge: preserva apenas regras cujos operandos e resultado estão no vocabulário
    pruned_merges: list[tuple[str, str]] = []
    for left, right in full_merges:
        res = _merge_result(left, right)
        direct = left + right
        if (direct in selected_vocab or res in selected_vocab):
            if left in selected_vocab and right in selected_vocab:
                pruned_merges.append((left, right))

    return selected_vocab, pruned_merges


def build_remapping_tensors(
    old_vocab: dict[str, int],
    new_vocab: dict[str, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Constrói tensores de projeção de vocabulário para destilação de logits de teacher para student.

    Gera dois tensores 1D em int64:
      1. `new_to_old` (shape: [|V_new|]):
         Mapeia cada índice no vocabulário do student para o índice correspondente no vocabulário do teacher:
             student_logits = teacher_logits[:, new_to_old]
      2. `old_to_new` (shape: [max_old_id + 1]):
         Mapeia cada índice no vocabulário do teacher para o novo índice no student,
         ou -1 caso o token tenha sido podado/descartado:
             old_to_new[old_id] = new_id (se preservado) ou -1 (se descartado)

    Invariante de destilação:
        Para todo new_id em range(|V_new|):
            old_to_new[new_to_old[new_id]] == new_id

    Args:
        old_vocab: Dicionário mapeando token (str) para id original (int).
        new_vocab: Dicionário mapeando token (str) para novo id (int).

    Returns:
        Tupla (new_to_old, old_to_new) de tensores torch.Tensor com dtype torch.int64.
    """
    max_old_id = max(old_vocab.values()) if old_vocab else -1
    old_to_new = torch.full((max_old_id + 1,), -1, dtype=torch.int64)
    new_to_old = torch.zeros(len(new_vocab), dtype=torch.int64)

    for token, new_id in new_vocab.items():
        if token in old_vocab:
            new_to_old[new_id] = old_vocab[token]
        else:
            raise KeyError(f"Token '{token}' de new_vocab não existe em old_vocab.")

    for token, old_id in old_vocab.items():
        if token in new_vocab:
            old_to_new[old_id] = new_vocab[token]

    return new_to_old, old_to_new


def export_pruned_tokenizer(
    source_tokenizer_path: str,
    output_dir: str,
    target_size: int = 98304,
    frequencies_path: str | None = None,
) -> None:
    """Exporta o tokenizador podado no formato HuggingFace com tensores de remapeamento para KD.

    Executa:
      1. Carregamento dos dados do tokenizador fonte (tokenizer.json, configs adicionais se presentes);
      2. Leitura opcional das frequências observadas de calibração;
      3. Identificação e blindagem dos 256 bytes elementares (<0x00> a <0xFF>) e tokens especiais;
      4. Seleção gulosa fechada sob merges BPE com target_size exato via `select_pruned_vocab`;
      5. Ordenação canônica do novo vocabulário, posicionando `<|endoftext|>` em target_size - 1;
      6. Cálculo e salvamento dos tensores de remapeamento `remapping.pt`;
      7. Gravação de `tokenizer.json`, `tokenizer_config.json` e `special_tokens_map.json` em `output_dir`.

    Args:
        source_tokenizer_path: Caminho para diretório HF ou arquivo tokenizer.json.
        output_dir: Diretório de destino para os artefatos podados.
        target_size: Tamanho exato do vocabulário alvo (padrão: 98304).
        frequencies_path: Caminho opcional para arquivo de frequências (JSON, PT ou NPY).
    """
    source_p = Path(source_tokenizer_path)
    if source_p.is_file():
        tokenizer_json_path = source_p
        source_dir = source_p.parent
    elif (source_p / "tokenizer.json").is_file():
        tokenizer_json_path = source_p / "tokenizer.json"
        source_dir = source_p
    else:
        raise FileNotFoundError(
            f"Arquivo tokenizer.json não encontrado em: {source_tokenizer_path}"
        )

    with open(tokenizer_json_path, "r", encoding="utf-8") as f:
        tok_data = json.load(f)

    model = tok_data.get("model", {})
    old_vocab: dict[str, int] = dict(model.get("vocab", {}))
    added_tokens = tok_data.get("added_tokens", [])
    for item in added_tokens:
        if isinstance(item, dict) and "content" in item and "id" in item:
            if item["content"] not in old_vocab:
                old_vocab[item["content"]] = item["id"]

    merges_raw = model.get("merges", [])
    full_merges: list[tuple[str, str]] = []
    is_str_merges = True
    for m in merges_raw:
        if isinstance(m, str):
            parts = m.split(" ")
            full_merges.append((parts[0], parts[1]))
            is_str_merges = True
        elif isinstance(m, (list, tuple)):
            full_merges.append((m[0], m[1]))
            is_str_merges = False

    # Blindagem de tokens obrigatórios (256 bytes e tokens especiais)
    byte_tokens = {f"<0x{i:02X}>" for i in range(256)}
    known_specials = {
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<think>",
        "</think>",
        "<tool_call>",
        "</tool_call>",
        "<tool_response>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|image_pad|>",
    }
    for item in added_tokens:
        if isinstance(item, dict) and "content" in item:
            known_specials.add(item["content"])

    protected_tokens = {t for t in (byte_tokens | known_specials) if t in old_vocab}

    # Ordenação canônica do vocabulário original por id
    full_vocab = sorted(old_vocab.keys(), key=lambda t: old_vocab[t])

    # Leitura opcional de frequências
    frequencies: dict[str, int] = {}
    if frequencies_path is not None:
        freq_p = Path(frequencies_path)
        if not freq_p.exists():
            raise FileNotFoundError(f"Arquivo de frequências não encontrado: {frequencies_path}")

        if freq_p.suffix == ".json":
            with open(freq_p, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                for k, v in loaded.items():
                    if k in old_vocab:
                        frequencies[k] = int(v)
                    elif isinstance(k, int) or (isinstance(k, str) and k.isdigit()):
                        token_id = int(k)
                        if token_id < len(full_vocab):
                            frequencies[full_vocab[token_id]] = int(v)
            elif isinstance(loaded, list):
                for token_id, v in enumerate(loaded):
                    if token_id < len(full_vocab):
                        frequencies[full_vocab[token_id]] = int(v)
        elif freq_p.suffix == ".pt":
            loaded = torch.load(freq_p, weights_only=True)
            if isinstance(loaded, torch.Tensor):
                counts = loaded.tolist()
                for token_id, v in enumerate(counts):
                    if token_id < len(full_vocab):
                        frequencies[full_vocab[token_id]] = int(v)
            elif isinstance(loaded, dict):
                for k, v in loaded.items():
                    if isinstance(k, str) and k in old_vocab:
                        frequencies[k] = int(v)
                    elif isinstance(k, int) or (isinstance(k, str) and k.isdigit()):
                        token_id = int(k)
                        if token_id < len(full_vocab):
                            frequencies[full_vocab[token_id]] = int(v)
        elif freq_p.suffix == ".npy":
            import numpy as np

            counts = np.load(freq_p)
            for token_id, v in enumerate(counts):
                if token_id < len(full_vocab):
                    frequencies[full_vocab[token_id]] = int(v)
        else:
            raise ValueError(f"Extensão de arquivo de frequências não suportada: {freq_p.suffix}")

    # Seleção gulosa fechada sob merges BPE
    selected_vocab, pruned_merges = select_pruned_vocab(
        frequencies=frequencies,
        full_vocab=full_vocab,
        full_merges=full_merges,
        protected_tokens=protected_tokens,
        target_size=target_size,
    )

    # Atribuição contígua de novos IDs com posição canônica de <|endoftext|> em target_size - 1
    new_vocab: dict[str, int] = {}
    if "<|endoftext|>" in selected_vocab:
        regular = sorted(
            [t for t in selected_vocab if t != "<|endoftext|>"],
            key=lambda t: old_vocab[t],
        )
        for i, t in enumerate(regular):
            new_vocab[t] = i
        new_vocab["<|endoftext|>"] = target_size - 1
    else:
        regular = sorted(selected_vocab, key=lambda t: old_vocab[t])
        for i, t in enumerate(regular):
            new_vocab[t] = i

    # Atualização da estrutura JSON do tokenizador
    tok_data["model"]["vocab"] = new_vocab
    if is_str_merges:
        tok_data["model"]["merges"] = [f"{l} {r}" for l, r in pruned_merges]
    else:
        tok_data["model"]["merges"] = [[l, r] for l, r in pruned_merges]

    new_added_tokens = []
    for item in added_tokens:
        if isinstance(item, dict) and "content" in item:
            c = item["content"]
            if c in new_vocab:
                item_copy = dict(item)
                item_copy["id"] = new_vocab[c]
                new_added_tokens.append(item_copy)
    tok_data["added_tokens"] = new_added_tokens

    # Validação da estrutura serializada com o backend HuggingFace
    tok_obj = Tokenizer.from_str(json.dumps(tok_data))

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    # 1. Exportação do tokenizer.json
    tok_obj.save(str(out_p / "tokenizer.json"))

    # 2. Exportação dos tensores de remapeamento para KD
    new_to_old, old_to_new = build_remapping_tensors(old_vocab, new_vocab)
    torch.save(
        {"new_to_old": new_to_old, "old_to_new": old_to_new},
        out_p / "remapping.pt",
    )

    # 3. Exportação do tokenizer_config.json
    tok_config_src = source_dir / "tokenizer_config.json"
    if tok_config_src.is_file():
        with open(tok_config_src, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        cfg["vocab_size"] = target_size
        if "added_tokens_decoder" in cfg and isinstance(cfg["added_tokens_decoder"], dict):
            new_decoder = {}
            for old_id_str, meta in cfg["added_tokens_decoder"].items():
                if isinstance(meta, dict) and "content" in meta:
                    c = meta["content"]
                    if c in new_vocab:
                        new_id = new_vocab[c]
                        meta_copy = dict(meta)
                        meta_copy["id"] = new_id
                        new_decoder[str(new_id)] = meta_copy
            cfg["added_tokens_decoder"] = new_decoder
    else:
        cfg = {
            "add_prefix_space": False,
            "added_tokens_decoder": {},
            "bos_token": None,
            "clean_up_tokenization_spaces": False,
            "eos_token": "<|endoftext|>",
            "model_max_length": 32768,
            "pad_token": "<|endoftext|>",
            "tokenizer_class": "Qwen2TokenizerFast",
            "vocab_size": target_size,
        }
        if "<|endoftext|>" in new_vocab:
            cfg["added_tokens_decoder"][str(new_vocab["<|endoftext|>"])] = {
                "content": "<|endoftext|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True,
            }
    with open(out_p / "tokenizer_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    # 4. Exportação do special_tokens_map.json
    spec_src = source_dir / "special_tokens_map.json"
    if spec_src.is_file():
        with open(spec_src, "r", encoding="utf-8") as f:
            spec = json.load(f)
    else:
        spec = {
            "eos_token": "<|endoftext|>",
            "pad_token": "<|endoftext|>",
        }
    with open(out_p / "special_tokens_map.json", "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2, ensure_ascii=False)

