#!/usr/bin/env python3
"""
soreRNN-LM: Módulo Unificado de Tokenização
Suporta:
- Tokenizer Nativo em Português: 'pierreguillou/gpt2-small-portuguese' (vocab=50257, 40% melhor compressão para PT-BR)
- Tokenizer Multilíngue LLaMA: 'openlm-research/open_llama_3b_v2' (vocab=32000)
- Tokenizer Padrão GPT-2 (tiktoken): (vocab=50257)
"""

import os
import sys
from typing import List, Optional, Union

class SoreTokenizer:
    def __init__(self, model_type: str = "pt", model_path: Optional[str] = None):
        self.model_type = model_type.lower()
        self._hf_tok = None
        self._tiktoken = None
        self._eos_token_id = 50256
        self._eos_token = "<|endoftext|>"
        self._vocab_size = 50257

        if self.model_type in ("pt", "portuguese", "pt-br"):
            target_id = model_path or "pierreguillou/gpt2-small-portuguese"
            try:
                from transformers import AutoTokenizer
                self._hf_tok = AutoTokenizer.from_pretrained(target_id)
                self._eos_token = self._hf_tok.eos_token or "<|endoftext|>"
                self._eos_token_id = self._hf_tok.eos_token_id if self._hf_tok.eos_token_id is not None else 0
                self._vocab_size = len(self._hf_tok)
            except Exception as e:
                print(f"[Aviso Tokenizer] Não foi possível carregar '{target_id}': {e}. Usando fallback tiktoken gpt2.", file=sys.stderr)
                import tiktoken
                self._tiktoken = tiktoken.get_encoding("gpt2")
                self._eos_token = "<|endoftext|>"
                self._eos_token_id = 50256
                self._vocab_size = 50257

        elif self.model_type in ("llama", "open_llama"):
            target_id = model_path or "openlm-research/open_llama_3b_v2"
            try:
                from transformers import AutoTokenizer
                self._hf_tok = AutoTokenizer.from_pretrained(target_id)
                self._eos_token = self._hf_tok.eos_token or "</s>"
                self._eos_token_id = self._hf_tok.eos_token_id if self._hf_tok.eos_token_id is not None else 2
                self._vocab_size = len(self._hf_tok)
            except Exception as e:
                print(f"[Aviso Tokenizer] Falha ao carregar LLaMA ({e}), fallback para tiktoken.", file=sys.stderr)
                import tiktoken
                self._tiktoken = tiktoken.get_encoding("gpt2")
                self._eos_token_id = 50256
                self._vocab_size = 50257

        elif self.model_type in ("gpt2", "tiktoken"):
            import tiktoken
            self._tiktoken = tiktoken.get_encoding("gpt2")
            self._eos_token = "<|endoftext|>"
            self._eos_token_id = 50256
            self._vocab_size = 50257

        else:
            # Caminho genérico ou modelo HF
            from transformers import AutoTokenizer
            self._hf_tok = AutoTokenizer.from_pretrained(model_path or model_type)
            self._eos_token = self._hf_tok.eos_token or "<|endoftext|>"
            self._eos_token_id = self._hf_tok.eos_token_id if self._hf_tok.eos_token_id is not None else 0
            self._vocab_size = len(self._hf_tok)

    @property
    def eos_token_id(self) -> int:
        return self._eos_token_id

    @property
    def eos_token(self) -> str:
        return self._eos_token

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def encode(self, text: str, allowed_special: Optional[set] = None) -> List[int]:
        if self._hf_tok is not None:
            # Hugging Face AutoTokenizer
            return self._hf_tok.encode(text, add_special_tokens=False)
        else:
            # Tiktoken
            allowed = allowed_special if allowed_special is not None else {self._eos_token}
            return self._tiktoken.encode(text, allowed_special=allowed)

    def decode(self, tokens: List[int]) -> str:
        if self._hf_tok is not None:
            return self._hf_tok.decode(tokens)
        else:
            return self._tiktoken.decode(tokens)

_CACHED_TOKENIZERS = {}

def get_tokenizer(name: str = "pt", model_path: Optional[str] = None) -> SoreTokenizer:
    key = (name.lower(), model_path)
    if key not in _CACHED_TOKENIZERS:
        _CACHED_TOKENIZERS[key] = SoreTokenizer(model_type=name, model_path=model_path)
    return _CACHED_TOKENIZERS[key]

if __name__ == "__main__":
    tok = get_tokenizer("pt")
    print(f"Tokenizer carregado: tipo=pt, vocab_size={tok.vocab_size}, eos_id={tok.eos_token_id}")
    sample = "Treinamento do modelo soreRNN com vocabulário adaptado para português."
    encoded = tok.encode(sample)
    decoded = tok.decode(encoded)
    print(f"Texto original ({len(sample)} chars): {sample}")
    print(f"Tokens ({len(encoded)}): {encoded}")
    print(f"Decodificado: {decoded}")
