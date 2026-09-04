#!/usr/bin/env python3
"""
Inferência Interativa e Geração de Texto em Tempo O(1)
para o Modelo RNN Linear (Griffin / soreRNN).
"""

import os
import sys
import argparse
import numpy as np
import tiktoken

class LinearRNNGenerator:
    def __init__(self, checkpoint_path: str, vocab_size: int = 50257, emb_dim: int = 32, hidden_dim: int = 64):
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.enc = tiktoken.get_encoding("gpt2")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint não encontrado em: {checkpoint_path}")

        raw = np.fromfile(checkpoint_path, dtype=np.float32)
        offset = 0

        # Carregamento na ordem exata salva pelo train_sequential.cpp
        def read_tensor(shape):
            nonlocal offset
            numel = int(np.prod(shape))
            tensor = raw[offset : offset + numel].reshape(shape)
            offset += numel
            return tensor

        self.w_emb = read_tensor((vocab_size, emb_dim))
        self.w_head = read_tensor((vocab_size, emb_dim))
        self.b_head = read_tensor((vocab_size,))

        self.w_gate = read_tensor((hidden_dim, emb_dim))
        self.b_gate = read_tensor((hidden_dim,))
        self.w_in = read_tensor((hidden_dim, emb_dim))
        self.b_in = read_tensor((hidden_dim,))
        self.w_out = read_tensor((emb_dim, hidden_dim))
        self.b_out = read_tensor((emb_dim,))

        print(f"[Modelo] Checkpoint carregado: {checkpoint_path}")
        print(f" -> Vocabulário: {vocab_size} | Emb Dim: {emb_dim} | Hidden Dim: {hidden_dim}")
        print(f" -> Total de Parâmetros: {offset:,} floats (~{offset * 4 / (1024*1024):.2f} MB)")

    def step(self, token: int, h_prev: np.ndarray):
        """Passo de inferência autoregressiva da RNN Linear em tempo O(1) e memória O(1)."""
        x_emb = self.w_emb[token] # [emb_dim]

        # 1. Decay Gate: a = sigmoid(W_gate * x + b_gate)
        g = self.w_gate @ x_emb + self.b_gate
        a = 1.0 / (1.0 + np.exp(-np.clip(g, -20.0, 20.0)))

        # 2. Input Candidate: u = W_in * x + b_in
        u = self.w_in @ x_emb + self.b_in

        # 3. Recorrência Linear Real-Gated: h = a * h_prev + u
        h = a * h_prev + u

        # 4. Projeção de Saída: y = W_out * h + b_out
        y = self.w_out @ h + self.b_out

        # 5. Logits do Vocabulário: logits = W_head * y + b_head
        logits = self.w_head @ y + self.b_head

        return logits, h

    def sample_token(self, logits: np.ndarray, temperature: float = 0.8, top_k: int = 40):
        if temperature <= 0.0:
            return int(np.argmax(logits))

        # Estabilidade numérica
        logits = logits / max(temperature, 1e-4)
        max_logit = np.max(logits)
        exp_logits = np.exp(logits - max_logit)

        # Top-K filtering
        if top_k > 0 and top_k < len(logits):
            top_indices = np.argpartition(logits, -top_k)[-top_k:]
            mask = np.zeros_like(logits, dtype=bool)
            mask[top_indices] = True
            exp_logits[~mask] = 0.0

        probs = exp_logits / np.sum(exp_logits)
        return int(np.random.choice(len(probs), p=probs))

    def generate(self, prompt: str, max_new_tokens: int = 50, temperature: float = 0.8, top_k: int = 40):
        tokens = self.enc.encode(prompt, allowed_special={"<|endoftext|>"})
        h = np.zeros(self.hidden_dim, dtype=np.float32)

        # Pré-alimentação (Prefill) do prompt na memória oculta h
        logits = None
        for tok in tokens:
            logits, h = self.step(tok, h)

        generated_tokens = []
        current_tok = tokens[-1] if tokens else 0

        print(prompt, end="", flush=True)

        for _ in range(max_new_tokens):
            if logits is None:
                logits, h = self.step(current_tok, h)

            next_tok = self.sample_token(logits, temperature=temperature, top_k=top_k)
            generated_tokens.append(next_tok)

            word = self.enc.decode([next_tok])
            print(word, end="", flush=True)

            if word in ["<|endoftext|>", "<|user|>"]:
                break

            current_tok = next_tok
            logits, h = self.step(current_tok, h)

        print()
        return self.enc.decode(generated_tokens)

def interactive_chat(checkpoint_path: str):
    bot = LinearRNNGenerator(checkpoint_path)
    print("\n" + "=" * 65)
    print("  CHAT INTERATIVO - RNN LINEAR soreRNN (Tempo O(1) por Token)")
    print("  (Digite 'sair' para encerrar)")
    print("=" * 65 + "\n")

    while True:
        try:
            user_input = input("Você: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["sair", "exit", "quit"]:
                print("Até mais!")
                break

            formatted_prompt = f"<|user|>\n{user_input}\n<|assistant|>\n"
            print("soreRNN: ", end="", flush=True)
            bot.generate(formatted_prompt, max_new_tokens=40, temperature=0.7)
            print()
        except (KeyboardInterrupt, EOFError):
            print("\nEncerrando...")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inferência da RNN Linear")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_chat_final.bin")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--max_tokens", type=int, default=50)
    parser.add_argument("--temp", type=float, default=0.7)
    args = parser.parse_args()

    if args.prompt:
        bot = LinearRNNGenerator(args.checkpoint)
        bot.generate(args.prompt, max_new_tokens=args.max_tokens, temperature=args.temp)
    else:
        interactive_chat(args.checkpoint)
