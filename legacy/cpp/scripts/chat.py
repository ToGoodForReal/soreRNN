#!/usr/bin/env python3
"""
Inferência Interativa e Geração de Texto em Tempo O(1)
para o Modelo soreRNN-LM v2 (152.3M Parâmetros / 12 Camadas).
Implementa Conv1D, RG-LRU, GELU MLP, Weight Tying e Amostragem com Repetition Penalty.
"""

import os
import sys
import argparse
import struct
import numpy as np
from tokenizer_utils import get_tokenizer

class StackedRNNGenerator:
    def __init__(
        self,
        checkpoint_path: str,
        vocab_size: int = 50257,
        d_model: int = 1024,
        num_layers: int = 12,
        d_mlp: int = 2560,
        conv_kernel: int = 4,
        eps: float = 1e-5,
        tokenizer_name: str = "pt"
    ):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint não encontrado em: {checkpoint_path}")

        self.tokenizer_name = tokenizer_name
        self.eps = eps

        with open(checkpoint_path, "rb") as f:
            magic = f.read(4)
            if magic == b"SORE":
                (version, header_size, v_size, d_m, n_l, d_m_mlp, c_k, tie, eps_val,
                 step, tokens, cursor, lr, has_opt, num_p) = struct.unpack("<IIIIIIII f QQQ f II", f.read(68))
                vocab_size = v_size
                d_model = d_m
                num_layers = n_l
                d_mlp = d_m_mlp
                conv_kernel = c_k
                eps = eps_val
                print(f"[soreRNN-LM] Header SORE v{version} detectado: L={num_layers}, D={d_model}, MLP={d_mlp}, Passo={step}")
                f.seek(header_size)
                raw = np.fromfile(f, dtype=np.float32)
            else:
                f.seek(0)
                file_size_bytes = os.path.getsize(checkpoint_path)
                total_floats = file_size_bytes // 4
                if d_model == 1024 and num_layers == 12:
                    if total_floats > 250_000_000:
                        d_model = 1280
                        num_layers = 18
                        d_mlp = 3200
                    else:
                        d_model = 1024
                        num_layers = 12
                        d_mlp = 2560
                raw = np.fromfile(f, dtype=np.float32)
        offset = 0

        def read_tensor(shape):
            nonlocal offset
            numel = int(np.prod(shape))
            if offset + numel > len(raw):
                raise ValueError(f"Fim de arquivo prematuro lendo tensor {shape}: precisa {numel}, resta {len(raw) - offset}")
            tensor = raw[offset : offset + numel].reshape(shape)
            offset += numel
            return tensor

        # 1. Embedding e LM Head (Weight Tied)
        self.w_emb = read_tensor((vocab_size, d_model))
        self.b_head = read_tensor((vocab_size,))
        self.final_norm_gamma = read_tensor((d_model,))

        # 2. 12 Camadas
        self.layers = []
        for l in range(num_layers):
            layer = {
                # Bloco Recorrente + Conv1D
                "norm1_gamma": read_tensor((d_model,)),
                "w_conv": read_tensor((d_model, conv_kernel)),
                "b_conv": read_tensor((d_model,)),
                "w_gate": read_tensor((d_model, d_model)),
                "b_gate": read_tensor((d_model,)),
                "w_in": read_tensor((d_model, d_model)),
                "b_in": read_tensor((d_model,)),
                "w_out": read_tensor((d_model, d_model)),
                "b_out": read_tensor((d_model,)),
                # Bloco MLP
                "norm2_gamma": read_tensor((d_model,)),
                "w_mlp1": read_tensor((d_mlp, d_model)),
                "b_mlp1": read_tensor((d_mlp,)),
                "w_mlp2": read_tensor((d_model, d_mlp)),
                "b_mlp2": read_tensor((d_model,)),
            }
            self.layers.append(layer)

        self.enc = get_tokenizer(self.tokenizer_name)

        print(f"[soreRNN-LM v2] Checkpoint carregado: {checkpoint_path}")
        print(f" -> Arquitetura: {num_layers} camadas | d_model={d_model} | d_mlp={d_mlp} | K={conv_kernel}")
        print(f" -> Weight Tying: Sim (w_head compartilhado com w_emb)")
        print(f" -> Parâmetros lidos: {offset:,} floats (~{offset * 4 / (1024*1024):.2f} MB)")
        print(f" -> Tokenizer: {self.tokenizer_name} (vocab={self.enc.vocab_size})")

    def rms_norm(self, x: np.ndarray, gamma: np.ndarray) -> np.ndarray:
        rms = np.sqrt(np.mean(x ** 2) + self.eps)
        return (x / rms) * gamma

    def gelu(self, x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))

    def step(self, token: int, h_states: list, conv_buffers: list):
        """Passo O(1) através das 12 camadas empilhadas com Conv1D, RG-LRU e MLP."""
        x = self.w_emb[token]

        for l in range(self.num_layers):
            layer = self.layers[l]

            # --- Bloco 1: Recorrência + Conv1D ---
            x_norm1 = self.rms_norm(x, layer["norm1_gamma"])

            # Causal Depthwise Conv1D
            buf = conv_buffers[l] # [3, D]
            w_c = layer["w_conv"] # [D, 4]
            # lag 3: buf[0], lag 2: buf[1], lag 1: buf[2], lag 0: x_norm1
            x_conv = (
                buf[0] * w_c[:, 0] +
                buf[1] * w_c[:, 1] +
                buf[2] * w_c[:, 2] +
                x_norm1 * w_c[:, 3] +
                layer["b_conv"]
            )
            # Atualiza rolling buffer de convolução
            buf[0] = buf[1]
            buf[1] = buf[2]
            buf[2] = x_norm1

            # Gate & Sigmoid
            g = layer["w_gate"] @ x_conv + layer["b_gate"]
            a = 1.0 / (1.0 + np.exp(-np.clip(g, -20.0, 20.0)))

            # Projeção de Entrada & Recorrência Linear
            u = layer["w_in"] @ x_conv + layer["b_in"]
            h_states[l] = a * h_states[l] + u

            # Projeção de Saída & Conexão Residual 1
            y_rnn = layer["w_out"] @ h_states[l] + layer["b_out"]
            x = x + y_rnn

            # --- Bloco 2: MLP Channel Mixing ---
            x_norm2 = self.rms_norm(x, layer["norm2_gamma"])
            m1 = layer["w_mlp1"] @ x_norm2 + layer["b_mlp1"]
            # Fast GELU aproximado: 0.5 * x * (1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
            m1_act = 0.5 * m1 * (1.0 + np.tanh(0.79788456 * (m1 + 0.044715 * m1 ** 3)))
            y_mlp = layer["w_mlp2"] @ m1_act + layer["b_mlp2"]

            # Conexão Residual 2
            x = x + y_mlp

        # RMSNorm Final
        x_final = self.rms_norm(x, self.final_norm_gamma)

        # LM Head Logits com Weight Tying (w_emb)
        logits = self.w_emb @ x_final + self.b_head
        return logits, h_states, conv_buffers

    def sample_token(
        self,
        logits: np.ndarray,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
        context_tokens: list = None
    ):
        logits = logits.copy()

        # Repetition Penalty
        if context_tokens and repetition_penalty != 1.0:
            for prev_tok in set(context_tokens[-30:]):
                if logits[prev_tok] > 0:
                    logits[prev_tok] /= repetition_penalty
                else:
                    logits[prev_tok] *= repetition_penalty

        if temperature <= 0.0:
            return int(np.argmax(logits))

        logits = logits / max(temperature, 1e-4)

        # Top-K
        if top_k > 0 and top_k < len(logits):
            top_indices = np.argpartition(logits, -top_k)[-top_k:]
            mask = np.zeros_like(logits, dtype=bool)
            mask[top_indices] = True
            logits[~mask] = -1e9

        # Softmax
        max_logit = np.max(logits)
        exp_logits = np.exp(logits - max_logit)
        probs = exp_logits / np.sum(exp_logits)

        # Top-P (Nucleus)
        if top_p < 1.0:
            sorted_indices = np.argsort(probs)[::-1]
            sorted_probs = probs[sorted_indices]
            cum_probs = np.cumsum(sorted_probs)
            cutoff = cum_probs > top_p
            if np.any(cutoff):
                cutoff_idx = np.where(cutoff)[0][0] + 1
                sorted_probs[cutoff_idx:] = 0.0
                sorted_probs = sorted_probs / np.sum(sorted_probs)
                probs = np.zeros_like(probs)
                probs[sorted_indices] = sorted_probs

        sum_probs = np.sum(probs)
        if sum_probs == 0 or np.isnan(sum_probs):
            return int(np.argmax(logits))

        probs = probs / sum_probs
        return int(np.random.choice(len(probs), p=probs))

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 60,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15
    ):
        tokens = self.enc.encode(prompt, allowed_special={"<|endoftext|>"})
        h_states = [np.zeros(self.d_model, dtype=np.float32) for _ in range(self.num_layers)]
        conv_buffers = [np.zeros((3, self.d_model), dtype=np.float32) for _ in range(self.num_layers)]

        # Prefill do prompt
        logits = None
        for tok in tokens:
            logits, h_states, conv_buffers = self.step(tok, h_states, conv_buffers)

        generated_tokens = []
        current_tok = tokens[-1] if tokens else 0
        all_tokens = list(tokens)
        for _ in range(max_new_tokens):
            if logits is None:
                logits, h_states, conv_buffers = self.step(current_tok, h_states, conv_buffers)

            next_tok = self.sample_token(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                context_tokens=all_tokens
            )
            generated_tokens.append(next_tok)
            all_tokens.append(next_tok)

            word = self.enc.decode([next_tok])
            print(word, end="", flush=True)
            if "<|endoftext|>" in word or "<|user|>" in word or "</s>" in word:
                break

            current_tok = next_tok
            logits, h_states, conv_buffers = self.step(current_tok, h_states, conv_buffers)

        print()
        return self.enc.decode(generated_tokens)

def interactive_chat(checkpoint_path: str):
    bot = StackedRNNGenerator(checkpoint_path)
    is_sft = "sft" in checkpoint_path.lower()
    mode_name = "SFT Instruído (Chatbot)" if is_sft else "Base Pretrain (Completador de Texto e Raciocínio)"
    print("\n" + "=" * 65)
    print(f"  CHAT INTERATIVO - soreRNN-LM ({mode_name})")
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

            if not is_sft:
                if any(kw in user_input.lower() for kw in ["calcule", "quanto", "problema", "some", "multiplique"]):
                    if not user_input.startswith("Problema:"):
                        formatted_prompt = f"Problema: {user_input}\nRaciocínio:"
                    else:
                        formatted_prompt = user_input + "\nRaciocínio:"
                    print("soreRNN [Raciocínio]: ", end="", flush=True)
                else:
                    formatted_prompt = user_input
                    print("soreRNN [Completando]: ", end="", flush=True)
            else:
                formatted_prompt = f"<|user|>\n{user_input}\n<|assistant|>\n"
                print("soreRNN: ", end="", flush=True)

            bot.generate(formatted_prompt, max_new_tokens=80, temperature=0.7, repetition_penalty=1.15)
            print()
        except (KeyboardInterrupt, EOFError):
            print("\nEncerrando...")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inferência da soreRNN-LM")
    default_ckpt = "checkpoints/sore_lm_300m_final.bin"
    if not os.path.exists(default_ckpt):
        default_ckpt = "checkpoints/sore_lm_150m_final.bin"
    if not os.path.exists(default_ckpt):
        default_ckpt = "checkpoints/sore_lm_300m_pretrain.bin"

    parser.add_argument("--checkpoint", type=str, default=default_ckpt)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--max_tokens", type=int, default=80)
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--rep_penalty", type=float, default=1.15)
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        for fallback in [
            "checkpoints/sore_lm_300m_final.bin",
            "checkpoints/sore_lm_300m_pretrain.bin",
            "checkpoints/sore_lm_150m_final.bin",
            "checkpoints/sore_lm_150m_pretrain.bin",
        ]:
            if os.path.exists(fallback):
                args.checkpoint = fallback
                break

    if args.prompt:
        bot = StackedRNNGenerator(args.checkpoint)
        bot.generate(args.prompt, max_new_tokens=args.max_tokens, temperature=args.temp, repetition_penalty=args.rep_penalty)
    else:
        interactive_chat(args.checkpoint)
