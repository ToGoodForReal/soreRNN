"""Modelo de FLOPs de treino por token (forward + backward), base do MFU.

`flops_per_token(cfg, T)` soma três termos:

- **matmuls com pesos:** `6 · (P_mm + (1 + mtp_layers) · V · D)`. `P_mm` (`matmul_params`) é a soma
  dos pesos 2-D de todos os `nn.Linear` de `blocks` e `mtp`: projeções da GDN e da atenção, SwiGLU,
  `mtp.proj` e, com `residual="hyper"`, as projeções do hyper-residual. Embedding (lookup, sem
  matmul) não entra. A cabeça `V · D` conta uma vez por uso: a loss do LM e a do MTP.
- **atenção:** `6 · n_attn · attn_heads · attn_head_dim · T`. `QKᵀ` e `AV` custam
  `2 · 2 · H · Dh · T` por token no forward com máscara cheia; a causal corta pela metade e o
  backward dobra o forward (fwd + bwd = 3 × fwd). `n_attn` inclui a camada de atenção do MTP. Com
  documentos empacotados a máscara de documento faz menos que isso: o termo é um teto.
- **GDN:** `6 · n_gdn · Hv · Dk · Dv · 2`, independente de `T`. Cada head de v tem um estado
  `Dk × Dv` (`Dk = Dv = gdn_head_dim`) e, por token, faz cerca de duas operações de `2 · Dk · Dv`
  com ele (o produto com `k` do delta rule e o produto com `q` da saída); fwd + bwd = 3 × fwd.

Convenção do MFU (PaLM): o forward recomputado pelo grad checkpoint não conta, e a conv curta, as
normas e os gates elemento a elemento ficam de fora.
"""
from __future__ import annotations

from sore.model.config import ModelConfig


def _gdn_params(cfg: ModelConfig) -> int:
    d, hv = cfg.d_model, cfg.gdn_v_heads
    qk = cfg.gdn_qk_heads * cfg.gdn_head_dim
    v = hv * cfg.gdn_head_dim
    return d * (2 * qk + 2 * v + 2 * hv) + v * d  # q, k, v, g, a, b + o


def _attn_params(cfg: ModelConfig) -> int:
    d = cfg.d_model
    q = cfg.attn_heads * cfg.attn_head_dim
    kv = cfg.attn_kv_heads * cfg.attn_head_dim
    return d * (2 * q + 2 * kv) + q * d  # q, gate, k, v + o


def _ffn_params(cfg: ModelConfig) -> int:
    return 3 * cfg.d_model * cfg.ffn_dim  # gate, up, down


def _residual_params(cfg: ModelConfig) -> int:
    """Pesos 2-D de um residual do tronco: zero no plain; `down`, `up` e `write_proj` no hyper."""
    if cfg.residual == "plain":
        return 0
    d, n, r = cfg.d_model, cfg.hc_branches, cfg.hc_rank
    return d * r + r * n * d + d * n


def matmul_params(cfg: ModelConfig) -> int:
    """Soma dos pesos 2-D dos `nn.Linear` de `blocks` e `mtp` (sem embedding nem cabeça de LM)."""
    mixer = {"gdn": _gdn_params(cfg), "attn": _attn_params(cfg)}
    per_block = _ffn_params(cfg) + 2 * _residual_params(cfg)  # 2 residuais por bloco
    trunk = sum(mixer[t] + per_block for t in cfg.layer_types())
    # MTP: proj [2D -> D] e um Block de atenção com residual plain
    mtp = cfg.mtp_layers * (2 * cfg.d_model * cfg.d_model + _attn_params(cfg) + _ffn_params(cfg))
    return trunk + mtp


def flops_per_token(cfg: ModelConfig, seq_len: int) -> float:
    """FLOPs de treino (fwd + bwd) por token com contexto `seq_len`; fórmula no docstring do módulo."""
    types = cfg.layer_types()
    n_attn = types.count("attn") + cfg.mtp_layers
    n_gdn = types.count("gdn")
    head = (1 + cfg.mtp_layers) * cfg.vocab_size * cfg.d_model
    dense = 6 * (matmul_params(cfg) + head)
    attn = 6 * n_attn * cfg.attn_heads * cfg.attn_head_dim * seq_len
    gdn = 6 * n_gdn * cfg.gdn_v_heads * cfg.gdn_head_dim * cfg.gdn_head_dim * 2
    return float(dense + attn + gdn)
