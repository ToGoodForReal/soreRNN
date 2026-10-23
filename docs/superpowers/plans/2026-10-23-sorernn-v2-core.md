# soreRNN v2 — Núcleo do Modelo e Loop de Treino: Plano de Implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pacote PyTorch `sore/` com o modelo soreQ4 (GDN + atenção com gate, 3:1, MTP, hyper-residual opcional), loss em chunks, Muon+AdamW, schedule WSD, loader empacotado com reset por documento, trainer retomável e benchmark. Tudo validado no modo `tiny` na RTX 3050 (WSL2) e pronto para rodar o preset `b0_8` na 4090.

**Architecture:** Cada camada é `Block(norm → mixer → residual, norm → SwiGLU → residual)`. O mixer é `GatedDeltaNet` (backend `fla` ou `torch` de referência) ou `GatedAttention` (backend `sdpa` ou `flex`). Os lotes são empacotados como `[1, N]` com `cu_seqlens`, e todo estado (recorrência, conv curta, atenção, MTP) reseta na fronteira de documento. O treino usa autocast BF16 com pesos FP32, checkpointing por bloco, Muon nas matrizes ocultas e AdamW no resto.

**Tech Stack:** Python 3.12, PyTorch, flash-linear-attention (Triton), numpy, pyyaml, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-10-23-sorernn-v2-design.md`

## Global Constraints

- Todos os comandos rodam no WSL `Ubuntu-24.04` (usuário `wellington`), em `/mnt/c/Users/wellington.lara/Documents/GitHub/soreRNN`, com `export PATH=$HOME/.local/bin:$PATH UV_PROJECT_ENVIRONMENT=$HOME/.venvs/sorernn`. Testes rodam com `uv run pytest`.
- `requires-python = ">=3.12,<3.13"`. Dependências de runtime: `torch`, `flash-linear-attention`, `numpy`, `pyyaml`. Dev: `pytest`. Nada além disso sem perguntar. As versões ficam travadas em `uv.lock` (Task 1). Versões verificadas no WSL com a 3050 em 2026-10-23: `torch==2.14.1` (wheel cu130), `triton==3.8.0`, `flash-linear-attention==0.5.2`.
- A suíte passa em CPU sem Triton e sem FLA. Testes que precisam de GPU usam `@pytest.mark.cuda`; os que precisam de FLA usam `@pytest.mark.fla`. Ambos são pulados automaticamente quando o recurso não existe.
- Tokens no disco são `uint32` little-endian. `ignore_index = -100` em todo lugar.
- Layout empacotado: `input_ids` tem shape `[1, N]`. `cu_seqlens` é `torch.long` (int64, padrão do FLA 0.5.2) com shape `[S+1]`, começa em 0, termina em N, é estritamente crescente e não tem segmentos vazios.
- Identificadores em inglês; comentários e docs em português (padrão do repo).
- Commits no formato `feat:`, `chore:`, `test:`, terminando com `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Vazamento entre documentos empacotados no modelo inteiro** (conv, GDN, atenção e MTP juntos). Um documento empacotado deve produzir os mesmos logits que o mesmo documento sozinho. Teste em Task 7.
2. **Segmentos curtos e comprimentos não múltiplos de 64** (chunk do FLA), por exemplo `cu_seqlens=[0,1,2,100,257]`. O FLA deve bater com a referência. Teste em Task 4.
3. **Retomada logo depois de uma troca de estágio de contexto** (seq_len 128 → 256). Os losses devem bater com uma execução sem interrupção. Teste em Task 10.
4. **EOS consecutivos e documento de 1 token**: nunca podem gerar segmento vazio em `cu_seqlens`. Teste em Task 9.
5. **Entradas extremas no gating do GDN** (x×100): as saídas devem continuar finitas. Teste em Task 4.

---

### Task 1: Reestruturação do repo e esqueleto Python

**Files:**
- Move (`git mv`): `CMakeLists.txt include src tests examples scripts` → `legacy/cpp/`
- Create: `pyproject.toml`, `sore/__init__.py`, `tests/conftest.py`, `tests/test_smoke.py`, `uv.lock`, `.gitattributes` (`* text=auto eol=lf` — o checkout é compartilhado entre Windows e WSL)
- Modify: `README.md` (reescrita curta: stack v2, setup WSL, ponteiro para `legacy/cpp`), `.gitignore` (+ `.venv/ runs/ data/ *.pt`)

**Interfaces:**
- Produces: pacote `sore` com `__version__ = "0.2.0.dev0"`. Markers `cuda` e `fla` registrados em `conftest.py`, com skip automático: `cuda` exige `torch.cuda.is_available()`; `fla` exige isso e `import fla` funcionando. Fixture `device` devolve `"cuda"` se houver GPU, senão `"cpu"`.

- [ ] **Step 1:** `git mv` dos diretórios C++ para `legacy/cpp/`. Rodar `git status` e confirmar que só aparecem renames.
- [ ] **Step 2:** Escrever `tests/test_smoke.py::test_import` (`import sore; assert sore.__version__ == "0.2.0.dev0"`) e um teste `@pytest.mark.cuda` `test_cuda_bf16_matmul` (matmul BF16 512×512 finito).
- [ ] **Step 3:** Criar `pyproject.toml` (build backend hatchling, pacote `sore`, `[tool.pytest.ini_options] markers = ["cuda", "fla"]`) e rodar `uv lock && uv sync`.
- [ ] **Step 4:** Rodar `uv run pytest -q`. Esperado: 2 passed no WSL com a 3050.
- [ ] **Step 5:** Commit `chore: move C++ engine to legacy/cpp and scaffold sore v2 package`.

### Task 2: Config, presets e primitivas (RMSNorm, SwiGLU, varlen helpers)

**Files:**
- Create: `sore/model/__init__.py`, `sore/model/config.py`, `sore/model/layers.py`, `sore/model/varlen.py`
- Test: `tests/model/test_config.py`, `tests/model/test_varlen.py`

**Interfaces:**
- Produces:
  - `ModelConfig` (dataclass) com os campos:
    - Dimensões: `vocab_size, d_model, n_layers, ffn_dim`.
    - Layout: `layer_pattern: tuple[str, ...] = ("gdn","gdn","gdn","attn")`.
    - GDN: `gdn_qk_heads, gdn_v_heads, gdn_head_dim=128, conv_kernel=4`.
    - Atenção: `attn_heads, attn_kv_heads, attn_head_dim=128, rope_theta=1e7, rope_partial=0.25`.
    - Residual: `residual: Literal["plain","hyper"]="plain", hc_branches=4, hc_rank=128`.
    - MTP: `mtp_layers: int = 1, mtp_weight: float = 0.3`.
    - Gerais: `tie_embeddings=True, norm_eps=1e-6, init_std=0.02`.
    - Execução: `gdn_backend: Literal["fla","torch"]="fla", attn_backend: Literal["sdpa","flex"]="sdpa", grad_checkpoint=False`.
  - Métodos de `ModelConfig`: `layer_types() -> list[str]`, `to_dict() -> dict`, `from_dict(d) -> ModelConfig` (classmethod).
  - `PRESETS: dict[str, ModelConfig]` e `get_preset(name: str, **overrides) -> ModelConfig`.
  - `RMSNorm(dim, eps)`; `SwiGLU(d_model, ffn_dim)` com `gate_proj`, `up_proj`, `down_proj` sem bias.
  - `doc_ids_from_cu_seqlens(cu_seqlens: Tensor, n: int) -> LongTensor[n]` e `positions_from_cu_seqlens(cu_seqlens, n) -> LongTensor[n]` (posição dentro do documento).

Presets (valores do spec):

| preset | vocab | d_model | n_layers | ffn_dim | gdn qk/v heads | attn heads/kv | mtp | backends |
|---|---|---|---|---|---|---|---|---|
| `tiny` | 512 | 128 | 4 | 384 | 1/2 (head_dim 64) | 2/1 (head_dim 64) | 1 | torch / sdpa |
| `s150m` | 65536 | 768 | 12 | 2304 | 6/12 | 6/2 | 1 | fla / flex |
| `b0_8` | 65536 | 1280 | 24 | 3840 | 10/20 | 10/2 | 1 | fla / flex |

- [ ] **Step 1:** Escrever os testes:
  - `test_b0_8_layer_types`: 18 `"gdn"` e 6 `"attn"`; `types[3] == types[23] == "attn"`.
  - `test_invalid_layers_raises`: `n_layers=6` com padrão de 4 → `ValueError`.
  - `test_roundtrip`: `ModelConfig.from_dict(c.to_dict()) == c`.
  - `test_doc_ids`: `cu=[0,3,5]`, `n=5` → `[0,0,0,1,1]`.
  - `test_positions`: mesmo `cu` → `[0,1,2,0,1]`.
- [ ] **Step 2:** Rodar `uv run pytest tests/model -q`. Esperado: FAIL (ImportError).
- [ ] **Step 3:** Implementar. `__post_init__` valida a divisibilidade de `n_layers` por `len(layer_pattern)`, `gdn_v_heads % gdn_qk_heads == 0` e `attn_heads % attn_kv_heads == 0`.
- [ ] **Step 4:** Rodar os testes. Esperado: PASS.
- [ ] **Step 5:** Commit `feat(model): config presets, RMSNorm, SwiGLU, varlen helpers`.

### Task 3: Referência do gated delta rule e conv causal varlen

**Files:**
- Create: `sore/model/gdn_ref.py`, `sore/model/conv.py`
- Test: `tests/model/test_gdn_ref.py`, `tests/model/test_conv.py`

**Interfaces:**
- Consumes: `doc_ids_from_cu_seqlens` (Task 2).
- Produces:
  - `naive_gated_delta_rule(q, k, v, g, beta, scale: float, cu_seqlens: Tensor | None = None) -> Tensor`
    - Shapes: `q,k [B,T,H,Dk]`, `v [B,T,H,Dv]`, `g,beta [B,T,H]`; saída `[B,T,H,Dv]`.
    - Calcula em FP32 e devolve no dtype de `v`. Supõe q e k já normalizados em L2.
    - Com `cu_seqlens`, exige `B == 1` e zera o estado a cada segmento.
  - `causal_conv1d_varlen(x: Tensor[B,T,C], weight: Tensor[C,K], doc_ids: LongTensor[T] | None) -> Tensor[B,T,C]`
    - `weight[:, K-1]` multiplica `x_t` e `weight[:, K-1-j]` multiplica `x_{t-j}`.
    - Termos com `doc_ids[t-j] != doc_ids[t]` ou `t-j < 0` são zerados.

Algoritmo (convenção do FLA; estado `h` de shape `[B,H,Dk,Dv]`):
```python
for t in range(T):
    h = h * g[:, t].exp()[..., None, None]
    v_t = (v[:, t] - (h * k[:, t, ..., None]).sum(-2)) * beta[:, t, ..., None]
    h = h + k[:, t, ..., None] * v_t[..., None, :]
    o[:, t] = torch.einsum("bhd,bhdm->bhm", q[:, t] * scale, h)
```

- [ ] **Step 1:** Escrever os testes:
  - `test_single_step_closed_form`: T=1 → `o = scale * (q·k) * beta * v`.
  - `test_varlen_equals_separate`: `cu=[0,5,12]`; o resultado empacotado é igual a rodar `[0:5]` e `[5:12]` separados (atol 1e-6).
  - `test_conv_matches_conv1d_single_doc`: igual a `F.conv1d(x.transpose(1,2), weight[:,None,:], padding=K-1, groups=C)[..., :T]`.
  - `test_conv_no_leak`: alterar o doc 0 não muda a saída do doc 1.
- [ ] **Step 2:** Rodar os testes. Esperado: FAIL.
- [ ] **Step 3:** Implementar. A conv soma K deslocamentos com máscara (sem `F.conv1d`), para manter a mesma implementação em CPU e GPU.
- [ ] **Step 4:** Rodar os testes. Esperado: PASS.
- [ ] **Step 5:** Commit `feat(model): reference gated delta rule and varlen causal conv`.

### Task 4: Camada GatedDeltaNet (backends torch e fla)

**Files:**
- Create: `sore/model/gdn.py`
- Test: `tests/model/test_gdn.py`

**Interfaces:**
- Consumes: `ModelConfig`, `RMSNorm`, `naive_gated_delta_rule`, `causal_conv1d_varlen`, `doc_ids_from_cu_seqlens`.
- Produces: `GatedDeltaNet(cfg: ModelConfig)` com `forward(x: Tensor[B,T,D], cu_seqlens: Tensor | None = None) -> Tensor[B,T,D]`.
  - Parâmetros: `q_proj, k_proj (D→Hqk·Dk)`, `v_proj, g_proj (D→Hv·Dv)`, `a_proj, b_proj (D→Hv)`, `conv_q, conv_k, conv_v` (pesos `[C,K]`), `A_log [Hv]`, `dt_bias [Hv]`, `o_norm = RMSNorm(Dv)`, `o_proj (Hv·Dv→D)`. Todos sem bias.

Detalhes que o implementador não deduz sozinho:
- q, k e v passam pela conv e depois por `silu`. q e k são repetidos de `Hqk` para `Hv` heads com `repeat_interleave(Hv // Hqk, dim=2)`.
- `g = -exp(A_log.float()) * softplus(a_proj(x).float() + dt_bias)`; `beta = sigmoid(b_proj(x))`; `scale = Dk ** -0.5`.
- Backend `torch`: `F.normalize` em q e k, depois `naive_gated_delta_rule`.
- Backend `fla`: `fla.ops.gated_delta_rule.chunk_gated_delta_rule(q, k, v, g, beta, scale=scale, cu_seqlens=cu_seqlens, use_qk_l2norm_in_kernel=True)`. Assinatura confirmada no FLA 0.5.2: `(q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=False, use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False, state_v_first=False, cu_seqlens=None, cu_seqlens_cpu=None, cp_context=None)` → `(o, final_state)`. Passar também `cu_seqlens_cpu` (cópia em CPU feita pelo loader) para evitar sync host↔device.
- Saída: `o_proj((o_norm(o) * silu(g_proj(x))).flatten(-2))`.
- Init: `A_log = log(U(1, 16))`; `dt_bias = softplus⁻¹(dt)`, com `dt` log-uniforme em `[1e-3, 1e-1]`.

- [ ] **Step 1:** Escrever os testes:
  - `test_shapes_finite` (tiny, torch).
  - `test_packed_equals_separate` (torch, `cu=[0,7,20]`).
  - `test_extreme_inputs_finite`: `x * 100` → saída `isfinite().all()`.
  - `@fla test_fla_matches_torch`: fp32, `cu=[0,1,2,100,257]`, D=128. Saída com atol 1e-3; gradientes de `x` e de `q_proj.weight` com rtol 1e-2.
- [ ] **Step 2:** Rodar os testes. Esperado: FAIL.
- [ ] **Step 3:** Implementar `GatedDeltaNet`.
- [ ] **Step 4:** Rodar `uv run pytest tests/model/test_gdn.py -v`. Esperado: PASS, com o teste FLA executado (não pulado) no WSL.
- [ ] **Step 5:** Commit `feat(model): GatedDeltaNet layer with FLA and reference backends`.

### Task 5: GatedAttention com RoPE parcial e máscara de documento

**Files:**
- Create: `sore/model/rope.py`, `sore/model/attention.py`
- Test: `tests/model/test_attention.py`

**Interfaces:**
- Consumes: `ModelConfig`, `RMSNorm`, `positions_from_cu_seqlens`, `doc_ids_from_cu_seqlens`.
- Produces:
  - `PartialRotary(head_dim: int, partial: float, theta: float)` com `forward(q, k, positions: LongTensor[T]) -> tuple[Tensor, Tensor]`. Gira só as primeiras `int(head_dim * partial)` dims (par).
  - `GatedAttention(cfg)` com `forward(x: Tensor[B,T,D], cu_seqlens: Tensor | None = None) -> Tensor[B,T,D]`.
    - Projeções `q_proj, gate_proj (D→H·Dh)` e `k_proj, v_proj (D→Hkv·Dh)`; `q_norm, k_norm = RMSNorm(Dh)`; `o_proj`.
    - Saída: `o_proj(attn * sigmoid(gate_proj(x)))`.
    - Backend `sdpa`: máscara booleana densa (causal e mesmo documento), `enable_gqa=True`.
    - Backend `flex`: `flex_attention` com `create_block_mask` (mask_mod causal e de documento) e `enable_gqa=True`.

- [ ] **Step 1:** Escrever os testes:
  - `test_rope_identity_at_zero`: posição 0 não altera q.
  - `test_rope_untouched_tail`: dims ≥ rot_dim não mudam.
  - `test_causal`: perturbar o token 9 não muda as saídas 0..8.
  - `test_packed_equals_separate` (sdpa).
  - `@cuda test_flex_matches_sdpa`: T=300, `cu=[0,120,300]`, atol 2e-3 em fp32.
- [ ] **Step 2:** Rodar os testes. Esperado: FAIL.
- [ ] **Step 3:** Implementar.
- [ ] **Step 4:** Rodar os testes. Esperado: PASS.
- [ ] **Step 5:** Commit `feat(model): gated GQA attention with partial RoPE and document masking`.

### Task 6: Residuais (plain e gated hyper) e Block

**Files:**
- Create: `sore/model/residual.py`, `sore/model/block.py`
- Test: `tests/model/test_residual_block.py`

**Interfaces:**
- Consumes: `ModelConfig`, `RMSNorm`, `SwiGLU`, `GatedDeltaNet`, `GatedAttention`.
- Produces:
  - `init_stream(x: Tensor[B,T,D], cfg) -> H`: plain → `H = x`; hyper → `x.unsqueeze(2).expand(B,T,n,D).clone()`.
  - `StreamFinal(cfg)` com `forward(H) -> [B,T,D]`: plain → identidade; hyper → soma ponderada por `softmax(mix_logits)`, que começa uniforme.
  - Protocolo de residual por sub-bloco: `read(H) -> x [B,T,D]`, `write(H, y [B,T,D]) -> H`.
  - `PlainResidual`: `read = identidade`, `write = H + y`.
  - `GatedHyperResidual(d, n, rank, layer_idx)`, com `H` de shape `[B,T,n,D]`:
    - `z = rms(H.mean(2))`
    - `read`: `r = sigmoid(up(silu(down(z)))).view(B,T,n,D)`; `x = (r * H).sum(2)`. `up` começa com peso zero e bias `logit(1/n)`, ou seja, a média no início.
    - `write`: `w = 2 * sigmoid(write_bias + W_w z)`, com shape `[B,T,n]`; `H = H + w[...,None] * y[:,:,None]`. `W_w` começa em zero. `write_bias` começa com `w = 1` no ramo `layer_idx % n` e `w = 0.1` nos outros (quebra de simetria).
  - `make_residual(cfg, layer_idx, sub: int) -> Residual`.
  - `Block(cfg, layer_idx, mixer_type: str)` com `forward(H, cu_seqlens) -> H`. Cada um dos dois sub-blocos (mixer e FFN) tem seu próprio residual e faz `H = res.write(H, f(norm(res.read(H))))`.

- [ ] **Step 1:** Escrever os testes:
  - `test_plain_block_shapes` para `gdn` e `attn`.
  - `test_hyper_reduces_to_mean_at_init`: com `H` de ramos idênticos, `read(H)` é igual ao ramo.
  - `test_hyper_symmetry_breaks`: depois de um backward num Block tiny com `residual="hyper"`, as linhas de `W_w.weight.grad` diferem entre ramos.
  - `test_hyper_packed_equals_separate`.
- [ ] **Step 2:** Rodar os testes. Esperado: FAIL.
- [ ] **Step 3:** Implementar.
- [ ] **Step 4:** Rodar os testes. Esperado: PASS.
- [ ] **Step 5:** Commit `feat(model): plain and gated hyper-connection residuals, Block`.

### Task 7: SoreLM, loss em chunks e MTP

**Files:**
- Create: `sore/model/loss.py`, `sore/model/mtp.py`, `sore/model/lm.py`
- Test: `tests/model/test_loss.py`, `tests/model/test_lm.py`

**Interfaces:**
- Consumes: tudo das Tasks 2–6.
- Produces:
  - `chunked_lm_loss(hidden: Tensor[N,D], weight: Tensor[V,D], labels: LongTensor[N], chunk_size: int = 8192) -> tuple[Tensor, int]`: devolve (soma da CE em fp32, número de tokens válidos). Cada chunk passa por `torch.utils.checkpoint`, para não guardar os logits.
  - `mtp_labels(labels: LongTensor[1,N], cu_seqlens: Tensor | None) -> LongTensor[1,N]`: `out[t] = labels[t+1]` se `t+1` estiver no mesmo documento, senão `-100`.
  - `MTPModule(cfg)` com `forward(h: [1,N,D], emb_next: [1,N,D], cu_seqlens) -> [1,N,D]`. Faz `proj(cat(norm_h(h), norm_e(emb_next)))`, passa por `Block(dataclasses.replace(cfg, residual="plain"), layer_idx=cfg.n_layers, mixer_type="attn")` e depois por `final_norm`. `h` é a saída de `StreamFinal` seguida da norma final do tronco. `emb_next[t] = embed(input_ids[t+1])` dentro do documento, e zero na última posição de cada documento.
  - `LMOutput` (dataclass): `loss, lm_loss, mtp_loss: Tensor | None; n_tokens: int`.
  - `SoreLM(cfg)` com:
    - `forward(input_ids: LongTensor[1,N], cu_seqlens: Tensor | None = None, labels: LongTensor[1,N] | None = None, inputs_embeds: Tensor | None = None) -> LMOutput`.
    - `logits(input_ids, cu_seqlens) -> Tensor[1,N,V]` (só para teste e inferência).
    - `num_params(exclude_embedding: bool = False) -> int`.
  - `loss = lm_loss + mtp_weight * mtp_loss`, cada um como média sobre os tokens válidos.
  - Init: normal(0, 0.02). `o_proj` e `down_proj` escalados por `1/sqrt(2 * n_layers)`.
  - Com `cfg.grad_checkpoint`, cada Block roda sob `checkpoint(use_reentrant=False)`.

- [ ] **Step 1:** Escrever os testes:
  - `test_chunked_matches_full`: soma e gradientes iguais a `F.cross_entropy(..., reduction="sum")` (rtol 1e-5), com `chunk_size=7` e N=50 (sobra de chunk).
  - `test_init_loss_near_log_vocab`: tiny → `|lm_loss - ln(512)| < 0.5`.
  - `test_packed_logits_equal_separate`: tiny, plain e hyper, `cu=[0,9,30]`, atol 1e-5. (Review Focus 1)
  - `test_mtp_labels_mask_boundaries`.
  - `test_param_counts`, em `torch.device("meta")`: `b0_8` dentro de `[0.75e9, 0.85e9]`; `s150m` dentro de `[0.13e9, 0.17e9]`. Se ficar fora, ajustar só `ffn_dim` (múltiplo de 128) e anotar no commit.
  - `test_backward_all_params_get_grad`: tiny, nenhum parâmetro com `grad is None`.
- [ ] **Step 2:** Rodar os testes. Esperado: FAIL.
- [ ] **Step 3:** Implementar.
- [ ] **Step 4:** Rodar `uv run pytest tests/model -q`. Esperado: PASS.
- [ ] **Step 5:** Commit `feat(model): SoreLM with chunked loss and MTP head`.

### Task 8: Muon + AdamW e schedule WSD

**Files:**
- Create: `sore/train/__init__.py`, `sore/train/optim.py`, `sore/train/schedule.py`
- Test: `tests/train/test_optim.py`, `tests/train/test_schedule.py`

**Interfaces:**
- Consumes: `SoreLM`.
- Produces:
  - Muon: usar o nativo `torch.optim.Muon(params, lr=lr_muon, weight_decay=weight_decay, momentum=0.95, nesterov=True, ns_steps=5, adjust_lr_fn="match_rms_adamw")`. Esse `adjust_lr_fn` aplica a escala Moonlight, então o lr fica na mesma escala do AdamW. Confirmar com `inspect` que `"match_rms_adamw"` é aceito no torch 2.14.1; se não for, anotar o valor aceito no commit. Não implementar Newton-Schulz próprio.
  - `split_param_groups(model) -> tuple[list[nn.Parameter], list[nn.Parameter]]` devolve (muon, adamw):
    - Muon recebe parâmetros com `ndim == 2` dentro de `blocks`/`mtp` cujo nome não contém `conv`.
    - Todo o resto vai para AdamW (embedding, normas, `A_log`, `dt_bias`, conv, hyper-residual).
  - `build_optimizers(model, lr_muon, lr_adamw, weight_decay) -> list[torch.optim.Optimizer]`. AdamW usa `betas=(0.9, 0.95)`, `eps=1e-8` e decay só em `ndim >= 2`.
  - `wsd_multiplier(step: int, warmup: int, total: int, decay_frac: float) -> float`: warmup linear de 0 a 1, platô em 1 e decay `1 - sqrt(progresso)` até 0 no último `decay_frac`.

- [ ] **Step 1:** Escrever os testes:
  - `test_one_step_updates_all`: um passo de `build_optimizers` num SoreLM tiny altera todos os parâmetros que têm grad.
  - `test_split`: `embed.weight` no AdamW, `blocks.0.mixer.q_proj.weight` no Muon, `conv_q` no AdamW, e nenhum parâmetro repetido ou faltando.
  - `test_wsd_points`: `(0, 10, 100, 0.2) → 0.0`; `(10, ...) → 1.0`; `(50, ...) → 1.0`; `(100, ...) → 0.0`; `(90, ...) → 1 - sqrt(0.5)`.
- [ ] **Step 2:** Rodar os testes. Esperado: FAIL.
- [ ] **Step 3:** Implementar.
- [ ] **Step 4:** Rodar os testes. Esperado: PASS.
- [ ] **Step 5:** Commit `feat(train): Muon+AdamW split and WSD schedule`.

### Task 9: Shards uint32 e stream empacotado retomável

**Files:**
- Create: `sore/train/data.py`, `scripts/make_synthetic_shard.py`
- Test: `tests/train/test_data.py`

**Interfaces:**
- Produces:
  - `write_token_shard(path: Path, tokens: np.ndarray) -> None`: grava `uint32` cru mais o sidecar `path.with_suffix(".json")` com `{"n_tokens": int, "dtype": "uint32"}`.
  - `Batch` (dataclass): `input_ids: LongTensor[1,N]`, `labels: LongTensor[1,N]`, `cu_seqlens: LongTensor[S+1]`.
  - `PackedTokenStream(paths: list[Path], seq_len: int, micro_batch: int, eos_id: int, seed: int)` com:
    - `__iter__` / `__next__ -> Batch`
    - `set_shape(seq_len: int, micro_batch: int) -> None`
    - `state_dict() -> dict` e `load_state_dict(d) -> None`
    - `tokens_consumed: int`

Comportamento:
- Usa memmap dos shards concatenados. As janelas de `seq_len + 1` tokens são amostradas por permutação determinística, a partir de `(seed, epoch, seq_len)`.
- `N = micro_batch * seq_len`. `labels[t] = -100` quando `input_ids[t] == eos_id`.
- `cu_seqlens` é a união de: 0, N, múltiplos de `seq_len` e `t+1` para cada EOS em `input_ids[t]`. Sem duplicatas e sem segmentos vazios.
- `set_shape` começa uma nova permutação e mantém `tokens_consumed`.
- `make_synthetic_shard.py --out data/synth.bin --n-tokens 2000000 --vocab 512 --eos 0 --seed 0` gera documentos de comprimento aleatório (8–600) com padrões aprendíveis (sequências aritméticas mod vocab).

- [ ] **Step 1:** Escrever os testes:
  - `test_cu_seqlens_handcrafted`: tokens com EOS em posições conhecidas → `cu_seqlens` esperado exato.
  - `test_labels_masked_at_eos`.
  - `test_consecutive_eos_no_empty_segments`: diferenças de `cu_seqlens` todas ≥ 1. (Review Focus 4)
  - `test_resume_determinism`: `state_dict` após 3 batches; um stream novo com `load_state_dict` produz um 4º batch idêntico.
  - `test_set_shape_changes_shape`.
- [ ] **Step 2:** Rodar os testes. Esperado: FAIL.
- [ ] **Step 3:** Implementar.
- [ ] **Step 4:** Rodar os testes. Esperado: PASS.
- [ ] **Step 5:** Commit `feat(train): uint32 shards and resumable packed token stream`.

### Task 10: Checkpoint, trainer, CLI e configs

**Files:**
- Create: `sore/train/config.py`, `sore/train/checkpoint.py`, `sore/train/trainer.py`, `sore/train/__main__.py`, `configs/tiny_local.yaml`, `configs/s150m_ablation.yaml`, `configs/b0_8_pretrain.yaml`
- Test: `tests/train/test_trainer.py`

**Interfaces:**
- Consumes: `SoreLM`, `get_preset`, `build_optimizers`, `wsd_multiplier`, `PackedTokenStream`.
- Produces:
  - `Stage(until_tokens: int, seq_len: int, micro_batch: int)`.
  - `TrainConfig` com os campos:
    - Modelo e dados: `model_preset: str`, `model_overrides: dict`, `data_paths: list[str]`, `eos_id: int`.
    - Batch e estágios: `grad_accum: int`, `stages: list[Stage]`, `total_tokens: int`.
    - Otimização: `lr_muon=0.02`, `lr_adamw=3e-3`, `warmup_steps=500`, `decay_frac=0.2`, `weight_decay=0.1`, `grad_clip=1.0`.
    - Execução: `compile: bool = False`, `log_every=10`, `ckpt_every_tokens: int`, `out_dir: str`, `seed=0`.
  - `load_train_config(path: Path) -> TrainConfig` — `ValueError` se `seq_len * micro_batch` não for igual em todos os estágios (tokens/passo constante ⇒ schedule por passo é exato).
  - `save_checkpoint(path, model, optimizers, data_state: dict, train_state: dict) -> None`: grava em `.tmp` e faz `os.replace`. Inclui `model_cfg` e os estados de RNG do torch, cuda, numpy e random.
  - `load_checkpoint(path, model, optimizers) -> tuple[dict, dict]`.
  - `TrainSummary(steps: int, tokens: int, skipped_steps: int, last_loss: float)`.
  - `train(cfg: TrainConfig, resume: Path | None = None, max_steps: int | None = None) -> TrainSummary`.
  - CLI: `python -m sore.train --config X.yaml [--resume ckpt.pt] [--max-steps N]`.

Comportamento do loop:
- `total_steps = total_tokens // (micro_batch * seq_len * grad_accum)` (constante entre estágios).
- Cada micro-batch roda sob autocast bf16 na GPU ou fp32 na CPU. O loss soma `lm + λ·mtp`, normalizado pelo número de tokens válidos do passo inteiro.
- Se o grad norm não for finito, o passo é pulado: `zero_grad`, `skipped += 1` e nenhum otimizador anda.
- Clip em 1.0. O lr de cada otimizador é `base * wsd`.
- O estágio troca quando `tokens_consumed >= until_tokens` (`stream.set_shape`).
- Log em JSONL (`runs/<out_dir>/metrics.jsonl`) com: step, tokens, loss, lm_loss, mtp_loss, grad_norm, lr_muon, tok_s, mem_gb e seq_len.
- Checkpoint `last.pt` a cada `ckpt_every_tokens` e ao terminar.

Valores dos configs (do spec):
- **`b0_8_pretrain.yaml`:**
  - `grad_accum=64` (≈524K tokens por passo).
  - `stages=[{until_tokens: 35e9, seq_len: 4096, micro_batch: 2}, {until_tokens: 50e9, seq_len: 8192, micro_batch: 1}]`, `total_tokens=50e9`.
  - `grad_checkpoint=true`, `compile=true`.
- **`tiny_local.yaml`:** usa `data/synth.bin`, `grad_accum=1`, `stages=[{until 200000, seq 128, mb 4}, {until 400000, seq 256, mb 2}]`.

- [ ] **Step 1:** Escrever os testes (CPU, tiny, shard sintético num tmp_path):
  - `test_overfit`: 40 passos; o loss final fica abaixo de 0,5× o inicial.
  - `test_resume_equivalence`: 6 passos direto contra 3 + checkpoint + resume + 3, com a troca de estágio no passo 3; os losses dos passos 4–6 batem com atol 1e-5. (Review Focus 3)
  - `test_nonfinite_step_skipped`: monkeypatch faz o loss do passo 2 virar NaN; `skipped_steps == 1` e os parâmetros ficam iguais aos do fim do passo 1.
- [ ] **Step 2:** Rodar os testes. Esperado: FAIL.
- [ ] **Step 3:** Implementar.
- [ ] **Step 4:** Rodar `uv run pytest -q`. Esperado: tudo passa.
- [ ] **Step 5:** Smoke na 3050:
  ```
  uv run python scripts/make_synthetic_shard.py --out data/synth.bin --n-tokens 2000000 --vocab 512 --eos 0 --seed 0
  uv run python -m sore.train --config configs/tiny_local.yaml --max-steps 300
  ```
  Esperado: o loss cai, nenhum passo é pulado e a troca de estágio aparece em `metrics.jsonl`.
- [ ] **Step 6:** Commit `feat(train): resumable trainer with WSD, stage curriculum, nonfinite guard`.

### Task 11: FLOPs, benchmark de throughput e runbook da 4090

**Files:**
- Create: `sore/model/flops.py`, `scripts/bench.py`, `docs/RUNBOOK_4090.md`
- Test: `tests/model/test_flops.py`

**Interfaces:**
- Consumes: `SoreLM`, `get_preset`, `build_optimizers`.
- Produces:
  - `flops_per_token(cfg: ModelConfig, seq_len: int) -> float`: treino (fwd+bwd) = `6·(params de matmul sem embedding + V·D da head)` mais o termo de atenção `6·n_attn·attn_heads·attn_head_dim·seq_len` (causal) e o termo do GDN `6·n_gdn·Hv·Dk·Dv·2`.
  - `scripts/bench.py --preset P --seq-len T --micro-batch B --steps S [--compile] [--grad-checkpoint]`: imprime tok/s, MFU (pico configurável `--peak-tflops`, padrão 165 para a 4090) e pico de VRAM.

- [ ] **Step 1:** Escrever os testes:
  - `test_flops_at_least_6n`: `flops_per_token(b0_8, 4096) >= 6 * num_params(exclude_embedding=True)`.
  - `test_flops_monotonic_in_seq_len`.
- [ ] **Step 2:** Rodar os testes. Esperado: FAIL.
- [ ] **Step 3:** Implementar `flops.py` e `bench.py`.
- [ ] **Step 4:** Rodar os testes. Esperado: PASS. Depois rodar na 3050: `uv run python scripts/bench.py --preset s150m --seq-len 2048 --micro-batch 1 --steps 10 --grad-checkpoint --peak-tflops 35`. Esperado: termina sem OOM e imprime as métricas. Anotar os números no runbook.
- [ ] **Step 5:** Fazer o trainer gravar `mfu` em cada linha do `metrics.jsonl` (via `flops_per_token` e `peak_tflops` do `TrainConfig`, padrão 165). Depois escrever `docs/RUNBOOK_4090.md` com:
  - Setup: `uv sync` e `nvidia-smi`.
  - Portão obrigatório: `bench.py --preset b0_8` a 4096 e a 8192, com `--compile --grad-checkpoint`. Precisa caber em 22GB; se não couber, baixar `micro_batch` e subir `grad_accum`, mantendo os tokens por passo.
  - Lançamento com `tmux` e `python -m sore.train --config configs/b0_8_pretrain.yaml`.
  - Retomada com `--resume runs/.../last.pt`.
  - O que monitorar em `metrics.jsonl`.
- [ ] **Step 6:** Commit `feat: FLOPs model, throughput benchmark, 4090 runbook`.
