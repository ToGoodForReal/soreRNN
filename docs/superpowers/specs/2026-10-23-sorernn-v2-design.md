# soreRNN v2 — Design (arquitetura estilo Qwen4, recorrente moderna, ~0.8B, PT-BR)

Data: 2026-10-23 · Branch: `feat/torch-stack` · Base congelada: tag `v0.1-cpp-engine`

## 1. Meta e restrições

- **Meta:** superar o Qwen3.5-0.8B (modo reasoning) num placar PT-BR fixo (seção 8), com reasoning (`<think>`) e tool calling.
- **Estratégia (decidida):** pré-treino **do zero**, ~0.8B parâmetros, **~50B tokens** (~6–7 semanas numa RTX 4090).
- **Hardware de treino:** 1× RTX 4090 24GB, Linux nativo. **Dev local:** Windows 11 + WSL2 (Ubuntu 24.04) + RTX 3050 Laptop 6GB.
- **Stack:** Python 3.12, PyTorch (BF16), flash-linear-attention (FLA, kernels Triton chunk-wise), uv. O motor C++/CUDA vai para `legacy/cpp/` e não recebe mais features.
- **Risco aceito:** do zero com 50B tokens contra um modelo treinado com dezenas de trilhões. As alavancas para compensar são dados PT de alta qualidade, destilação de logits na fase de annealing e no SFT, e pós-treino com RL verificável.

## 2. Arquitetura "soreQ4": Qwen4 (Qwen3.8-Flash-Next) adaptada

Referência: `Qwen/Qwen3.8-Flash-Next` (`model_type: qwen4_exp`). O núcleo é recorrente: 3 de cada 4 camadas são Gated DeltaNet. Adaptações para 0.8B denso numa GPU:

| Componente Qwen4 | soreQ4 v1 | Motivo |
|---|---|---|
| 12×(3 GDN + 1 Qwen Sparse Attention), MoE em todas | 6×(3 GDN + 1 atenção global com gate), **FFN SwiGLU densa** | MoE não compensa em 0.8B numa GPU única. QSA (orçamento de 2048 tokens) quase não economiza a 4K–8K |
| QSA (micro-blocos de 4 tokens, indexer MQA) | Atenção global com gate; QSA entra no plano de contexto longo (≥16K) | Mantém o ponto de troca: `attention_impl` configurável |
| Gated residual (hyper-connections, 4 ramos, rank 320) | `GatedHyperResidual` (n=4, rank configurável), **ligável por config**, decidido por ablação em 150M | Sem fórmula oficial. Implementação própria baseada em Hyper-Connections |
| Engram n-gram 51B na camada 2 | Plano de ablação separado (tabela hash de n-gramas, esparsa, em RAM) | Complexidade de otimizador esparso. Interface reservada |
| MTP 1 camada | MTP 1 camada (atenção + FFN), peso λ=0.3→0.1 | Ganho de dados e decoding especulativo |
| RoPE parcial 25%, θ=1e7 | RoPE parcial 25% nas camadas de atenção, θ=1e7 | Extensão de contexto via YaRN depois |
| Muon + AdamW, sem warmup de batch size | Muon (matrizes 2D ocultas) + AdamW (embedding, normas, escalares) e batch fixo desde o passo 0 | Igual ao Qwen4 |
| Vocab 248.320, embedding não amarrado | Tokenizer Qwen3.5/3.8 **podado** (alvo ~64K–96K), embedding **amarrado** | Libera cerca de 190M de params para o "cérebro" |

**Preset `b0_8`:** d_model=1280, 24 camadas (6 grupos [GDN, GDN, GDN, ATTN]), FFN SwiGLU 3840.
- GDN: 10 heads QK × 128, 20 heads V × 128 (GVA 2×), conv curta k=4, gate de saída com RMSNorm.
- Atenção: 10 heads Q × 128, 2 heads KV, gate sigmoid na saída, RoPE parcial 25%.
- Vocab 65.536 (provisório até o pruning), embedding amarrado, MTP 1 camada.
- Total ≈ 0,79B.

**Preset `s150m`:** igual em forma, com d=768, 12 camadas e FFN 2304. Usado para ablações.

**Preset `tiny`:** d=128, 4 camadas, vocab 512. Para testes locais.

**Formulação do GDN** (Yang et al., Gated Delta Networks):
`S_t = α_t S_{t-1}(I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ`, `o_t = S_t q_t`
- q e k normalizados em L2.
- `α_t = exp(g_t)`, com `g_t = −exp(A_log)·softplus(W_a x + dt_bias)`.
- `β_t = σ(W_b x)`.
- Saída: `o_proj(RMSNorm(o) ⊙ silu(W_g x))`.

**Fronteiras de documento:** o estado reseta em cada documento. As sequências são empacotadas com `cu_seqlens` (varlen no FLA e máscara de documento na atenção). Isso corrige o vazamento de estado do motor C++.

## 3. Contexto progressivo

O treino segue estágios definidos em config (`stages: [{until_tokens, seq_len}]`):
- 4K até cerca de 70% dos tokens, depois 8K.
- Extensões para 16K/32K/64K ficam num plano posterior: upsampling de documentos longos, YaRN só nas camadas de atenção e QSA para mais de 16K.

O schedule de LR é **WSD** (warmup-stable-decay). A fase estável permite trocar de estágio e retomar o treino sem refazer o warmup.

## 4. Tokenizer

O Qwen3.8 usa vocabulário de 248.320 tokens, o mesmo do Qwen3.5.

Plano: podar o tokenizer por frequência num corpus amostrado (PT, EN, código, JSON). O conjunto mantido é fechado sob merges, para que todo token preservado continue alcançável, e inclui:
- os 256 bytes,
- todos os tokens especiais de chat, think e tool,
- os tokens de visão reservados.

O mapa `old_id → new_id` permite destilar logits restritos ao subvocabulário. Métricas de aceite: fertilidade em PT (tokens por palavra) no máximo 3% pior que o vocab completo, e cobertura de 100% (fallback para bytes). Os tokens passam a ser armazenados como `uint32`.

## 5. Dados de pré-treino (50B tokens)

Fontes candidatas (verificar licença de cada uma): FineWeb-2 `por_Latn`, GigaVerbo, CulturaX-pt, Corpus Carolina, BrWaC e Wikipedia PT. Replay: FineWeb-Edu, FineMath e código. O pipeline usa classificador de qualidade educacional PT e dedup MinHash.

Mistura inicial: 65% PT web filtrado, 10% PT de alta qualidade, 15% EN educacional e matemática, 10% código e JSON. No annealing (últimos ~10%) entram dados de alta qualidade com mais peso, mais instrução e reasoning.

## 6. Treino (eficiência na 4090)

- Mixed precision: autocast BF16, pesos mestre FP32.
- Activation checkpointing por bloco, obrigatório nos presets ≥150M.
- Cross-entropy em chunks: não materializa `[N, V]`.
- `torch.compile` opcional. FP8 (torchao) fica para depois, como experimento.
- Proteções: passo é pulado se aparecer NaN ou Inf; grad clip em 1.0; alerta de spike de loss.
- Checkpoint retomável: modelo, otimizadores, RNG, cursor de dados e estágio.
- Métricas: tokens/s, MFU estimado, pico de VRAM, loss principal, loss MTP e grad norm.
- FSDP2 fica só como caminho opcional, caso apareça multi-GPU em nuvem.

## 7. Pós-treino (planos posteriores)

1. **Template ChatML com tokens especiais:** `<|im_start|>{system,user,assistant,tool}`, `<think>…</think>`, `<tool_call>{json}</tool_call>`, `<tool_response>`, `/think` e `/no_think`.
2. **SFT multi-turno com máscara apenas no assistant.** Três blocos: instrução geral PT; reasoning destilado (traces curtos e verificados); tool calling (xLAM/APIGen, Hermes-FC e ToolACE traduzidos, mais sintéticos validados por JSON Schema, incluindo chamadas paralelas, recusa e erro de ferramenta).
3. **Destilação de logits (KD)** de teacher Qwen3.5/3.8 sobre o subvocabulário, online no SFT e no annealing.
4. **Preferência on-policy** (DPO, SimPO ou APO).
5. **GRPO/DAPO com recompensa verificável:** matemática e tool calling (AST, schema e execução).

## 8. Placar (Fase 0, plano separado)

Tudo roda no mesmo harness para o Qwen3.5-0.8B (thinking e non-thinking) e para o nosso modelo:
- Open PT LLM Leaderboard: ENEM, BLUEX, OAB, ASSIN2 RTE/STS, FaQuAD-NLI, HateBR, PT Hate Speech, tweetSentBR.
- GSM8K-PT.
- Tool calling PT no estilo BFCL (AST e execução).

## 9. Multimodal (congelado)

Interface `inputs_embeds` no `SoreLM.forward` e IDs reservados para `<|vision_start|>`, `<|vision_end|>` e `<|image_pad|>`. O projetor MLP **não** entra em nenhum plano até o motor de texto atingir maturidade.

## 10. Planos de implementação

1. **Núcleo do modelo e loop de treino** (este ciclo): `docs/superpowers/plans/2026-10-23-sorernn-v2-core.md`
2. Tokenizer podado e pipeline de dados PT (streaming, filtro, dedup, shards `uint32`)
3. Harness de avaliação e baseline do Qwen3.5-0.8B
4. Ablações em 150M: hyper-residual, engram, MTP, Muon vs AdamW
5. Extensão de contexto (YaRN e QSA)
6. Pós-treino: SFT, KD, preferência e GRPO
