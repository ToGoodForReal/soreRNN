# Relatório Técnico: Protocolo de Dia 1 na RTX 4090 & Análise de FP8

**Data:** 27 de Outubro de 2026  
**Ambiente:** Linux x86_64, NVIDIA GeForce RTX 4090 24GB (sm89, Driver 595.99.02, CUDA 13.2)  
**Stack de Software:** PyTorch `2.14.1+cu130`, Triton `3.8.0`, Flash-Linear-Attention `0.5.2`, TorchAO `0.18.0`  
**Alvo:** Modelo `b0_8` (781,2M parâmetros, vocabulário 65.536, pré-treino de 50B tokens)

---

## 1. Resumo Executivo

1. **Validação e Suíte de Testes:**
   - 368 testes passaram sem erros. O teste de FP8 nativo (`test_fp8_real_smoke`) executou e passou na RTX 4090.
2. **Benchmark Inicial (scripts/bench.py):**
   - Apontou `fp8=tensorwise` com política de checkpointing `ffn` como o mais rápido (**17.691 tok/s**, MFU 58,7%, 21,24 GB VRAM), mas essa receita produz gradientes NaN no treino real (ver §3.1).
   - O melhor exato (BF16, exatas off) atingiu **15.306 tok/s** (MFU 50,8%), porém com VRAM alocada de **22,98 GB**, acima do critério de $\le 22$ GB do runbook.
   - A única configuração exata medida que respeita o critério de VRAM é `full, frac 0.5, exatas ON`: **13.550 tok/s**, 19,94 GB.
3. **Portão A/B de Qualidade (3e8 tokens, 4096x2):**
   - **Referência BF16 (`ffn`, frac 1.0, exatas ON):** 572 passos concluídos (~5,5 h), loss de `14,74` para `2,91` a **~14.720 tok/s** (MFU 48,9%), pico de **22,37 GB** alocados (acima do critério de 22 GB).
   - **Candidata FP8 Tensorwise:** todos os passos observados foram descartados (`gnorm nan`, `skipped: true`). O log desse run foi perdido, mas a causa foi reproduzida e confirmada depois: um bug de compilação do Inductor (ver §3.1).
   - **Candidata FP8 Rowwise:** interrompida após **18 de 572 passos** (~3%). Sem skips, loss acompanhando a do BF16, mas a **~13.650 tok/s** (MFU 45,3%), mais lenta que o BF16. O critério de qualidade (loss final FP8 $\le$ 1,01 × BF16) **não foi avaliado**.
4. **Decisão:** FP8 descartado (o rowwise é mais lento que o BF16 em todas as GEMMs do b0_8; o tensorwise compilado gera NaN, e os contornos que o corrigem derrubam o throughput). O pré-treino segue em **BF16 com `checkpoint_policy: full, checkpoint_frac: 0.5`**, dentro do critério de 22 GB (ver §5.3).

---

## 2. Matriz de Benchmark do Dia 1 (Resultados Reais)

Tabela completa gerada por `uv run python scripts/summarize_bench.py runs/day1/bench.jsonl`:

| Configuração / Overrides | Flags | tok/s | MFU | VRAM Alocado / Reservado (GB) | Classificação |
|---|---|---|---|---|---|
| **b0_8 4096x2** · `ffn, frac 1.0, exatas off` | `[compile fp8=tensorwise ckpt]` | **17.691** | **58,7%** | 21,24 / 22,02 | $\ge$ 15.000 tok/s (inválido: grad NaN no treino) |
| **b0_8 8192x1** · `ffn, frac 1.0, exatas off` | `[compile fp8=tensorwise ckpt]` | **17.649** | **61,0%** | 21,24 / 22,10 | $\ge$ 15.000 tok/s (inválido: grad NaN no treino) |
| **b0_8 4096x2** · `ffn, frac 1.0, exatas off` | `[compile ckpt]` (BF16) | **15.306** | **50,8%** | 22,98 / 23,37 | $\ge$ 15.000 tok/s (**VRAM > 22 GB**) |
| **b0_8 8192x1** · `ffn, frac 1.0, exatas off` | `[compile ckpt]` (BF16) | **15.277** | **52,8%** | 22,98 / 23,37 | $\ge$ 15.000 tok/s (**VRAM > 22 GB**) |
| **b0_8 4096x2** · `ffn, frac 1.0, exatas off` | `[compile fp8=rowwise ckpt]` | **14.346** | **47,6%** | 21,23 / 22,09 | $\ge$ 11.800 tok/s (piso viável) |
| **b0_8 8192x1** · `ffn, frac 1.0, exatas off` | `[compile fp8=rowwise ckpt]` | **14.310** | **49,4%** | 21,24 / 22,09 | $\ge$ 11.800 tok/s (piso viável) |
| **b0_8 4096x2** · `ffn, frac 1.0, exatas ON` | `[compile ckpt]` (BF16) | **14.208** | **47,2%** | 22,36 / 22,97 | $\ge$ 11.800 tok/s (**VRAM > 22 GB**) |
| **b0_8 8192x1** · `ffn, frac 1.0, exatas ON` | `[compile ckpt]` (BF16) | **14.088** | **48,7%** | 22,36 / 22,97 | $\ge$ 11.800 tok/s (**VRAM > 22 GB**) |
| **b0_8 4096x2** · `full, frac 0.5, exatas ON` | `[compile ckpt]` (BF16) | **13.550** | **45,0%** | 19,94 / 20,55 | $\ge$ 11.800 tok/s (VRAM $\le$ 22 GB) |
| **b0_8 8192x1** · `full, frac 0.5, exatas ON` | `[compile ckpt]` (BF16) | **13.405** | **46,3%** | 19,94 / 20,55 | $\ge$ 11.800 tok/s (VRAM $\le$ 22 GB) |
| **b0_8 4096x2** · `full, frac 1.0, gdn_path ON` | `[compile ckpt]` (BF16) | **13.423** | **44,6%** | 13,10 / 14,01 | $\ge$ 11.800 tok/s (VRAM ultra-baixa) |
| **b0_8 4096x2** · `full, frac 1.0, todas off` | `[compile ckpt]` (BF16 base) | **13.361** | **44,4%** | 13,10 / 13,96 | $\ge$ 11.800 tok/s (VRAM ultra-baixa) |
| *demais variações com frac 0.5/0.25 (selective/ffn)* | - | OOM | - | - | Estouro de memória |

---

## 3. Investigação Detalhada dos Problemas de FP8

### 3.1 Falha no FP8 Tensorwise: Gradientes NaN

Durante a execução da candidata FP8 com `--set fp8=tensorwise`, o console do trainer emitiu (transcrito do terminal):
```text
step 86/572 | loss 14.3847 | gnorm nan | lr 2.55e-03 | 17,229 tok/s | MFU 57.5% | 16.79 GB | seq 4096 | SKIPPED (grad norm não finito)
```

> [!WARNING]
> **Evidência perdida:** `configs/b0_8_fp8_ab_fp8.yaml` usava o mesmo `out_dir` (`runs/fp8_ab/fp8`) para as duas receitas, e o run rowwise subsequente sobrescreveu o `metrics.jsonl` do tensorwise. A linha acima e a afirmação de que todos os passos foram descartados não podem ser auditadas. O config agora grava em `runs/fp8_ab/fp8_rowwise` e o runbook manda passar `--set out_dir=runs/fp8_ab/fp8_tensorwise` ao trocar a receita. Para reabrir a questão, o tensorwise precisa ser re-executado.

#### Mecânica da Falha no Código
Em [sore/train/trainer.py](../sore/train/trainer.py) (função de passo de treino):
```python
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
loss, lm_loss, mtp_loss, norm = torch.cat([sums, grad_norm.reshape(1).to(sums)]).tolist()
finite = math.isfinite(norm)
for opt in optimizers:
    for group in opt.param_groups:
        group["lr"] = group["initial_lr"] * lr_mult
    if finite:
        opt.step()
    opt.zero_grad(set_to_none=True)
return {"loss": loss, "lm_loss": lm_loss, "mtp_loss": mtp_loss, "grad_norm": norm, "skipped": not finite}
```

- Quando `norm` é `NaN`, `finite` torna-se `False` e `opt.step()` é pulado.
- Com todos os passos pulados, os parâmetros nunca são atualizados e a loss não cai.

#### Causa Raiz (confirmada): bug de compilação do Inductor
Investigação posterior no b0_8 e em repros isolados (mesma stack):

| Cenário | Resultado |
|---|---|
| b0_8, tensorwise, **eager** | grad norm 48,7 / 46,4 / 43,0 (BF16: 49,4 / 46,9 / 43,7); sem NaN |
| b0_8, tensorwise, **compile** (com ou sem checkpointing; com ou sem as alavancas do FLA) | NaN desde o passo 1 nas projeções do mixer de todas as camadas |
| Duas `nn.Linear` em tensorwise, compiladas (sem nada do SoreLM) | dW da 2ª camada com NaN em exatamente 1/16 das entradas |
| Idem com `round_scales_to_power_of_2=True` | sem NaN, mas dW com **50% de erro relativo** (silencioso) |
| Idem com Inductor `split_reductions=False` ou `triton.cooperative_reductions=True` | dW igual ao eager (erro vs fp32 de 4,7%, o do próprio FP8) |
| rowwise compilado | igual ao rowwise eager; correto |

- O NaN atinge o dW de um `Float8Linear` cuja entrada é produzida no mesmo grafo compilado. O padrão de 1/16 e o efeito das flags apontam para a redução dividida (split reduction) do amax gerada pelo Inductor no torch 2.14.1 / torchao 0.18.
- Não é limitação numérica do FP8: em eager, o tensorwise é correto.
- Os contornos do Inductor corrigem o resultado, mas inviabilizam o throughput no b0_8 (`ffn, frac 1.0`, 4096x2): `split_reductions=False` → **1.027 tok/s**; `cooperative_reductions` → **8.448 tok/s**; BF16 → 14.292 tok/s.

#### Por que o benchmark não detectou
`scripts/bench.py` executa o passo completo (forward, backward, `clip_grad_norm_` e `opt.step()` com Muon+AdamW), mas **descarta o valor retornado por `clip_grad_norm_`** e nunca verifica se é finito. O tempo medido é o de um passo real; o que falta é a checagem de que o passo produziu gradientes válidos. Com gradientes NaN, o throughput reportado (17.691 tok/s) não corresponde a um treino utilizável.

---

### 3.2 Avaliação do FP8 Rowwise: Estabilidade vs Sobrecusto

Com `fp8=rowwise` (padrão de [configs/b0_8_fp8_ab_fp8.yaml](../configs/b0_8_fp8_ab_fp8.yaml)), o run foi interrompido após **18 passos** (log em `runs/fp8_ab/fp8_rowwise/metrics.jsonl`):

1. **Estabilidade Numérica (18 passos):**
   - `skipped: false` em todos os 18 passos.
   - `gnorm` entre 4,55 e 25,68, com pico de **25,68 no passo 5**. O BF16 teve pico equivalente no mesmo passo (26,77), então o pico é do início do warmup, não do FP8.
   - `loss` de `14,742` (passo 1) para `11,710` (passo 18); no BF16, `14,741` → `11,707` no mesmo passo.
   - 18 passos são ~3% do A/B. **O critério de qualidade (loss final FP8 $\le$ 1,01 × BF16) não foi avaliado.**
2. **Degradação de Throughput:**
   - BF16: **~14.720 tok/s** (passos de ~35,6 s).
   - FP8 Rowwise: **~13.650 tok/s** (passos de ~38,4 s).
   - O FP8 Rowwise é **~7,3% mais lento** que o BF16 (equivalentemente, o BF16 é **~7,8% mais rápido**).
3. **Desempenho por camada (fwd+bwd compilado, 8192 tokens, µs):**

   | Camada (in → out) | BF16 | rowwise | tensorwise | rowwise vs BF16 |
   |---|---|---|---|---|
   | GDN q/k (1280 → 1280) | 598 | 1.070 | 538 | 0,56x |
   | GDN v/g (1280 → 2560) | 1.165 | 1.445 | 877 | 0,81x |
   | GDN o (2560 → 1280) | 1.187 | 1.994 | 951 | 0,60x |
   | Atenção k/v (1280 → 256) | 179 | 752 | 268 | 0,24x |
   | FFN gate/up (1280 → 4352) | 1.946 | 2.004 | 1.432 | 0,97x |
   | FFN down (4352 → 1280) | 2.028 | 3.213 | 1.609 | 0,63x |

   O rowwise perde do BF16 em **todas** as formas de GEMM do b0_8, então restringir o FP8 às camadas grandes não resolve. O tensorwise ganharia ~1,25x no FFN e no v/g/o, mas é o caso que o compile quebra. *(Os tempos do tensorwise nesta tabela medem só a velocidade; o dW dele compilado está errado.)*

---

## 4. Comparativo Direto: BF16 vs FP8 Rowwise (b0_8 na RTX 4090)

| Métrica | Treino BF16 (`checkpoint_policy=ffn`) | Treino FP8 Rowwise (`checkpoint_policy=ffn`) | Vantagem |
|---|---|---|---|
| **Passos executados** | 572 / 572 | 18 / 572 (interrompido) | — |
| **Estabilidade Numérica** | 0 skips; `gnorm` 0,05–26,77 (pico no passo 5, warmup) | 0 skips; `gnorm` 4,55–25,68 (pico no passo 5) | Sem diferença observada em 18 passos |
| **Qualidade (loss final)** | 2,91 | não medida | Indeterminado |
| **Throughput Real de Treino** | **~14.720 tok/s** | **~13.650 tok/s** | **BF16 (+7,8%)** |
| **MFU Efetivo** | **48,9%** | **45,3%** | **BF16** |
| **Duração do Pré-Treino (50B)** | **~39,3 dias (~5,6 semanas)** | **~42,4 dias (~6,1 semanas)** | **BF16 (-3,1 dias)** |
| **Fidelidade Numérica** | Exata | Lossy (sujeita ao A/B de gap de perda) | **BF16** |
| **VRAM Alocada** | **22,37 GB** pico em 572 passos (reservado 22,97 GB) | **20,27 GB** no passo 18 (run incompleto) | FP8 (~-2,1 GB) |

> [!IMPORTANT]
> O pico de **22,37 GB** do BF16 **viola** o critério "VRAM pico (alocado) ≤ 22 GB" do runbook da 4090 ([`RUNBOOK_4090.md`](RUNBOOK_4090.md)). O run completou 572 passos sem OOM, mas apenas no estágio 4096x2; o estágio 8192x1 do pré-treino não foi validado em treino real (no benchmark, ambos os estágios ficam em 22,36 GB).

---

## 5. Recomendações e Alterações no Código

### 1. Atualizar [scripts/bench.py](../scripts/bench.py) *(feito)*
Guardar o retorno do `clip_grad_norm_` que `train_step` já executa e verificá-lo **fora** da região cronometrada (sem segunda chamada de clip e sem `.item()` entre os eventos CUDA):
```python
# em train_step: devolver o norm em vez de descartá-lo
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
...
return wall, phases, grad_norm

# em main, depois de medir o passo:
wall, step_phases, grad_norm = train_step(model, optimizers, make_batches, device)
if not torch.isfinite(grad_norm):
    print(f"[bench] AVISO: grad_norm não finito no passo {step + 1}", flush=True)
```
O resultado sai com `grad_finite: false` no `bench.jsonl`, e o `summarize_bench.py` o marca como inválido, o põe no fim da tabela e o exclui da escolha da melhor configuração exata.

### 2. Recusar `fp8="tensorwise"` com `compile=True` *(feito)*
Com a causa confirmada (§3.1), `TrainConfig` e `scripts/bench.py` recusam a combinação, e o `day1_bench.sh` não roda mais o tensorwise. Vale reavaliar após atualizar o torch/torchao: o repro de duas `nn.Linear` compiladas basta para saber se o bug foi corrigido.

### 3. Configuração Oficial de Pré-Treino em [configs/b0_8_pretrain.yaml](../configs/b0_8_pretrain.yaml) *(feito: opção dentro do critério)*
FP8 fica desligado (`fp8: "off"`). Para o checkpointing, há duas opções medidas:

- **Dentro do critério do runbook (adotada):**
  ```yaml
  fp8: "off"
  model_overrides:
    grad_checkpoint: true
    checkpoint_policy: "full"
    checkpoint_frac: 0.5
    fused_ce: true
    fused_norm_gate: true
    gdn_fla_conv: true
    gdn_kernel_gate: true
  ```
  *13.550 tok/s no benchmark, 19,94 GB alocados; ~42,7 dias para 50B tokens. Não foi executada no A/B de 572 passos.*

- **Mais rápida, mas fora do critério de VRAM:**
  ```yaml
  fp8: "off"
  model_overrides:
    grad_checkpoint: true
    checkpoint_policy: "ffn"
    checkpoint_frac: 1.0
  ```
  *~14.720 tok/s no treino real, 22,37 GB alocados; ~39,3 dias para 50B tokens.* Adotá-la exige revisar formalmente o critério de 22 GB no runbook e validar o estágio 8192x1 em treino real antes de disparar os 50B tokens.
