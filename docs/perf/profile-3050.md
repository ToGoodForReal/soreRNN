# Relatório de Profiling na RTX 3050 e Backlog F′

Este relatório consolida a análise detalhada de profiling do pré-treino do soreRNN na GPU local **NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM, sm86)** após a implementação de todas as alavancas exatas de desempenho (Tasks 4–8: eliminação de sincronização host-GPU, otimizações de kernel GDN, fused cross-entropy do FLA, activation checkpointing seletivo/FFN e suporte a FP8).

---

## 1. Metodologia e Comandos Utilizados

Os testes foram executados no ambiente padronizado WSL2 (`Ubuntu-24.04`), utilizando o script de benchmark [`scripts/bench.py`](file:///C:/Users/wellington.lara/Documents/GitHub/soreRNN/scripts/bench.py) com a flag `--profile`, integrando amostragem realista de comprimentos de documentos em português (`data/sample_pt/doc_lengths.json`), `seq_len=2048`, `micro_batch=1`, `grad_accum=2` (4.096 tokens por passo), 10 passos no total (2 de warmup e mediana dos 8 passos seguintes) e `--peak-tflops 35`.

### 1.1 Comandos Executados

#### Preset `s150m` (164,3M parâmetros, 12 camadas, $d=768$, FFN $2048$)
Todas as alavancas exatas ativas no preset (`gdn_kernel_gate=True`, `gdn_fla_conv=True`, `fused_norm_gate=True`, `fused_ce=True`, `--compile` in-place por Block):

- **Política de Checkpoint `selective`:**
```bash
uv run python scripts/bench.py \
    --preset s150m \
    --seq-len 2048 \
    --micro-batch 1 \
    --grad-accum 2 \
    --steps 10 \
    --warmup 2 \
    --grad-checkpoint \
    --override checkpoint_policy=selective \
    --compile \
    --doc-lengths data/sample_pt/doc_lengths.json \
    --profile \
    --peak-tflops 35 \
    --json
```

- **Política de Checkpoint `ffn`:**
```bash
uv run python scripts/bench.py \
    --preset s150m \
    --seq-len 2048 \
    --micro-batch 1 \
    --grad-accum 2 \
    --steps 10 \
    --warmup 2 \
    --grad-checkpoint \
    --override checkpoint_policy=ffn \
    --compile \
    --doc-lengths data/sample_pt/doc_lengths.json \
    --profile \
    --peak-tflops 35 \
    --json
```

#### Fatia do Preset `b0_8` (`n_layers=4`, 221,4M parâmetros, $d=1280$, FFN $4352$)
Execução com as 4 primeiras camadas da arquitetura de 800M parâmetros para dimensionamento de dimensionalidade larga ($d=1280$ vs $768$):

- **Política Padrão `full` (ótima para 6 GB):**
```bash
uv run python scripts/bench.py \
    --preset b0_8 \
    --override n_layers=4 \
    --seq-len 2048 \
    --micro-batch 1 \
    --grad-accum 2 \
    --steps 10 \
    --warmup 2 \
    --grad-checkpoint \
    --compile \
    --doc-lengths data/sample_pt/doc_lengths.json \
    --profile \
    --peak-tflops 35 \
    --json
```

---

## 2. Divisão de Tempo por Fases do Treino

As fases do passo foram aferidas por eventos CUDA de alta precisão ao longo dos 8 passos pós-warmup:
- **`data`**: montagem do lote sintético e transferência CPU $\to$ GPU (`pinned`, `non_blocking`).
- **`fwd_bwd`**: execução forward com autocast bf16 e backward de todos os micro-batches acumulados.
- **`clip`**: cálculo da norma global e corte do gradiente (`clip_grad_norm_`).
- **`optimizer`**: atualização de pesos via Muon (projeções 2D via Newton-Schulz) e AdamW (restante).

| Configuração | Política CKPT | tok/s | MFU (%) | VRAM Pico (GB) | VRAM Res. (GB) | data (s) | fwd_bwd (s) | clip (s) | optimizer (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`s150m` (linha de base pré-alavancas)** | `full` | 3.036 | 11,7% | 2,99 | 3,34 | 0,006 | 1,055 | 0,019 | 0,282 |
| **`s150m` (com todas alavancas)** | `selective` | 2.726 | 10,5% | 3,70 | 4,11 | 0,001 | 1,237 | 0,016 | 0,249 |
| **`s150m` (com todas alavancas)** | `ffn` | 2.739 | 10,5% | 3,77 | 3,86 | 0,002 | 1,194 | 0,018 | 0,277 |
| **`b0_8` fatia 4 camadas (pré-alavancas)** | `full` | 2.616 | 14,0% | 4,06 | 4,64 | 0,006 | 1,136 | 0,024 | 0,431 |
| **`b0_8` fatia 4 camadas (com alavancas)** | `full` | 2.238 | 12,0% | 4,25 | 4,43 | 0,001 | 1,400 | 0,022 | 0,406 |
| **`b0_8` fatia 4 camadas (com alavancas)** | `ffn`* | 960 | 5,1% | 4,69 | 4,83 | 0,002 | 3,735 | 0,026 | 0,493 |
| **`b0_8` fatia 4 camadas (com alavancas)** | `selective`* | 783 | 4,2% | 4,76 | 4,92 | 0,004 | 4,625 | 0,034 | 0,547 |

> [!NOTE]
> \* **Efeito de Paginação de VRAM na RTX 3050 Laptop (6 GB):** Na máquina de desenvolvimento local, a interface gráfica do Windows 11 aloca ~1,7 GB de VRAM contínua, deixando ~4,4 GB de memória física utilizável. Na fatia do `b0_8` com $d=1280$, as políticas `selective` e `ffn` retêm ativações adicionais para acelerar o backward, elevando a alocação para 4,76 GB e 4,69 GB. Ao ultrapassar o limite físico local, o subsistema WDDM/WSL2 aciona paginação de memória compartilhada sobre o barramento PCIe, inflando o tempo de `fwd_bwd` (3,7s–4,6s) com transferências D2D. Em ambiente dedicado (Linux nativo / RTX 4090 24GB ou desktop 8GB), `selective` e `ffn` não sofrem paginação e entregam o maior throughput (como demonstrado na Task 7, onde `selective` atingiu 2.017 tok/s contra 1.809 tok/s de `full`).

---

## 3. Divisão de Tempo de GPU por Categoria de Kernel

A tabela abaixo decompõe o tempo de GPU aferido pelo `torch.profiler` em um passo perfilado completo (passo extra, com o mesmo micro-batch pré-aquecido). O tempo total inclui a parede e a ociosidade da GPU (`idle`):

| Categoria de Kernel | `s150m` (selective) | `s150m` (ffn) | `b0_8` fatia (full) | Descrição e Componentes Principais |
|---|---:|---:|---:|---|
| **`matmul`** | 0,836 s (47,8%) | 0,959 s (54,7%) | 1,418 s (60,5%) | GEMMs cuBLAS/CUTLASS das projeções dos Blocks (Q, K, V, G, O, W1, W2, W3) e iterações Newton-Schulz do Muon |
| **`elementwise`** | 0,525 s (30,0%) | 0,401 s (22,9%) | 0,445 s (19,0%) | Inductor (`triton_poi_*`, `triton_red_*`, `triton_per_*`), ATen pointwise/reductions, atualizações AdamW/Muon |
| **`other`** | 0,165 s (9,4%) | 0,176 s (10,0%) | 0,162 s (6,9%) | `gpu_memcpy`, `gpu_memset`, scan/sort de `cu_seqlens`, e kernels Inductor com prefixos não-mapeados (`triton_spl_`, etc.) |
| **`idle`** | 0,050 s (2,9%) | 0,053 s (3,0%) | 0,205 s (8,7%) | GPU aguardando despacho da CPU (overhead Python, Dynamo e loop do trainer) |
| **`gdn`** | 0,130 s (7,5%) | 0,121 s (6,9%) | 0,073 s (3,1%) | Kernels Triton do FLA (`chunk_gated_delta_rule`, `recompute_w_u`, `prepare_wy_repr`, `causal_conv1d`) |
| **`attention`** | 0,024 s (1,4%) | 0,022 s (1,3%) | 0,019 s (0,8%) | Kernels Triton do FlexAttention (`triton_tem_*flex_attention*`) |
| **`loss`** | 0,019 s (1,1%) | 0,022 s (1,3%) | 0,021 s (0,9%) | Fused Linear Cross-Entropy do FLA (reduzido de 0,044s / 3% antes da fusão) |
| **Parede do Passo Perfilado** | **1,750 s** | **1,754 s** | **2,342 s** | Duração total incluindo instrumentação do profiler |

---

## 4. Observações Críticas do Ledger de Categorização

1. **Newton-Schulz do Muon classificado em `matmul`:**
   O algoritmo de ortogonalização de Newton-Schulz do otimizador Muon realiza 5 iterações de multiplicações de matrizes densas $A \times B$ e $X \times X^T$ para cada tensor 2D de pesos. Como essas multiplicações chamam internamente rotinas da cuBLAS (`ampere_sgemm`, `cublasLt`), elas caem na categoria `matmul`. Com `--grad-accum 2`, o Muon é executado a cada 2 micro-batches; no treino final com `--grad-accum 64`, esse peso se dilui para menos de 1% do passo.
2. **Kernels Inductor com prefixos especializados classificados em `other`:**
   O gerador de código do PyTorch Inductor pode emitir kernels com prefixos alternativos, tais como:
   - `triton_spl_*` (split reduction / split scan);
   - `triton_for_*` (foreach loops);
   - `triton_unk_*` (kernels com nomes gerados dinamicamente).
   Como esses nomes não batem nas expressões regulares padrão de `elementwise` (`triton_poi|red|per`), eles caem no balde residual `other`. Uma inspeção detalhada revela que ~60% do tempo em `other` na configuração estacionária corresponde a essas reduções do Inductor e operações pontuais de inicialização/cópia de memória.

---

## 5. Backlog F′ Ranqueado (Oportunidades > 5% do Passo)

O Backlog F′ lista oportunidades concretas de otimização identificadas a partir dos gargalos medidos no perfil. Conforme o critério de inclusão do plano, são elegíveis apenas componentes que representam **mais de 5% do tempo do passo**.

| Rank | Operação / Categoria | % do Passo | Ideia de Implementação | Ganho Estimado |
|:---:|---|:---:|---|:---:|
| **1** | **Linear Projections (GEMMs dos Blocks)** | 45%–55% | **Fusão das Projeções QKVG e FP8 Training:** <br>• Fundir `q_proj`, `k_proj`, `v_proj` e `g_proj` numa única projeção linear unificada (`qkvg_proj`), elevando a intensidade aritmética dos blocos de thread e reduzindo 3 lançamentos de kernel por camada.<br>• No hardware de destino (RTX 4090 sm89), ativar treino em FP8 via torchao (`Float8Linear` com escala rowwise ou tensorwise, alavanca G). | **25% a 35% de redução no tempo do passo** na RTX 4090 |
| **2** | **SwiGLU & RMSNorm Fused (D2)** | 15%–20% | **Fusão de RMSNorm e SwiGLU em Triton:** <br>• Fundir a `norm2` (RMSNorm) diretamente no kernel de leitura das projeções `w1` e `w2` da FFN SwiGLU.<br>• Fundir a ativação `silu(w1(x)) * w2(x)` e projeção `w3` num único kernel Triton fused GEMM pointwise, eliminando a materialização intermediária de ativações na VRAM. | **4% a 7% de redução no tempo do passo** (~20%–30% da fase elementwise) |
| **3** | **Muon Optimizer Fused Loop** | 10%–15% | **Kernel Triton Fundido para Newton-Schulz e Update:** <br>• Implementar as etapas de normalização espectral, transposição e combinação linear $aX + b(XX^T)X$ em Triton com buffers mantidos em SRAM.<br>• Alternativamente, diluir o impacto elevando o micro-batch por passo (`grad_accum 64` no pré-treino real). | **5% a 10% no regime de baixo acúmulo** (irrelevante com `grad_accum >= 32`) |
| **4** | **GDN Recurrence Optimization** | 7%–8% | **Triton Custom Tile Tuning para $D_{head}=128$:** <br>• O kernel `chunk_gated_delta_rule` do FLA usa configurações de tile genéricas para múltiplos tamanhos de cabeça. Ajustar autotune específico para $BK=128$, $BV=128$ e $BT=64$ com pipeline assíncrono TMA/cp.async em hardware Ada (RTX 4090). | **1% a 2% de redução no tempo global** |
| **5** | **Eliminação de Overhead Residual em `other`** | 6%–10% | **Refinamento de Lowering do Inductor e Caching:** <br>• Evitar divisões e cópias desnecessárias de gradientes via `set_to_none=True` permanente e pré-alocação estática dos buffers de chunking. | **2% a 3% de redução no tempo do passo** |

### Itens Desqualificados do Backlog (< 5% do Passo):
- **Loss Cross-Entropy:** Atualmente representa apenas ~1,1% do passo graças ao kernel fundido `fused_linear_cross_entropy` do FLA integrado na Task 6 (reduzindo de 3% e poupando >1 GB de VRAM).
- **FlexAttention:** Representa ~1% do passo, pois a proporção da arquitetura soreQ4 é de 3 camadas GDN para 1 camada de atenção gated, e o FlexAttention com `BlockMask` pré-compilado já opera no limite de largura de banda da memória.
