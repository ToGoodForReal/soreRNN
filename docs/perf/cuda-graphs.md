# Parecer Técnico: Viabilidade de CUDA Graphs em Blocos GDN com FLA Varlen

Este documento apresenta a análise de viabilidade técnica e empírica da captura de passos de treino ou camadas [`Block`](file:///C:/Users/wellington.lara/Documents/GitHub/soreRNN/sore/model/block.py) do soreRNN em **CUDA Graphs** (`torch.cuda.CUDAGraph`, `torch.cuda.make_graphed_callables` ou `torch.compile(mode="reduce-overhead")`), considerando o empacotamento de documentos com comprimentos variáveis (`varlen`) e a biblioteca Flash Linear Attention (`fla` 0.5.2).

---

## 1. Resumo Executivo e Recomendação Objetiva

> [!IMPORTANT]
> **Recomendação Definitiva:** **NÃO IMPLEMENTAR CUDA Graphs para os Blocks GDN com FLA varlen.**
>
> 1. **Incompatibilidade Fundamental com CUDA Graphs:** CUDA Graphs exigem dimensões estáticas de grid e alocações imutáveis de memória para cada nó de lançamento no fluxo CUDA. No algoritmo `chunk_gated_delta_rule` com empacotamento varlen, o número de chunks e o grid dos kernels Triton variam dinamicamente a cada micro-batch conforme os comprimentos dos documentos empacotados.
> 2. **Falha Silenciosa de Corretude no Replay:** A execução de um CUDA Graph capturado sob uma distribuição de documentos e reexecutado com uma nova distribuição resulta em **erro numérico massivo** (diferença máxima absoluta $> 1,02$ em tensores normalizados, correspondendo a ~100% de erro relativo), pois o grafo relança os kernels Triton com o grid anterior e índices desatualizados.
> 3. **Padding de Chunks é Inviável e Antieconômico:** O FLA 0.5.2 não suporta mascaramento de chunks fictícios (*dummy chunks*). Fazer padding exigiria bifurcar e reescrever ~8 kernels Triton (forward e backward) para mascarar estados recorrentes $h_t$, com alto risco de instabilidade matemática e perda de desempenho por processamento de blocos vazios.
> 4. **Retorno Marginal Inexistente:** O tempo de espera da GPU pela CPU (`idle`) medido na RTX 3050 é de apenas **2,9% a 3,0%** no `s150m` e **8,7%** na fatia do `b0_8` com `grad_accum=2`. No treino de produção na RTX 4090 com `grad_accum=64`, o custo de despacho da CPU dilui-se para **$< 0,5\%$ do passo**, tornando o ganho de CUDA Graphs totalmente insignificante frente aos custos e riscos.

---

## 2. Evidência Empírica: Experimento de Captura de CUDA Graph

Para validar o comportamento do ecossistema PyTorch e FLA sob CUDA Graphs, foi executado um conjunto de testes empíricos com uma camada GDN real (`tiny`, $d=128$, `gdn_backend="fla"`, kernels fundidos ativos) submetida a três abordagens distintas de captura com `cu_seqlens` variáveis.

### 2.1 Script de Teste Executado

O ensaio foi executado no WSL2 com o script [`scripts/test_cuda_graph_gdn.py`](file:///C:/Users/wellington.lara/Documents/GitHub/soreRNN/scripts/test_cuda_graph_gdn.py):
```bash
wsl.exe -d Ubuntu-24.04 -u wellington --exec bash -c \
    'cd /mnt/c/Users/wellington.lara/Documents/GitHub/soreRNN && \
     export PATH=$HOME/.local/bin:$PATH UV_PROJECT_ENVIRONMENT=$HOME/.venvs/sorernn UV_LINK_MODE=copy && \
     uv run python scripts/test_cuda_graph_gdn.py'
```

### 2.2 Resultados Observados

#### Caso 1: `torch.compile(mode="reduce-overhead")`
- **Comportamento:** O compilador PyTorch Dynamo intercepta os métodos `chunk_indices_without_sync` e `chunk_gated_delta_rule` através de graph breaks (`torch.compiler.disable`).
- **Resultado:** A compilação executa sem erro fatal aparente porque o Inductor **isola o kernel GDN fora do CUDA Graph**. Apenas os subgrafos ATen elementares (como as normas e a FFN) são capturados em grafos parciais, enquanto toda a recorrência GDN continua sendo despachada em modo eager pelo runtime do Python, mantendo o overhead de host inalterado.

#### Caso 2: Captura Manual via `torch.cuda.CUDAGraph()`
- **Procedimento:** Warmup com `cu_seqlens = [0, 256, 512]` (2 documentos de 256 tokens), captura do Block dentro de `torch.cuda.graph(g, stream=s)`, e subsequente replay com um novo particionamento `cu_seqlens = [0, 128, 512]` (documentos de 128 e 384 tokens) atualizado in-place.
- **Saída do Terminal:**
```text
=== Teste 2: torch.cuda.CUDAGraph() manual ===
Iniciando captura de CUDA graph com static_cu = [0, 256, 512]...
Captura do CUDA graph bem sucedida!
Replay com mesma entrada...
Replay com mesma entrada OK!

Atualizando static_cu in-place para [0, 128, 512] (novo particionamento)...
Replay executou sem erro no driver CUDA.
Diferença máxima absoluta entre Replay e Saída Correta: 1.023438
FALHA DE CORRETUDE: O replay do CUDA Graph NÃO produziu a computação correta para o novo cu_seqlens!
Causa fundamental: no CUDA graph, os nós dos kernels Triton são gravados com os grids e ponteiros de chunk fixados na captura.
```

#### Caso 3: Captura via `torch.cuda.make_graphed_callables`
- **Saída do Terminal:**
```text
=== Teste 3: torch.cuda.make_graphed_callables ===
make_graphed_callables concluído!
Execução 1 com sample_args OK!
Chamada 2 com novo cu_seqlens OK!
Diferença máxima Chamada 2 vs esperado: 1.359375
```

---

## 3. Análise Técnica do Problema do Grid de Chunks

### 3.1 Como o FLA Determina o Grid de Lançamento

No algoritmo `chunk_gated_delta_rule` (e em `causal_conv1d`), a sequência total $T$ de tokens é segmentada em blocos de tamanho fixo $BT=64$. Quando documentos múltiplos são empacotados num micro-batch (`varlen`), as fronteiras dos documentos não coincidem necessariamente com múltiplos de 64.

Para impedir que a atenção recorrente cruze a fronteira entre documentos distintos, o FLA segmenta cada documento separadamente:
$$\text{n\_chunks} = \sum_{i=0}^{S-1} \left\lceil \frac{\text{cu\_seqlens}[i+1] - \text{cu\_seqlens}[i]}{BT} \right\rceil$$

A função `prepare_chunk_indices` gera uma tabela mapeando cada chunk para seu respectivo documento e índice intra-documento: `[n_chunks, 2]`. 

Ao disparar os kernels Triton (`chunk_gated_delta_rule_fwd_intra`, `chunk_gated_delta_rule_fwd_h`, `chunk_fwd_o` e seus backward correspondentes), o grid de lançamento da GPU é configurado como:
```python
NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
chunk_gated_delta_rule_fwd_kkt_solve_kernel[(NT, B * HV)](...)
```

### 3.2 O Conflito Irreconciliável com CUDA Graphs

1. **Topologia e Grids Estáticos:** Um CUDA Graph é uma gravação estática de operações ao nível de hardware. Quando um nó de kernel é capturado, suas dimensões de bloco de threads e grid (`gridDim.x = NT`, `gridDim.y = B * HV`) são gravadas de forma imutável nos registradores do grafo na GPU.
2. **Número Variável de Chunks:** Na amostragem de dados reais (`doc_lengths.json`), um micro-batch de 2.048 tokens pode conter 2 documentos longos ($\text{n\_chunks} = 32$) ou 8 documentos curtos ($\text{n\_chunks} = 38$). No momento em que o número de documentos ou comprimentos varia, o grid necessário muda.
3. **Ponteiros de Índices Congelados:** Mesmo que o número total de chunks permanecesse numericamente idêntico, a matriz de mapeamento `chunk_indices` (que dita quais tokens pertencem a qual documento e zera o estado $h$ na fronteira) reside num endereço de memória congelado na captura. Alterações dinâmicas não alteram a sequência de offsets capturada pelos nós do grafo.

Por essas razões, o replay do CUDA Graph executa a computação matemática como se os documentos ainda tivessem os comprimentos do lote original de warmup, resultando em corrupção catastrófica de gradientes.

---

## 4. Análise de Viabilidade de Padding de Chunks a um Máximo Fixo

Uma proposta teórica para contornar o problema do grid dinâmico seria fixar o número de chunks a um valor máximo fixo $\text{MAX\_CHUNKS}$:
$$\text{MAX\_CHUNKS} = \left\lceil \frac{T}{BT} \right\rceil + S_{max}$$

Abaixo analisa-se detalhadamente a viabilidade prática dessa saída:

### 4.1 O que exigiria o Padding de Chunks
1. **Tensores Estáticos:** Pré-alocar tensores de índices `chunk_indices` com shape fixo `[MAX_CHUNKS, 2]`, preenchendo as entradas não utilizadas com valores sentinela (ex: `-1`).
2. **Modificação dos Kernels Triton de Recorrência:**
   - No kernel `fused_kkt_solve`: o loop principal precisaria testar se o chunk atual é válido. Para chunks inválidos, a matriz $A$ precisaria ser preenchida como a matriz identidade para evitar divisões por zero ou NaNs.
   - No kernel `chunk_gated_delta_rule_fwd_h`: este é o coração da recorrência, onde o estado oculto $h_t$ passa de chunk em chunk:
     $$h_{k+1} = h_k \cdot \exp(g_k) + w_k^T u_k$$
     Se um chunk for fictício (padding), o kernel precisa garantir que $h_{k+1} = h_k$. Caso contrário, o estado acumulado de um documento real seria contaminado com ruído numérico.
   - Nos kernels de projeção de saída `chunk_fwd_o`: mascarar saídas para evitar escrita em memória fora dos limites do lote de tokens.
   - **Nos Kernels de Backward (4 kernels adicionais):** Toda a passagem backward de gradientes para $q, k, v, \beta, g$ precisaria replicar a lógica idêntica de mascaramento para não retropropagar gradientes fictícios.

### 4.2 Trade-offs e Custos de Engenharia

| Dimensão | Avaliação | Detalhes |
|---|:---:|---|
| **Custo de Engenharia** | **Crítico / Inaceitável** | Exige a criação e manutenção de um fork profundo da biblioteca externa `flash-linear-attention`, reimplementando 8 kernels Triton altamente otimizados em nível de warp e assembly PTX. |
| **Risco de Corretude** | **Altíssimo** | Qualquer discrepância na transmissão do estado $h$ em chunks sentinela corrompe a estabilidade do pré-treino, gerando divergência de loss difícil de diagnosticar. |
| **Eficiência Computacional** | **Negativa** | O padding a um máximo fixo aloca threads e SMs da GPU para calcular chunks vazios. Em sequências com muitos documentos pequenos, o grid inflacionado consome ciclos úteis da GPU, anulando o benefício de eliminar o overhead da CPU. |
| **Gargalo Real do Sistema** | **Computação Densa (GEMMs)** | Conforme comprovado no perfil da RTX 3050 (`docs/perf/profile-3050.md`), **48% a 61% do tempo é consumido por Matmuls** e **19% a 30% por Elementwise**. O tempo de CPU ociosa é de apenas ~3%. |

---

## 5. Conclusão e Diretrizes de Arquitetura

1. **CUDA Graphs descartados para a arquitetura com GDN:** A combinação de modelos lineares recorrentes com empacotamento sem padding (*varlen packing*) é conceitualmente antagônica à rigidez de topologia estática exigida por CUDA Graphs.
2. **A estratégia adotada no soreRNN é a correta:**
   - Compilação seletiva in-place por Block via `torch.compile` (`mode=None` ou `max-autotune-no-cudagraphs`), isolando o kernel do FLA com graph breaks controlados;
   - Eliminação de sincronizações host-GPU através de tensores `cu_seqlens_cpu` pré-transferidos de forma assíncrona (Task 4);
   - Foco dos esforços de otimização onde residem os 80% do tempo real: GEMMs em FP8 (alavanca G) e fusão de projeções lineares e ativações SwiGLU.
