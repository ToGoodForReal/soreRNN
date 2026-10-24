# soreRNN v2 — Desempenho do pré-treino (meta: MFU ≥ 50% na RTX 4090)

Data: 2026-10-24 · Branch: `feat/perf-mfu` (a partir de `main` 400b8e9) · Spec-mãe: `docs/superpowers/specs/2026-10-23-sorernn-v2-design.md` (§6)

## 1. Intenção e porquê

O pré-treino do preset `b0_8` (781.242.576 params, 50B tokens, `configs/b0_8_pretrain.yaml`) roda numa única RTX 4090 24GB, que chega **em poucas semanas**. Até lá só existem duas RTX 3050 (notebook 6 GB em WSL2, onde está o ambiente, e um desktop 8 GB), sem NVLink e sem FP8.

O prazo de 6–7 semanas para os 50B tokens exige 11,8K–13,8K tok/s (tabela do `docs/RUNBOOK_4090.md` §2, com 5,48 GFLOP/token a 4K e pico de 165 TFLOPS). O usuário fixou a meta em **MFU ≥ 50%**, ou seja, **≥ ~15K tok/s**, o que dá ~39 dias (~5,5 semanas). Hoje, na 3050, o `s150m`@2048 faz 10–12% de MFU compilado. Esse número não prevê a 4090, mas mostra que há trabalho a fazer.

Este projeto deixa o treino o mais rápido possível **antes** da 4090 chegar, usando o que dá para fazer e validar nas 3050. Também entrega um protocolo de "dia 1" para medir, ligar as otimizações e escolher a configuração final na 4090 em poucas horas.

## 2. Metas, não-metas e anti-metas

**Metas**
1. `b0_8` na 4090 a **≥ 15.000 tok/s** nos dois estágios (4096×2 e 8192×1, `grad_accum` 64), com pico reservado ≤ 23 GB. O piso aceitável é 11.800 tok/s (7 semanas).
2. Cada otimização é uma opção de config independente, medível sozinha pelo benchmark.
3. Um benchmark que mostra **para onde vai o tempo do passo**: kernels do GDN, atenção, matmuls, loss, otimizador e tempo ocioso. Assim a escolha na 4090 sai de medição, não de palpite.
4. Protocolo de dia 1 no runbook: do `nvidia-smi` até a config final gravada em `b0_8_pretrain.yaml`, com os números medidos.

**Não-metas**
- Mudar a arquitetura (dimensões, padrão 3:1, vocab, MTP) para ganhar velocidade. A arquitetura é a da spec-mãe.
- Multi-GPU, inclusive usar as duas 3050 juntas.
- Velocidade de inferência.
- Atualizar torch, triton ou FLA. As versões continuam travadas.

**Anti-metas** (fracasso mesmo que "funcione")
- Ficar mais rápido aprendendo pior além da tolerância da §3.
- Ganho que só existe na 3050 e não aparece na 4090.
- Otimização que falha em silêncio: FP8 que não liga, compile que cai para eager, kernel fundido que volta ao caminho lento. Toda otimização ligada que não puder rodar levanta erro ou registra um aviso explícito no log.

## 3. Regra de qualidade (decidida pelo usuário)

Há duas classes de otimização:

- **Exatas:** mudam só a ordem ou a forma do cálculo, como checkpoint seletivo, kernels fundidos, menos sincronizações, pipeline de entrada e ajustes de compile.
  - Gate: testes de paridade contra o caminho atual (loss e gradientes dentro das tolerâncias já usadas na suíte, por exemplo rel 2e-3 em fp32) e uma corrida curta de `s150m` na 3050 com curva de loss igual à de referência dentro do ruído.
  - O ruído é medido rodando a referência com duas seeds.
  - Passando no gate, a otimização fica **ligada por padrão**.
- **Com perda:** FP8 e, se ainda faltar, estados do otimizador em bf16.
  - O usuário aceita **até ~1% de loss final pior**, mas só se a otimização for o que faz o prazo fechar.
  - Regra (decidida pelo usuário): uma otimização com perda só é ligada se a configuração exata, medida na 4090, ficar **abaixo de 15.000 tok/s**, a meta de 50%.
  - É provável que um híbrido em bf16 fique entre 12K e 14K. Por isso o FP8 está no **caminho crítico do dia 1**: chega à 4090 pronto, testado e com o A/B preparado.
  - Antes de ligar, um A/B na 4090 contra bf16 (mesmos dados e passos, corrida curta do `b0_8`) precisa mostrar loss final ≤ 1% pior.
  - Ficam **desligadas por padrão**.

**O que conta como "medido"** (minha decisão): o trainer rodando sobre shards PT reais, quando existirem. Antes disso, o `bench.py` com uma distribuição de tamanhos de documento configurável, ajustada ao histograma de uma amostra real. O sintético atual (4 documentos por janela) não serve para decidir um limiar, porque o tamanho dos documentos muda a esparsidade do flex e o número de chunks do GDN.

## 4. Ponto de partida

O que já existe em `main` 400b8e9:
- Compile por Block (R25).
- Checkpoint de gradiente do Block inteiro (`torch.utils.checkpoint`).
- CE em chunks com checkpoint (`loss_chunk_size` 8192).
- Uma sincronização host↔GPU por passo (`trainer.py:157`).
- AdamW `fused`, `torch.optim.Muon` (`match_rms_adamw`).
- `BlockMask` do flex montado uma vez por forward (R24).
- `chunk_gated_delta_rule` varlen do FLA 0.5.2.
- `scripts/bench.py` (`--compile`, `--grad-checkpoint`, `--grad-accum`) e `sore/model/flops.py`.

`b0_8`:
- d=1280, 24 camadas (18 GDN e 6 de atenção);
- FFN 4352, vocab 65.536 amarrado, MTP 1;
- 524.288 tokens por passo.

**Preset-alvo** (minha decisão): o `b0_8` como está em `main`, com `residual: plain`, sem engram e com Muon + AdamW.
- Os gates de memória e throughput valem para esse preset.
- O `residual: hyper` recebe só testes de paridade (correção).
- Se as ablações em 150M (plano 4 da spec-mãe) mudarem o preset, o passo de memória e throughput do protocolo de dia 1 é refeito.

Estimativa de memória:
- Estado de treino ~10 GB:
  - pesos e gradientes fp32: 6,2 GB;
  - momentum do Muon: ~2,8 GB;
  - m/v do AdamW no embedding: ~0,7 GB.
- Sobram ~12–13 GB para ativações. É isso que decide quanto checkpoint é preciso.

## 5. Alavancas (o como)

### Exatas (feitas e validadas nas 3050)

**A. Checkpoint seletivo/parcial.** O checkpoint do Block inteiro refaz um forward completo, ~33% de compute a mais que o MFU não conta. Modos de config:
- `none`;
- `full` (o atual);
- `selective`: salvar as saídas caras (matmuls, saída do kernel GDN e da atenção) e recomputar o barato (normas, ativações, gates), via `torch.utils.checkpoint.create_selective_checkpoint_contexts` (existe no torch 2.14.1, conferido);
- uma fração de camadas sob checkpoint.

O alvo é a menor recomputação que cabe em 24 GB a 8192×1 e 4096×2. Atenção: os ops do FLA são `autograd.Function` próprios, não ops `aten`, e a política tem que tratá-los. Também precisa funcionar junto com o compile por Block.

**B. Kernels fundidos do FLA**, que já é dependência:
- `FusedLinearCrossEntropyLoss` no lugar de `chunked_lm_loss`, na cabeça principal e na do MTP;
- `FusedRMSNormGated` para o `o_norm(o) ⊙ silu(g)` do GDN;
- RMSNorm fundida;
- SwiGLU fundido.

Contrato a manter: `ignore_index = -100` e a soma mais a contagem de tokens que a ponderação por token (R6) usa. O caminho atual em torch puro fica como fallback para CPU e para a suíte sem FLA.

**C. Menos sincronização host↔GPU.** Passar `cu_seqlens_cpu` ao FLA. A API aceita, conferido em `fla/ops/utils/index.py::prepare_chunk_indices`, e o loader já tem `cu_seqlens` na CPU. Também procurar outras sincronizações por camada, por exemplo na montagem do `BlockMask`. Meta: zero sincronizações dentro do forward/backward de um micro-batch.

**D. Pipeline de entrada.** Pinned memory e pré-carregamento do próximo micro-batch, para que a montagem e a cópia H2D se sobreponham ao compute.

**E. Ajuste de forma e compile.**
- `micro_batch` e `grad_accum` **por estágio** (minha decisão): relaxar a invariante atual de `seq_len × micro_batch` igual em todos os estágios. Continuam fixos os 524.288 tokens por passo e o `seq_len` de cada estágio (4K e depois 8K, definidos na spec-mãe). O N do micro-batch pode mudar na troca de estágio, ao custo de uma recompilação nesse momento. O `unit_len` do loader (R22) segue igual ao maior `seq_len`;
- opções do Inductor para os Blocks;
- correção da pendência da re-revisão: o primeiro Block de atenção compila duas vezes na partida (`sore/model/attention.py:48-57`).

**F′. Kernels próprios** (minha decisão): permitidos, mas só onde o profiler mostrar que o op passa de ~5% do tempo do passo e com teste de paridade. Valem três formas:
- kernels Triton no repositório;
- substituição local de trechos do FLA 0.5.2, que é MIT: copiados com atribuição, sem atualizar o pacote. Exemplo: GVA sem materializar o `repeat_interleave`, ou configs de autotune para sm89;
- outros módulos do FLA além dos da alavanca B, como o `causal_conv1d` varlen.

**F. CUDA graphs: só investigar.** No varlen do FLA, o número de chunks depende dos documentos do lote (`prepare_chunk_indices`), então o grid muda a cada lote. Entregar um parecer de viabilidade. Implementar só se for barato e exato.

### Com perda (só na 4090, sob a §3)

**G. FP8 via torchao.** O usuário aprovou a dependência. O `uv` resolve `torchao==0.18.0` no ambiente atual; a compatibilidade com o torch 2.14.1 é confirmada na instalação. Escopo:
- Entra nos Linear grandes: FFN e projeções do GDN e da atenção.
- Fica de fora: embedding/LM head, normas e o kernel GDN.
- O tipo de escala é decidido por medição.

Em GPU sem FP8 (< sm89), ligar a opção **dá erro**, nunca ignora em silêncio. Os testes de velocidade do FP8 ganham marcador próprio e são pulados nas 3050. A correção e o efeito na loss são testados nas 3050 pelo modo de emulação do torchao (`emulate`), se a 0.18 ainda o tiver. Se não tiver, o código fica isolado atrás da opção e é testado no dia 1.

Se o `torchao==0.18.0` não funcionar com o torch 2.14.1 (minha decisão), a ordem de fallback é:
1. um `torchao` mais antigo que seja compatível;
2. um Linear FP8 próprio sobre `torch._scaled_mm`, sem dependência nova e atrás da mesma opção;
3. se nenhum der certo, pular direto para a alavanca H.

**H. Só se ainda faltar:** estados do otimizador em bf16, para sobrar memória e recomputar menos. Fica listado como candidato, não como compromisso.

## 6. Medição

**Benchmark** (`scripts/bench.py` estendido):
- liga e desliga cada alavanca;
- descarta o aquecimento;
- reporta tok/s, MFU, pico alocado/reservado e a divisão do tempo do passo (via `torch.profiler`);
- grava JSON para comparar execuções.

**Na 3050** usa:
- o `s150m`;
- uma **fatia do `b0_8`**: poucas camadas com as larguras do `b0_8`, para que as formas dos kernels sejam as da 4090.

**MFU:** mantém a definição atual de `flops.py` contra o pico bf16 de 165 TFLOPS, também nas corridas FP8. Isso fica escrito no runbook.

**Dados das corridas de qualidade.** O shard sintético atual (progressões aritméticas) não serve: o modelo o aprende quase perfeitamente. O projeto gera uma **amostra real de texto PT**:
- tokenizada com o tokenizer do Qwen3.5 via `tokenizers`, dependência aprovada pelo usuário;
- vocab remapeado provisoriamente para caber em 65.536 (os 65.535 ids mais frequentes na amostra mais um id de "desconhecido");
- com a ordem de grandeza de algumas centenas de milhões de tokens.

Esse remapeamento é só para os testes. A poda de verdade, fechada sob merges, é do plano de dados. Quando o primeiro shard PT do plano de dados existir, os gates passam a usá-lo. A fonte e o formato da amostra ficam com quem constrói. Se a fonte exigir outra dependência (por exemplo `pyarrow`), perguntar.

**Máquinas** (decidido pelo usuário):
- O desktop com RTX 3050 8 GB, Ryzen 5 3600X e Linux nativo vira a **segunda máquina de trabalho**: roda as corridas longas de qualidade do `s150m` e as referências com duas seeds.
- O notebook fica com testes, benchmarks e desenvolvimento.
- O projeto entrega um script de setup do desktop e instruções no runbook: repo, `uv`, venv e driver com suporte a CUDA ≥ 13. Quem roda é o usuário. Disparar corridas por SSH a partir do notebook é opcional e só se ele quiser.

**Protocolo de dia 1 na 4090**, no runbook:
1. ambiente (`nvidia-smi` com CUDA ≥ 13) e suíte;
2. benchmark de base;
3. cada alavanca exata;
4. escolha da configuração;
5. se ficar abaixo de 15K tok/s, A/B do FP8;
6. números e config finais gravados no runbook e em `b0_8_pretrain.yaml`.

## 7. Restrições

- torch 2.14.1 (cu130), triton 3.8.0 e FLA 0.5.2 seguem travados. As dependências novas aprovadas são `torchao` (FP8) e `tokenizers` (amostra real), ambas travadas no `uv.lock`. Nenhuma outra sem perguntar.
- A suíte continua passando em CPU sem Triton e sem FLA, com os marcadores `cuda`/`fla` existentes e um novo para FP8 ou sm89.
- As novas opções são campos de execução: entram em `_EXECUTION_FIELDS` e podem mudar num `--resume`.
- Checkpoints do formato atual continuam retomáveis, ou falham com mensagem clara.
- Identificadores em inglês; comentários e docs em português; conventional commits.
- Execução local no WSL `Ubuntu-24.04`, como no plano do núcleo.

## 8. Fica a cargo de quem constrói

- A política exata do checkpoint seletivo e o que salvar.
- Nomes das opções.
- Formato da divisão de tempo do profiler.
- Tolerâncias exatas e o critério numérico de "dentro do ruído".
- Opções do Inductor.
- A forma da fatia do `b0_8`.
- A ordem das alavancas. Sugestão:
  1. a amostra real e o benchmark com divisão de tempo, porque todas as outras dependem deles;
  2. C e D, que são baratas;
  3. A;
  4. B;
  5. G preparado com emulação, por estar no caminho crítico;
  6. E;
  7. F′, onde o profiler apontar.

## 9. Riscos

- 50% de MFU em bf16 num híbrido com 18 camadas GDN pode não ser alcançável. O FP8 provavelmente vai ser necessário. Se, mesmo com FP8, o resultado ficar abaixo do piso de 11,8K tok/s, a decisão de prazo ou de tokens volta ao usuário.
- `torchao` 0.18 com o torch 2.14.1 só é confirmado na instalação.
- O FP8 só é testado no dia 1.
- Ganhos medidos na 3050 (Ampere, 16 SMs) podem não se repetir na 4090 (Ada). A fatia do `b0_8` e a divisão de tempo reduzem esse risco, mas não o eliminam.
