# Runbook: pré-treino do `b0_8` na RTX 4090

Operação do pré-treino de ~6–7 semanas do preset `b0_8` numa RTX 4090 24GB (Linux nativo), com
`configs/b0_8_pretrain.yaml`.

| Item | Valor |
|---|---|
| Modelo | `b0_8`: 781,2M parâmetros (697,4M sem embedding), vocab 65.536, MTP 1 camada |
| Tokens | 50B, 95.367 passos de 524.288 tokens (`grad_accum` 64) |
| Estágios | 4096 × micro_batch 2 até 35B tokens (~passo 66.758), depois 8192 × 1 |
| LR | 3e-3 no Muon e no AdamW. WSD: warmup de 1.000 passos, platô, decay nos últimos 20% (a partir do passo ~76.294) |
| FLOPs de treino por token | 5,48 GFLOP a 4K e 5,70 GFLOP a 8K (`sore/model/flops.py`) |
| Checkpoint | `runs/b0_8_pretrain/last.pt` a cada 5e7 tokens (~95 passos, ~70 min a 12K tok/s), ~6–7 GB |
| Snapshots | `snapshot_stepNNNNNNNN.pt` a cada 2,5e9 tokens (~4.768 passos, ~2,4 dias); ficam os 4 mais novos |

## 1. Setup

Requer Python 3.12 e driver NVIDIA com suporte a CUDA ≥ 13.0 (≥ 580 no Linux; os wheels `torch 2.14.1`
do `uv.lock` são `cu130`). No checkout do repositório:

```bash
uv sync                      # torch 2.14.1 (cu130), triton 3.8.0, flash-linear-attention 0.5.2 (uv.lock)
nvidia-smi                   # driver com CUDA >= 13.0, 24GB livres, nenhum outro processo na GPU
uv run python -c "import torch, fla; print(torch.__version__, torch.cuda.get_device_name())"
uv run pytest -q             # tudo passa; os testes @cuda/@fla têm de rodar, não ser pulados
```

Antes de lançar, edite `configs/b0_8_pretrain.yaml`:
- `data_paths` e `eos_id` reais (hoje são placeholders, ver seção 6). Os shards são `uint32` com o
  sidecar `.json` ao lado. O stream corta cada shard em unidades fixas de `unit_len` = 8192 tokens
  (o maior `seq_len` dos estágios; todo `seq_len` precisa dividir `unit_len`) e sorteia as unidades
  por época. No estágio de 4K cada unidade vira 2 janelas seguidas. A troca para 8K continua a mesma
  época a partir da próxima unidade inteira, sem reamostrar o que já foi visto. Um shard com menos
  de `unit_len + 1` (8193) tokens não entra, e até `unit_len` tokens do fim de cada shard ficam de
  fora. Prefira shards grandes.
- `peak_tflops` fica no padrão 165 (pico BF16 denso da 4090) e só serve para o `mfu` do log.
- `lr_muon` e `lr_adamw` (3e-3, padrão do `TrainConfig`) ainda não foram validados. Com o
  `match_rms_adamw` o lr do Muon tem a escala do AdamW, e o 0,02 da literatura vale para outra
  escala. A ablação no s150m (`configs/s150m_ablation.yaml`) deve varrer o lr antes do lançamento,
  por exemplo 1e-3 / 3e-3 / 6e-3.

## 2. Portão obrigatório (antes de lançar)

### 2.1 Memória

Rode o benchmark nos dois formatos de estágio, com `--compile --grad-checkpoint` como no treino:

```bash
uv run python scripts/bench.py --preset b0_8 --seq-len 4096 --micro-batch 2 --grad-accum 2 --steps 6 --compile --grad-checkpoint --json | tee bench_b0_8_4k.log
uv run python scripts/bench.py --preset b0_8 --seq-len 8192 --micro-batch 1 --grad-accum 2 --steps 6 --compile --grad-checkpoint --json | tee bench_b0_8_8k.log
```

- `--grad-accum 2` é necessário: com 1 os gradientes são liberados a cada passo e o pico fica
  subestimado. No treino o 2º micro-batch roda com os gradientes acumulados vivos. Na 3050 o pico
  do s150m sobe ~0,6 GB (os gradientes fp32) de `--grad-accum 1` para 2.
- **Critério:** sem OOM e `VRAM pico` (alocado) **≤ 22 GB** nos dois. O `reservado` também tem de
  deixar folga para os 24GB. GB aqui é 10⁹ bytes, igual ao `mem_gb` do `metrics.jsonl`.
- **Se não couber:** baixar `micro_batch` e subir `grad_accum`, mantendo os tokens por passo
  (524.288). O `TrainConfig` exige `seq_len × micro_batch` igual em todos os estágios e o estágio de
  8K já usa `micro_batch` 1. Por isso o primeiro recurso é reduzir `loss_chunk_size` em
  `model_overrides` (seção 6). Repita o portão depois de qualquer mudança.

### 2.2 Throughput e prazo

O passo do otimizador (Newton-Schulz do Muon) pesa muito com poucos micro-batches. Para estimar o
throughput real, rode com mais acúmulo:

```bash
uv run python scripts/bench.py --preset b0_8 --seq-len 4096 --micro-batch 2 --grad-accum 16 --steps 4 --warmup 1 --compile --grad-checkpoint
```

Prazo para 50B tokens e o MFU equivalente a 4K (5,48 GFLOP/token, pico 165 TFLOPs):

| tok/s | dias | semanas | MFU a 4K |
|---:|---:|---:|---:|
| 10.000 | 57,9 | 8,3 | 33% |
| 11.800 | 49,0 | 7,0 | 39% |
| 13.800 | 41,9 | 6,0 | 46% |
| 16.000 | 36,2 | 5,2 | 53% |

Para fechar em 6–7 semanas são necessários ~11,8K–13,8K tok/s, ou seja, MFU de ~39–46%. Se o
benchmark ficar bem abaixo disso, reavalie (`total_tokens`, otimizações) antes de lançar.

## 3. Lançamento

Use `tmux` para o treino sobreviver à queda da sessão SSH:

```bash
tmux new -s pretrain
mkdir -p runs/b0_8_pretrain
uv run python -m sore.train --config configs/b0_8_pretrain.yaml 2>&1 | tee -ia runs/b0_8_pretrain/train.log
# Ctrl-b d para destacar; `tmux attach -t pretrain` para voltar
```

O `-i` do `tee` é obrigatório. O Ctrl-C vai para o grupo inteiro (uv, python e tee). Sem `-i` o
`tee` morre, o próximo `print` do trainer dá `BrokenPipeError` e o processo cai sem salvar o
`last.pt` (testado).

O 1º passo é lento porque é nele que o `torch.compile` compila cada Block. A troca para 8K **não**
recompila: N = `seq_len × micro_batch` = 8192 nos dois estágios e o nº de documentos do micro-batch
já entra dinâmico. A única variante extra é a de um micro-batch com um documento só
(`cu_seqlens = [0, N]`, comum no estágio 8192 × 1), compilada uma vez quando aparece pela 1ª vez.
O `tok_s`/`mfu` desses passos é ruído.

Se `runs/b0_8_pretrain/last.pt` já existir, o comando acima recusa começar do zero
(`FileExistsError`) para não sobrescrever o treino: use `--resume` (seção 4) ou outro `out_dir`.

Checkpoints (`configs/b0_8_pretrain.yaml`):
- `ckpt_every_tokens: 5e7`: `last.pt` (troca atômica) a cada ~95 passos.
- `snapshot_every_tokens: 2.5e9` e `keep_snapshots: 4`: no mesmo passo do `last.pt` sai também
  `snapshot_stepNNNNNNNN.pt` (mesmo conteúdo), e só os 4 mais novos ficam no disco. Servem para
  voltar ~2,4–9,5 dias em caso de divergência. Reserve ~35 GB para `last.pt` + 4 snapshots (mais um
  `.tmp` durante a gravação). `snapshot_every_tokens: 0` desliga.

## 4. Retomada

Depois de queda, reboot ou parada manual, rode no tmux:

```bash
uv run python -m sore.train --config configs/b0_8_pretrain.yaml --resume runs/b0_8_pretrain/last.pt 2>&1 | tee -ia runs/b0_8_pretrain/train.log
```

A retomada restaura modelo, otimizadores, RNG, cursor de dados, estágio e contadores, e continua
exatamente do último checkpoint. O que veio depois dele é refeito. Para voltar a um snapshot, passe
o arquivo dele em `--resume`.

**Parada planejada:** dê Ctrl-C uma vez no tmux (ou `kill -TERM` no PID do `uv`, que repassa o
sinal ao Python). O passo em andamento termina (~45 s a 12K tok/s), o `last.pt` é salvo, o log
mostra `[train] SIGINT: parada no passo N com last.pt salvo` e o processo sai normalmente. Sob
`uv run`, um Ctrl-C chega duas vezes ao Python (o do terminal e o repassado pelo uv). Por isso
sinais repetidos em até 1 s contam como um só. Um 2º Ctrl-C/SIGTERM depois disso tem o efeito
padrão e interrompe na hora, sem salvar. Veja as limitações na seção 6.

## 5. O que monitorar em `metrics.jsonl`

Cada passo grava uma linha JSON em `runs/b0_8_pretrain/metrics.jsonl`:

| Campo | O que é | Esperado / alerta |
|---|---|---|
| `step`, `tokens` | passo (1-based) e tokens consumidos | crescem 1 e 524.288 por linha |
| `loss`, `lm_loss`, `mtp_loss` | `lm + λ·mtp`, médias ponderadas por token | caem. O `mtp_loss` fica acima do `lm_loss` (prevê t+2). O `loss` cai também porque o λ cai |
| `mtp_weight` | λ do MTP no passo: linear de `mtp_weight_start` (padrão: o `mtp_weight` do modelo, 0,3) até `mtp_weight_end` (0,1) no último passo | 0,3 → 0,1 |
| `spike` | depois do warmup, `loss > (1 + spike_threshold) × EMA` do loss (beta 0,98; `spike_threshold` 0,3). O passo com spike não entra na EMA | raro. Gera `[spike] passo N: ...` no `train.log`. Vários seguidos: investigar e, se for o caso, voltar a um snapshot |
| `grad_norm` | norma antes do clip (1,0) | estável depois do warmup. Subida contínua ou picos frequentes ≫ 1 indicam instabilidade |
| `skipped` | passo pulado por grad norm não finito (`loss`/`grad_norm` viram `null`) | raro. Vários seguidos: parar e investigar |
| `lr_muon` | lr do Muon (`initial_lr × wsd`), antes da escala `0,2·√max(A, B)` do `match_rms_adamw` | sobe por 1.000 passos, platô em 3e-3, decai a partir do passo ~76.294 |
| `tok_s`, `mfu` | throughput do passo e MFU (`flops_per_token × tok_s / pico`) | estáveis. Queda indica throttling térmico ou outro processo na GPU (`nvidia-smi`) |
| `mem_gb` | `max_memory_allocated` desde o início do processo (só sobe) | ≤ 22. Sobe na troca para 8K |
| `seq_len` | contexto do estágio | 4096 até 35B tokens, depois 8192 |

Resumo dos últimos 50 passos, mantendo o último registro de cada passo:

```bash
uv run python - <<'EOF'
import json
recs = {}
for line in open("runs/b0_8_pretrain/metrics.jsonl"):
    r = json.loads(line)
    recs[r["step"]] = r  # retomadas repetem passos: vale o último
last = [recs[s] for s in sorted(recs)][-50:]
def mean(k):
    xs = [r[k] for r in last if r[k] is not None]
    return sum(xs) / max(len(xs), 1)
r = last[-1]
print(f"passo {r['step']} | {r['tokens'] / 1e9:.2f}B tokens | seq {r['seq_len']} | loss {mean('loss'):.4f} "
      f"(mtp {mean('mtp_loss'):.4f}) | gnorm {mean('grad_norm'):.3f} | lr {r['lr_muon']:.2e} | "
      f"{mean('tok_s'):,.0f} tok/s | MFU {mean('mfu'):.1%} | {r['mem_gb']:.1f} GB | λ {r['mtp_weight']:.3f} | "
      f"pulados {sum(x['skipped'] for x in recs.values())} | spikes {sum(x['spike'] for x in recs.values())}")
EOF
```

Acompanhe também `tail -f runs/b0_8_pretrain/train.log` (uma linha a cada `log_every` passos, toda
vez que um passo é pulado e a cada `[spike]`) e `nvidia-smi dmon -s pucm` (potência, clocks e
memória). O `mtp_weight` do `ModelConfig` pode mudar na retomada (campo de execução), mas no
treino quem manda é o λ do `TrainConfig`.

## 6. Problemas conhecidos

- **Workaround do Inductor (torch 2.14.1).** Com `mix_order_reduction` ligado, a fusão quebra no
  1º backward do SoreLM (`AssertionError "509 v.s. 510"`). O trainer (e o `bench.py`) desliga a
  fusão automaticamente em `compile_model` (`sore/train/trainer.py`). Remova quando houver fix
  upstream.
- **Flex com T < 128 usa o sdpa.** O flex compilado falha no Inductor com T=127 ("Invalid
  FlexAttention decode kernel options"). A `GatedAttention` cai no sdpa denso abaixo de
  `FLEX_MIN_LEN` = 128, o que não afeta o treino (N ≥ 4096). O BlockMask do flex é montado uma vez
  por forward no `SoreLM` e compartilhado pelas camadas de atenção e pelo MTP (a T=8192, cada
  montagem custa ~21 ms e ~670 MB transitórios).
- **Na retomada, `lr_muon`, `lr_adamw` e `weight_decay` vêm do YAML.** O trainer reescreve o
  `initial_lr` e o `weight_decay` de cada param group a partir do config e imprime
  `[resume] aviso: ...` para cada valor que mudou em relação ao checkpoint. Os demais
  hiperparâmetros dos param groups (betas, eps, momentum do Muon) voltam do checkpoint.
- **`metrics.jsonl` pode repetir passos depois de crash + retomada.** Os passos após o último
  checkpoint são refeitos e gravados de novo. Para cada `step`, vale o último registro.
- **Crash sem sinal perde até `ckpt_every_tokens` de trabalho.** Queda de energia, OOM ou
  `kill -9` não passam pelo handler de SIGINT/SIGTERM. No `b0_8` isso é até 5e7 tokens, ~95
  passos, ~70 min a 12K tok/s.
- **Memória transitória da loss.** O `chunked_lm_loss` usa `loss_chunk_size` (padrão 8192 no
  `ModelConfig`) × vocab 65.536, o que dá ≈ 2–3 GB transitórios por chunk (LM e MTP) em modo eager
  (com `--compile`, o `ChunkCrossEntropy` funde o chunk). Se o estágio de 8K der OOM, baixe
  `loss_chunk_size` em `model_overrides` do YAML (ex.: `loss_chunk_size: 2048`; é campo de execução,
  aceito também em `--resume`) e repita o portão.
- **`eos_id` e `data_paths` dos configs são placeholders** até o plano de tokenizer e dados.
- **Compile por Block e graph breaks.** O `chunk_gated_delta_rule` do FLA é
  `torch.compiler.disable`. O custo real vinha do grad checkpoint: o `torch.utils.checkpoint`
  desliga o Dynamo, então com `torch.compile(model)` o tronco inteiro rodava eager. Por isso o
  `compile_model` compila cada Block in-place (tronco e MTP) e deixa o checkpoint por fora. Sobra
  um graph break inofensivo por bloco GDN, em volta do kernel do FLA. A CE de cada chunk da loss
  (`SoreLM.chunk_ce`) também é compilada: eager, ela materializa logits fp32 e log-probs
  `chunk × vocab` no backward, o que custava +1,2 GB no s150m a 2048. Embedding e normas fora dos
  Blocks rodam eager. O `int((labels != -100).sum())` de `chunked_lm_loss` continua fazendo uma
  sincronização GPU→CPU por loss (LM e MTP) a cada forward.
- **N estático no compile.** O `compile_model` usa `dynamic=False`, porque o lowering do flex no
  Inductor falha com Q_LEN dinâmico (`CantSplit`). Só o nº de documentos (`len(cu_seqlens)`) é
  dinâmico (`SoreLM.trunk`). N muda só entre processos, nunca dentro de um treino: o `TrainConfig`
  exige `seq_len × micro_batch` igual em todos os estágios.

## 7. Referência: benchmark na RTX 3050 Laptop 6GB (WSL2)

Configuração: `s150m` (164,3M parâmetros, 1,35 GFLOP/token a 2048), FLA + flex, seq 2048 ×
micro_batch 1, `--grad-accum 2`, `--grad-checkpoint`, `--peak-tflops 35`, 10 passos com 2 de
warmup e mediana dos outros 8. O comando base é
`uv run python scripts/bench.py --preset s150m --seq-len 2048 --micro-batch 1 --grad-accum 2 --steps 10 --grad-checkpoint --peak-tflops 35`.

| Variante | tok/s | MFU | VRAM pico (alocado / reservado) | 1º passo |
|---|---:|---:|---:|---:|
| eager (comando base) | 1.854 | 7,1% | 4,24 / 5,21 GB | 44 s |
| `--compile` (Blocks + CE da loss) | 2.767–3.227 | 10,6–12,4% | 3,00 / 3,26–3,34 GB | ~65–85 s a frio, ~9 s com o cache do Inductor |

Para comparação, nas mesmas condições: antes de montar o `BlockMask` do flex uma vez por forward
(R24), o modo eager fazia 1.240 tok/s (4,8% MFU, 1º passo em 53 s). No modo compilado, o antigo
`torch.compile(model)` fazia 2.335 tok/s com 3,00 GB, e só os Blocks compilados (CE da loss eager)
faziam 2.384–2.703 tok/s com 4,24 GB. A GPU de notebook esquenta e o throughput oscila ~10–15% entre
execuções.

No modo eager, com `--grad-accum 1`, cada passo de 2048 tokens gasta ~0,33 s no forward, ~0,87 s
no backward (com recomputação), ~0,30 s no Muon e ~0,02 s no AdamW. Por isso o Muon pesa ~20% do
passo sem acúmulo.

### Linha de base do plano de throughput (bench v2, 2026-10-24)

Ponto de partida das tasks do plano de throughput (`docs/superpowers/plans/2026-10-24-perf-mfu.md`),
medido na 3050 com o `scripts/bench.py` v2: comprimentos de documento sorteados de
`data/sample_pt/doc_lengths.json` (a amostra PT; o nº de documentos varia entre micro-batches), seq
2048 × micro_batch 1, `--grad-accum 2 --grad-checkpoint --compile`, 10 passos com 2 de warmup
(mediana dos 8 restantes), `--peak-tflops 35`. O comando, trocando `<preset>`:

```bash
uv run python scripts/bench.py --preset <preset> --seq-len 2048 --micro-batch 1 --grad-accum 2 --steps 10 --warmup 2 \
    --grad-checkpoint --compile --doc-lengths data/sample_pt/doc_lengths.json --profile --peak-tflops 35 \
    --out runs/perf-mfu/baseline_3050.jsonl
```

O `b0_8` aqui é uma **fatia**: `--preset b0_8 --override n_layers=4` (221,4M parâmetros, 1,87
GFLOP/token a 2048), porque o modelo inteiro não cabe em 6 GB. Serve para comparar alavancas entre
si na 3050, não para prever o tok/s da 4090. O tempo do passo (e o tok/s) do bench v2 inclui a fase
`data` (montagem do batch + cópia H2D), como o passo do trainer; as fases saem de eventos CUDA.

| Configuração | tok/s | MFU | VRAM pico (alocado / reservado) | data | fwd_bwd | clip | optimizer |
|---|---:|---:|---:|---:|---:|---:|---:|
| `s150m` (164,3M) | 3.036 | 11,7% | 2,99 / 3,34 GB | 0,006 s | 1,055 s | 0,019 s | 0,282 s |
| fatia do `b0_8` (`n_layers=4`) | 2.616 | 14,0% | 4,06 / 4,64 GB | 0,006 s | 1,136 s | 0,024 s | 0,431 s |

Fases em segundos por passo (2 × 2048 tokens), mediana dos passos medidos. O MFU do `s150m` oscila
~10% entre execuções (a GPU de notebook esquenta; as execuções de calibração deram 2.970–3.097 tok/s).

Tempo de GPU por categoria de kernel (`--profile`: um passo extra, perfilado, com os mesmos batches de
um passo de aquecimento; `idle` é a parede do passo menos a união dos intervalos de kernels). O
profiler custa ~15–20% de parede (1,53 s e 1,90 s contra 1,35 s e 1,57 s sem ele), então vale a
proporção entre categorias, não o valor absoluto:

| Categoria | `s150m` | fatia do `b0_8` |
|---|---:|---:|
| matmul (cuBLAS/CUTLASS, mm do Inductor) | 0,876 s (57%) | 1,284 s (68%) |
| elementwise (Inductor `triton_poi/red/per`, ATen pontual/redução, Muon/AdamW) | 0,413 s (27%) | 0,453 s (24%) |
| gdn (kernels do FLA) | 0,125 s (8%) | 0,069 s (4%) |
| loss (CE: `log_softmax`/`nll_loss`) | 0,044 s (3%) | 0,043 s (2%) |
| attention (flex) | 0,026 s (2%) | 0,023 s (1%) |
| other (memcpy/memset, scan/sort do `cu_seqlens`) | 0,003 s (0%) | 0,004 s (0%) |
| idle (GPU parada) | 0,047 s (3%) | 0,025 s (1%) |
| parede do passo perfilado | 1,534 s | 1,901 s |

Leituras: na 3050 a GPU não fica esperando a CPU (`idle` ≤ 3%), então o ganho está em kernels, não em
lançamento; o matmul domina (57–68%), e o `optimizer` (Muon) pesa 21–28% do passo com `--grad-accum 2`
(no treino real ele se dilui em 16–64 micro-batches). As categorias vêm de `sore/train/profiling.py`
(tabela `_RULES`, montada com nomes reais de kernel capturados nesta máquina; estenda-a ali).
Uma janela com **um só** documento (`cu_seqlens` com 2 entradas) faz o Dynamo recompilar os Blocks
(especialização do tamanho 2, guarda `3 <= cu_seqlens.size()[0]`) na primeira vez que aparece; por
isso o bench perfila um passo com batches já executados.

## 8. Amostra PT para testes

Os portões de qualidade (A/B de perda em ~20M tokens) precisam de texto de verdade: o shard
sintético de `make_synthetic_shard.py` é aprendido quase por completo. A amostra tem 3e8 tokens de
mC4 português (`allenai/c4`, `multilingual/c4-pt*.json.gz`, ODC-BY), tokenizados com o tokenizer do
Qwen3.5 e remapeados **provisoriamente** para 65.536 ids: EOS = 0, UNK = 1 e os 65.534 ids mais
frequentes da amostra viram 2, 3, ... Não é o vocabulário definitivo do projeto.

```bash
uv run python scripts/make_text_sample.py --out data/sample_pt --n-tokens 3e8
```

Precisa de rede (baixa 3 arquivos de ~200 MB em `data/raw/`; leva ~4 min no WSL com 16 núcleos).
Saída em `data/sample_pt/` (ignorada pelo git): `sample_000.bin`(+`.json`), `remap.npy`,
`doc_lengths.json` (até 100.000 comprimentos de documento, com EOS, por reservoir sampling) e
`meta.json`. Use `eos_id=0` no `PackedTokenStream` e `vocab_size=65536` no modelo. O
`meta.json` da geração real (2026-10-24):

```json
{
  "source_urls": [
    ".../allenai/c4/resolve/1588ec454efa1a09f29cd18ddd04fe05fc8653a2/multilingual/c4-pt.tfrecord-00000-of-01024.json.gz",
    ".../allenai/c4/resolve/1588ec454efa1a09f29cd18ddd04fe05fc8653a2/multilingual/c4-pt.tfrecord-00001-of-01024.json.gz",
    ".../allenai/c4/resolve/1588ec454efa1a09f29cd18ddd04fe05fc8653a2/multilingual/c4-pt.tfrecord-00002-of-01024.json.gz"
  ],
  "tokenizer_repo": "Qwen/Qwen3.5-0.8B",
  "tokenizer_revision": "2fc06364715b967f1860aea9cf38778875588b17",
  "eos_old_id": 248044,
  "n_tokens": 300000000,
  "n_docs": 387560,
  "unk_rate": 0.00193377
}
```

O shard termina em EOS (o último documento é truncado para fechar em `n_tokens` exatos). Cerca de
0,19% dos tokens caem em UNK.

## 9. Segunda máquina (desktop)

Os portões de qualidade (seção 10) rodam num desktop Linux com uma RTX 3050 8GB, enquanto a 4090
fica livre. O desktop só treina; não precisa de rede para os testes, só para o `uv sync` inicial.

**1. Levar o código.** Preferido: `git bundle`, que não exige push nem credenciais.

```bash
# na máquina principal (na raiz do repo)
git bundle create sorernn.bundle feat/perf-mfu
# copie para o desktop (scp, pendrive, ...); no desktop:
git clone -b feat/perf-mfu sorernn.bundle soreRNN
# atualizações: na principal, só os commits novos desde o último sha que o desktop tem
git bundle create update.bundle <sha-antigo>..feat/perf-mfu
# no desktop (dentro do repo): git pull ../update.bundle feat/perf-mfu
```

A alternativa é um `git push` da branch para um remoto que o desktop alcance; o push só acontece se
**o usuário** o autorizar (o plano e os agentes nunca empurram sozinhos).

**2. Levar a amostra PT** (seção 8; `data/` não vai para o git). Duas opções:
- Copiar: `rsync -avP data/sample_pt/ desktop:soreRNN/data/sample_pt/`. O treino só lê
  `sample_000.bin` e o sidecar `sample_000.json`; o resto (`remap.npy`, `doc_lengths.json`, `meta.json`)
  é documentação. É a opção segura: os bytes são idênticos.
- Regenerar no desktop (precisa de rede, ~4 min):
  `uv run python scripts/make_text_sample.py --out data/sample_pt --n-tokens 3e8 --revision 2fc06364715b967f1860aea9cf38778875588b17`.
  O `--revision` fixa o tokenizer (o `tokenizer_revision` do `meta.json`). Os dados são os 3 primeiros
  arquivos `c4-pt.tfrecord-0000{0,1,2}-of-01024.json.gz` do `allenai/c4` na revisão
  `1588ec454efa1a09f29cd18ddd04fe05fc8653a2` (as `source_urls` do `meta.json`), mas o script resolve o
  `main` do dataset no momento da execução. Se o `main` andou, confira o resultado:
  `sha256sum data/sample_pt/sample_000.bin` tem de bater com o da máquina principal. Se não bater, copie.

**3. Preparar o ambiente.**

```bash
bash scripts/setup_worker.sh --check   # só valida: driver com CUDA >= 13.0, uv, amostra, ambiente sincronizado
bash scripts/setup_worker.sh           # instala o uv se faltar, `uv sync --frozen` e `pytest -m "cuda or fla"`
```

O `--check` não instala nada e sai com 1 e uma mensagem se algo faltar (inclusive se
`torch.cuda.is_available()` for falso ou o `fla` não importar). Sem `--check`, o script
instala o `uv` com o instalador oficial (`curl -LsSf https://astral.sh/uv/install.sh | sh`) e roda os
testes `@cuda`/`@fla`, que têm de executar: o pytest sai com 0 quando tudo é pulado, então o script
falha se houver qualquer teste pulado ou nenhum executado. Aceita o formato novo do `nvidia-smi`
("CUDA UMD Version: 13.4") e o antigo ("CUDA Version: 13.4").

## 10. Campanha de qualidade

Toda alavanca de velocidade **exata** (que não deveria mudar a matemática) precisa de uma corrida
curta no `s150m` (`configs/s150m_quality.yaml`: 2e7 tokens de texto PT real, 305 passos de 65.536
tokens, `compile: true`) comparada com duas referências. Uma alavanca **com perda** só é aceita se a
configuração exata ficar abaixo de 15.000 tok/s.

| Regra | Passa se |
|---|---|
| Exata | `abs(gap) <= max(ruído, 0,5%)`, com `gap = (cand - ref) / ref` e `ruído = abs(ref_s1 - ref_s0) / ref_s0` |
| Com perda | `gap <= 1%` (só para cima: perda menor sempre passa) |

A perda é a média do `lm_loss` nos últimos 10% dos passos (`ceil(0,1 × n)` valores não nulos do
`metrics.jsonl`; passos pulados não contam).

```bash
tmux new -s quality
bash scripts/run_quality_gates.sh              # todas as corridas de RUNS, depois a tabela
bash scripts/run_quality_gates.sh --only ref_s0
```

- Cada corrida grava em `runs/quality/NOME`. Uma corrida que já terminou (`last.pt` e o último passo
  no `metrics.jsonl`) é pulada; uma interrompida é retomada de `last.pt`. Basta rodar o mesmo comando
  de novo depois de uma queda.
- `ref_s0` (seed 0) e `ref_s1` (seed 1) rodam com todas as alavancas da lista **desligadas**
  (`ALL_OFF` no topo do script). A comparação final só sai com as duas referências terminadas; o
  `compare_runs.py` imprime uma tabela markdown (perda final, gap, limiar, PASSA/REPROVA) e sai com 1
  se alguma candidata reprova.
- Na 3050 de 8GB o `micro_batch` 2 a 2048 deve caber. Se estourar a memória, use
  `EXTRA_SET="model_overrides.grad_checkpoint=true"` em **todas** as corridas, refs incluídas
  (apague `runs/quality` antes, para não misturar configurações).

**Como uma task de alavanca entra.** No `scripts/run_quality_gates.sh`: (1) acrescente a flag
"desligada" da alavanca a `ALL_OFF` (um `KEY=VALUE` por elemento, vale para todas as corridas);
(2) acrescente a linha `NOME:override` ligada em `RUNS`, por exemplo
`"fused_ce:model_overrides.fused_ce=true"`. As overrides de uma linha ficam depois de `ALL_OFF` e
vencem. Várias overrides numa linha ou no `EXTRA_SET` se separam por `;`. Alavanca com perda: o
nome também vai em `LOSSY`. Mude `ALL_OFF` e as referências já rodadas passam a ser de outra
configuração: apague `runs/quality/ref_s*` e rode de novo.

**Ensaio curto (valida o encanamento, não a qualidade; na 3050 do notebook leva ~5 min, quase tudo é o `torch.compile`).**

```bash
EXTRA_SET="total_tokens=2e5;warmup_steps=1;stages=[{until_tokens: 2e5, seq_len: 2048, micro_batch: 2}]" \
  bash scripts/run_quality_gates.sh --only ref_s0
```

`EXTRA_SET` vale para todas as corridas, separado por `;` (os valores têm espaços e vírgulas). O
`warmup_steps=1` é necessário: com 3 passos o warmup de 30 passos passaria do início do decay. Com
`--set` direto: `python -m sore.train --config configs/s150m_quality.yaml --set seed=1 --set model_overrides.fused_ce=true`;
`model_overrides.X` faz merge com o `model_overrides` do YAML, e chave desconhecida continua dando erro.
Apague `runs/quality/` depois do ensaio: a corrida curta deixa um `last.pt` que a campanha real
retomaria como se fosse dela (o passo final difere, então não é pulada, mas é continuada).

## 11. Treino FP8 via torchao (alavanca G)

O FP8 é uma **alavanca com perda** (spec §3): fica desligado por padrão (`fp8: "off"`) e só entra
se a melhor configuração exata na RTX 4090 ficar abaixo de 15.000 tok/s (meta de MFU ≥ 50%),
com o gate de loss final ≤ 1% pior que a referência em bf16.

### 11.1 Configuração e receitas

No YAML de treino (`TrainConfig`):

```yaml
fp8: "rowwise"          # "off" (padrão), "tensorwise" ou "rowwise"
fp8_emulate: false      # true apenas para validação em GPUs < sm89 (ex: RTX 3050)
```

- `"tensorwise"`: escala dinâmica por tensor (cuBLAS). Opção mais rápida.
- `"rowwise"`: escala dinâmica por linha (CUTLASS) com e4m3 e potências de 2. Mais precisa.
- **Escopo do filtro:** converte apenas matrizes `nn.Linear` grandes sob `blocks.` e `mtp.`
  com `in_features` e `out_features` múltiplos de 16, fora de `res_mixer`/`res_ffn`. Projeções
  pequenas como `a_proj` e `b_proj` (saída `v_heads`), embedding e cabeça de LM ficam sempre em
  alta precisão.
- **Contrato de checkpoint:** `apply_fp8` preserva os nomes dos parâmetros e as chaves do
  `state_dict`. Checkpoints salvos com FP8 podem ser retomados com FP8 desligado e vice-versa.
- **Ordem de inicialização:** modelo → `apply_fp8` → `build_optimizers` → `compile_model`.

### 11.2 Requisitos de hardware e recusa explícita

- FP8 nativo exige arquitetura NVIDIA Ada Lovelace ou Hopper (capacidade CUDA ≥ 8.9, como a RTX 4090).
- Em hardware com capacidade inferior (ex: RTX 3050 sm86), pedir `fp8: "tensorwise"` ou `"rowwise"`
  sem `fp8_emulate: true` levanta imediatamente `RuntimeError` com mensagem explicativa contendo `"FP8"`.

### 11.3 Benchmark de FP8 na 4090

Para medir o impacto de FP8 no throughput e no consumo de memória:

```bash
# Comparação bf16 vs rowwise no estágio 4K
uv run python scripts/bench.py --preset b0_8 --seq-len 4096 --micro-batch 2 --grad-accum 16 --steps 6 --compile --grad-checkpoint
uv run python scripts/bench.py --preset b0_8 --seq-len 4096 --micro-batch 2 --grad-accum 16 --steps 6 --compile --grad-checkpoint --fp8 rowwise
```

## 12. Protocolo de Dia 1 na RTX 4090

Este protocolo estabelece o roteiro passo a passo para o primeiro dia de operação na RTX 4090 física,
desde a validação do ambiente até a escolha e consolidação da configuração final de pré-treino do `b0_8`.

### 12.1 Definição do MFU (incluindo corridas FP8)

O MFU (*Model Flops Utilization*) é calculado rigorosamente como:

$$\text{MFU} = \frac{\text{flops\_per\_token}(\text{cfg}, \text{seq\_len}) \times \text{tok/s}}{\text{pico\_flops}}$$

- Na RTX 4090, o pico considerado é **sempre o pico BF16 denso de 165 TFLOPS**, **inclusive nas corridas com FP8**.
- **Justificativa metodológica:** manter a base de 165 TFLOPS em todas as corridas (BF16 e FP8) garante uma métrica uniforme e diretamente proporcional ao throughput real. O ganho de velocidade proporcionado pelo tensor core FP8 se traduz em um MFU nominalmente maior (por exemplo, superando 50%), permitindo comparação direta de custo-benefício em relação à meta de 15.000 tok/s.

### 12.2 Passos de Operação do Dia 1

Execute os 7 passos numerados na ordem abaixo:

1. **Validação do hardware e driver:**
   ```bash
   nvidia-smi
   ```
   Confirme que o driver NVIDIA reporta suporte a CUDA ≥ 13.0 (driver ≥ 580 no Linux), que a GPU é a RTX 4090 (24 GB de VRAM) e que não há processos concorrentes consumindo memória ou computação.

2. **Sincronização do ambiente:**
   ```bash
   uv sync
   ```
   Instala as versões travadas no `uv.lock`: `torch==2.14.1` (cu130), `triton==3.8.0`, `flash-linear-attention==0.5.2`, `torchao==0.18.0`.

3. **Verificação da suíte de testes:**
   ```bash
   uv run pytest -q
   ```
   Confirme que toda a suíte passa. Na 4090 (arquitetura Ada Lovelace, sm89), os testes com marcador `@fp8` (ex: `test_fp8_real_smoke`) **devem executar e passar**, sem nenhum skip por falta de hardware nativo.

4. **Execução da matriz de benchmark do Dia 1:**
   ```bash
   bash scripts/day1_bench.sh
   ```
   O script executa automaticamente a matriz no `b0_8` para os dois estágios de contexto (`8192x1` e `4096x2`), com `--grad-accum 8 --steps 6 --warmup 2 --doc-lengths data/sample_pt/doc_lengths.json`:
   - Linha de base (todas as alavancas desligadas, checkpoint `full`, com profiling);
   - Cada alavanca exata isolada (`fused_ce`, `gdn_path`, `ckpt_selective`, `ckpt_ffn`);
   - Todas as alavancas exatas juntas;
   - Todas as exatas $\times$ políticas de checkpoint (`full`, `selective`, `ffn`) $\times$ `checkpoint_frac` {1.0, 0.5, 0.25};
   - A melhor configuração exata com profiling e $\times$ FP8 (`tensorwise` e `rowwise`).
   Todos os resultados e eventuais OOMs são registrados em `runs/day1/bench.jsonl`.

5. **Sumarização e inspeção do throughput:**
   ```bash
   uv run python scripts/summarize_bench.py runs/day1/bench.jsonl
   ```
   Analise a tabela Markdown gerada:
   - Verifique quais configurações atingem a **meta de 15.000 tok/s** (MFU ≥ 50%);
   - Verifique que as configurações candidatas respeitam o **piso de 11.800 tok/s** e mantêm consumo de VRAM alocada $\le 22$ GB;
   - Identifique a melhor configuração puramente exata.

6. **Decisão sobre FP8 e validação de qualidade (se necessário):**
   - **Se a melhor configuração exata atingir $\ge 15.000$ tok/s:** O FP8 **não** é utilizado. O pré-treino segue exclusivamente com as otimizações exatas.
   - **Se a melhor configuração exata ficar $< 15.000$ tok/s:** O FP8 está autorizado a ser avaliado. Execute o portão de qualidade A/B de 3e8 tokens aplicando a melhor configuração exata medida em ambos os treinos via `--set`:
     ```bash
     # Exemplo: se a melhor configuração exata no benchmark usou selective checkpoint com frac 0.5
     uv run python -m sore.train --config configs/b0_8_fp8_ab_bf16.yaml \
         --set model_overrides='{checkpoint_policy: selective, checkpoint_frac: 0.5}'
     uv run python -m sore.train --config configs/b0_8_fp8_ab_fp8.yaml \
         --set model_overrides='{checkpoint_policy: selective, checkpoint_frac: 0.5}'
     ```
     *(Nota: chaves individuais também podem ser passadas pontualmente, ex: `--set model_overrides.checkpoint_policy=selective --set model_overrides.checkpoint_frac=0.5`; e caso a receita vencedora do FP8 no benchmark tenha sido `tensorwise`, sobrescreva com `--set fp8=tensorwise`)*.

     Em seguida, compare as perdas finais com `compare_runs.py` no modo lossy:
     ```bash
     uv run python scripts/compare_runs.py --ref runs/fp8_ab/bf16 --ref2 runs/fp8_ab/bf16 runs/fp8_ab/fp8 --lossy fp8
     ```
     **Critério de aprovação (spec §3):** $\text{loss final (FP8)} \le 1{,}01 \times \text{loss final (BF16)}$ (degradação máxima tolerada de 1%).

7. **Consolidação na configuração de pré-treino:**
   Grave os hiperparâmetros e overrides da configuração vencedora em `configs/b0_8_pretrain.yaml` (ajustando `model_overrides`, por exemplo `model_overrides: {grad_checkpoint: true, checkpoint_policy: selective, checkpoint_frac: 0.5}`, e `fp8` se aprovado no A/B) e anote os números medidos (tok/s, MFU, pico de VRAM) na documentação do runbook antes de disparar o treino final de 50B tokens.

### 12.3 Resultados e Decisões do Dia 1 (FP8 vs BF16)

Em 27 de Outubro de 2026, o protocolo foi executado na NVIDIA GeForce RTX 4090 24GB física (sm89, Driver 595.99.02, CUDA 13.2, PyTorch 2.14.1+cu130, TorchAO 0.18.0):

1. **Investigação do FP8 Tensorwise:**
   - Embora o benchmark inicial tenha reportado falsos ~17.691 tok/s, a corrida de validação A/B (3e8 tokens) revelou **falha crítica**: 100% dos passos resultaram em `gnorm: nan` e foram descartados (`skipped: true`). O otimizador nunca atualizou pesos (a perda permaneceu congelada em ~11.33).
   - Causa raiz: A escala dinâmica global de ativações no backward sob `torch.compile` e `grad_checkpoint` causa overflow/underflow de escala no `torchao 0.18`. O script `scripts/bench.py` foi atualizado para verificar finitude de gradiente e invalidar tais medições.

2. **Investigação do FP8 Rowwise:**
   - Estabilidade numérica comprovada (0 skips, loss decresceu normalmente), mas o throughput real foi de **~13.650 tok/s** (MFU 45,3%), tornando-se **~7,3% mais lento que o BF16**.
   - Causa raiz: Overhead das operações de redução linha a linha no `d_model=1280` superou o ganho computacional dos Tensor Cores FP8.

3. **Decisão Oficial:**
   - FP8 foi descartado para o pré-treino final de 50B tokens.
   - **Configuração Oficial Adotada (`configs/b0_8_pretrain.yaml`):**
     - `fp8: "off"` (BF16 puro).
     - `model_overrides`: `vocab_size: 98304`, `grad_checkpoint: true`, `checkpoint_policy: "ffn"`, `checkpoint_frac: 1.0`.
     - Throughput medido no treino A/B: **~14.720 tok/s** (MFU 48,9%), estimando a conclusão do pré-treino de 50B tokens em ~39,3 dias.
     - Pico de VRAM alocada: 22,37 GB (estável ao longo de 572 passos na RTX 4090, sem nenhum OOM).
     - Alternativa estrita $\le 22$ GB: `checkpoint_policy: "full"`, `checkpoint_frac: 0.5`, alavancas exatas ativadas (13.550 tok/s com 19,94 GB).



