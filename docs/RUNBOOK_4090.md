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
