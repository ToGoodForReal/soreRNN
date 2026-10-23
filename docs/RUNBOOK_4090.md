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
| Checkpoint | `runs/b0_8_pretrain/last.pt` a cada 2,5e8 tokens (~477 passos), ~6–7 GB cada |

## 1. Setup

No checkout do repositório:

```bash
uv sync                      # torch 2.14.1 (cu130), triton 3.8.0, flash-linear-attention 0.5.2 (uv.lock)
nvidia-smi                   # driver ok, 24GB livres, nenhum outro processo na GPU
uv run python -c "import torch, fla; print(torch.__version__, torch.cuda.get_device_name())"
uv run pytest -q             # tudo passa; os testes @cuda/@fla têm de rodar, não ser pulados
```

Antes de lançar, edite `configs/b0_8_pretrain.yaml`:
- `data_paths` e `eos_id` reais (hoje são placeholders, ver seção 6). Os shards são `uint32` com o
  sidecar `.json` ao lado. O stream corta cada shard em unidades de 8192 tokens (o maior `seq_len`
  dos estágios) e sorteia as unidades por época. No estágio de 4K cada unidade vira 2 janelas
  seguidas. A troca para 8K continua a mesma época a partir da próxima unidade inteira, sem
  reamostrar o que já foi visto. Um shard com menos de 8193 tokens não entra, e até 8192 tokens do
  fim de cada shard ficam de fora. Prefira shards grandes.
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
  8K já usa `micro_batch` 1. Por isso o primeiro recurso é reduzir o `chunk_size` da loss (seção 6).
  Repita o portão depois de qualquer mudança.

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
uv run python -m sore.train --config configs/b0_8_pretrain.yaml 2>&1 | tee -a runs/b0_8_pretrain/train.log
# Ctrl-b d para destacar; `tmux attach -t pretrain` para voltar
```

O 1º passo é lento porque é nele que o `torch.compile` compila cada Block. A troca para 8K **não**
recompila: N = `seq_len × micro_batch` = 8192 nos dois estágios e o nº de documentos do micro-batch
já entra dinâmico. A única variante extra é a de um micro-batch com um documento só
(`cu_seqlens = [0, N]`, comum no estágio 8192 × 1), compilada uma vez quando aparece pela 1ª vez.
O `tok_s`/`mfu` desses passos é ruído.

Opcional, numa segunda janela do tmux: o trainer guarda só um `last.pt` (troca atômica). Uma
cópia periódica permite voltar atrás em caso de divergência. Cada cópia tem ~6–7 GB, então apague
as antigas:

```bash
while sleep 43200; do cp runs/b0_8_pretrain/last.pt "runs/b0_8_pretrain/bak_$(date +%m%d_%H%M).pt"; done
```

## 4. Retomada

Depois de queda, reboot ou parada manual, rode no tmux:

```bash
uv run python -m sore.train --config configs/b0_8_pretrain.yaml --resume runs/b0_8_pretrain/last.pt 2>&1 | tee -a runs/b0_8_pretrain/train.log
```

A retomada restaura modelo, otimizadores, RNG, cursor de dados, estágio e contadores, e continua
exatamente do último checkpoint. O que veio depois dele é refeito. Para uma parada planejada,
espere um `last.pt` recente (`ls -l --time-style=full-iso runs/b0_8_pretrain/last.pt`) e dê
Ctrl-C. Veja as limitações na seção 6.

## 5. O que monitorar em `metrics.jsonl`

Cada passo grava uma linha JSON em `runs/b0_8_pretrain/metrics.jsonl`:

| Campo | O que é | Esperado / alerta |
|---|---|---|
| `step`, `tokens` | passo (1-based) e tokens consumidos | crescem 1 e 524.288 por linha |
| `loss`, `lm_loss`, `mtp_loss` | `lm + 0,3·mtp`, médias ponderadas por token | caem. O `mtp_loss` fica acima do `lm_loss` (prevê t+2). Spike sustentado: investigar |
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
      f"{mean('tok_s'):,.0f} tok/s | MFU {mean('mfu'):.1%} | {r['mem_gb']:.1f} GB | "
      f"pulados {sum(x['skipped'] for x in recs.values())}")
EOF
```

Acompanhe também `tail -f runs/b0_8_pretrain/train.log` (uma linha a cada `log_every` passos e
toda vez que um passo é pulado) e `nvidia-smi dmon -s pucm` (potência, clocks e memória).

## 6. Problemas conhecidos

- **Workaround do Inductor (torch 2.14.1).** Com `mix_order_reduction` ligado, a fusão quebra no
  1º backward do SoreLM (`AssertionError "509 v.s. 510"`). O trainer (e o `bench.py`) desliga a
  fusão automaticamente em `compile_model` (`sore/train/trainer.py`). Remova quando houver fix
  upstream.
- **Mudar o lr no YAML não vale na retomada.** O `initial_lr` (e os demais hiperparâmetros dos
  param groups, como `weight_decay`) volta do checkpoint.
- **`metrics.jsonl` pode repetir passos depois de crash + retomada.** Os passos após o último
  checkpoint são refeitos e gravados de novo. Para cada `step`, vale o último registro.
- **Não há checkpoint no SIGTERM nem no Ctrl-C.** Perde-se até `ckpt_every_tokens` de trabalho:
  2,5e8 tokens no `b0_8`, ~477 passos, ~6 h a 12K tok/s.
- **Memória transitória da loss.** O `chunked_lm_loss` usa por padrão `chunk_size` 8192 × vocab
  65.536, o que dá ≈ 2–3 GB transitórios por chunk (LM e MTP). Se o estágio de 8K der OOM, baixe o
  padrão em `sore/model/loss.py` (ex.: 2048) e repita o portão.
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
| eager (comando base) | 1.240 | 4,8% | 4,24 / 5,20 GB | 53 s |
| `--compile` (Blocks + CE da loss) | 2.959–3.227 | 11,4–12,4% | 3,00 / 3,28 GB | ~65 s a frio, ~10 s com o cache do Inductor |

Para comparação, nas mesmas condições: o antigo `torch.compile(model)` fazia 2.335 tok/s com
3,00 GB, e só os Blocks compilados (CE da loss eager) faziam 2.384–2.703 tok/s com 4,24 GB. A GPU
de notebook esquenta e o throughput oscila ~10–15% entre execuções.

No modo eager, com `--grad-accum 1`, cada passo de 2048 tokens gasta ~0,33 s no forward, ~0,87 s
no backward (com recomputação), ~0,30 s no Muon e ~0,02 s no AdamW. Por isso o Muon pesa ~22% do
passo sem acúmulo.
