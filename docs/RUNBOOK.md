# Runbook: pré-treino do `b0_4` na RTX 3050 8GB

Operação do pré-treino de ~2–3 meses do preset `b0_4` numa RTX 3050 8GB (desktop, Linux nativo), com
`configs/b0_4_pretrain.yaml`. O plano anterior (`b0_8`, 50B tokens na RTX 4090) está em
[`history/RUNBOOK_4090.md`](history/RUNBOOK_4090.md).

| Item | Valor |
|---|---|
| Modelo | `b0_4`: ~0,41B parâmetros (~0,31B sem embedding), d_model 1024, 16 camadas (12 GDN + 4 atenção), MTP 1 camada |
| Tokenizer | Qwen3.5 podado para 98.304 ids, com 25% de código na amostra de calibração (§2) |
| Tokens | 10B (~24 tokens/parâmetro), 38.146 passos de 262.144 tokens (`grad_accum` 64) |
| Estágios | 2048 × micro_batch 2 até 7B tokens (~passo 26.703), depois 4096 × 1 |
| LR | 3e-3 no Muon e no AdamW até a ablação (§5.3). WSD: warmup de 1.000 passos, platô, decay nos últimos 20% |
| FLOPs de treino por token | 3,16 GFLOP a 2K e 3,22 GFLOP a 4K (`sore/model/flops.py`); ~38% é a cabeça de LM (LM + MTP) |
| Memória fixa | ~5,3 GB: pesos, gradientes e estados do Muon/AdamW em fp32 (o autocast faz as contas em BF16) |
| Checkpoint | `runs/b0_4_pretrain/last.pt` a cada 1e7 tokens (~38 passos), ~3,7 GB |
| Snapshots | `snapshot_stepNNNNNNNN.pt` a cada 1e9 tokens (~3.815 passos); ficam os 4 mais novos (~15 GB) |
| Disco | ~40 GB de shards `uint32` + ~20 GB de checkpoints |

## 1. Setup

Requer Python 3.12, [uv](https://docs.astral.sh/uv/) e driver NVIDIA com suporte a CUDA ≥ 13.0 (≥ 580
no Linux; os wheels `torch 2.14.1` do `uv.lock` são `cu130`). No checkout do repositório:

```bash
bash scripts/setup_worker.sh --check   # só valida: driver, uv, ambiente sincronizado, torch com CUDA, fla
bash scripts/setup_worker.sh           # instala o uv se faltar, `uv sync --frozen` e roda os testes @cuda/@fla
uv run pytest -q                       # suíte inteira; os testes @fp8 são pulados (a 3050 é sm86, sem FP8)
```

Com 8 GB não sobra VRAM para o desktop gráfico. Rode a máquina sem sessão gráfica na NVIDIA (servidor,
`systemctl set-default multi-user.target` ou vídeo pela iGPU) e confira com `nvidia-smi` que nenhum outro
processo usa a GPU.

O código vai do notebook para o desktop por `git push`/`git pull` num remoto, ou por `git bundle`
(sem rede nem credenciais):

```bash
git bundle create sorernn.bundle reestruturacao-3050        # no notebook
git clone -b reestruturacao-3050 sorernn.bundle soreRNN     # no desktop
```

## 2. Tokenizer

O pré-treino usa o tokenizer do Qwen3.5 (BPE byte-level, 248K ids) podado para 98.304 ids
(`sore/train/tokenizer_prune.py`). O conjunto mantido é fechado sob os merges até os 256 tokens de byte,
então todo texto continua codificável, sem UNK. Os tokens adicionados (EOS, chat, tool, think, visão,
FIM) ficam todos. A poda é gulosa pela frequência numa amostra de ~2e8 tokens das fontes de §3. Na
amostra, o código vale 25% (`--code-frac`), contra 10% no treino. Assim a poda guarda mais tokens de
código (identificadores, indentação, operadores) do que a mistura do treino guardaria sozinha.

Primeiro, a amostra PT (§9). Além de servir aos portões, ela baixa o C4-pt em `data/raw/`, o texto PT
fora da amostra que o portão de fertilidade mede:

```bash
uv run python scripts/make_text_sample.py --out data/sample_pt --n-tokens 3e8
uv run python scripts/prune_tokenizer.py --out data/tokenizer        # --vocab-size 98304 --code-frac 0.25
```

O script imprime a fertilidade (tokens por palavra) do vocabulário completo e do podado em PT, EN e
código, e o round-trip `decode(encode(x)) == NFC(x)`. Ele **reprova** (sai com erro) se:
- PT ficar mais de 3% pior que o vocabulário completo (`--max-pt-fertility-loss`, critério da spec). A
  correção é baixar `--code-frac` ou subir `--vocab-size`;
- código ficar mais de 3% pior (`--max-code-fertility-loss`). A correção é subir `--code-frac` ou
  `--vocab-size`;
- houver qualquer falha de round-trip.

A saída é `data/tokenizer/{tokenizer.json, old_to_new.npy, counts.npy, meta.json}`. O `old_to_new`
(-1 para os podados) restringe os logits do teacher Qwen3.5 ao subvocabulário, para destilação.

**Confira o `eos_id`:** ele está em `data/tokenizer/meta.json`. Com o Qwen3.5 atual (26 tokens
adicionados, `<|endoftext|>` o primeiro deles) ele sai 98.278, que é o valor de
`configs/b0_4_pretrain.yaml` e `configs/s150m_ablation.yaml`. Se sair outro valor (outra revisão do
tokenizer, ou uma amostra com menos tokens distintos que o orçamento), corrija os dois configs. O
`scripts/launch_pretrain.sh` recusa começar se o config e os dados discordarem.

## 3. Dados

```bash
uv run python scripts/make_pretrain_data.py --tokenizer-dir data/tokenizer --out data/pretrain --total-tokens 10e9
```

As fontes e a mistura estão em `sore/train/sources.py` (`MIXTURE`): 65% FineWeb-2 PT, 10% PT de alta
qualidade (FineWeb2-HQ, Wikipedia), 15% EN educacional e matemática (FineWeb-Edu, FineMath) e 10% código
(github-code-clean só com licenças permissivas, OPC annealing). Tudo é lido em streaming do Hub, sem
cópia local dos parquets. O FineWeb2-HQ é gerado primeiro e o FineWeb-2 pula os mesmos ids de documento
(o HQ é um subconjunto dele).

- **Retomável:** cada arquivo de entrada vira um shard `data/pretrain/<fonte>/<fonte>_NNNNN.bin` (+ sidecar
  `.json`), gravado em `.tmp` e renomeado no fim. Basta rodar o mesmo comando de novo depois de uma queda.
- **Revisões fixas:** a 1ª execução grava as revisões dos datasets e a lista de arquivos em
  `data/pretrain/manifest.json`; as retomadas leem os mesmos arquivos.
- **Espaço:** 10B tokens `uint32` = 40 GB. O script recusa começar sem espaço livre para o que falta.
- **Tempo:** depende da rede e da CPU (a tokenização é em Rust, em lotes). Conte de horas a 1–2 dias.

## 4. Portões antes de lançar

### 4.1 Memória e throughput (benchmark)

```bash
uv run python scripts/bench.py --preset b0_4 --seq-len 2048 --micro-batch 2 --grad-accum 2 --steps 6 --compile --grad-checkpoint --json | tee bench_b0_4_2k.log
uv run python scripts/bench.py --preset b0_4 --seq-len 4096 --micro-batch 1 --grad-accum 2 --steps 6 --compile --grad-checkpoint --json | tee bench_b0_4_4k.log
```

- `--grad-accum 2` é necessário: com 1 os gradientes são liberados a cada passo e o pico fica
  subestimado.
- **Critério:** sem OOM, `VRAM pico` (alocado) **≤ 7,0 GB** e reservado ≤ 7,4 GB nos dois estágios. O
  contexto CUDA ocupa ~0,4–0,5 GB fora disso. GB aqui é 10⁹ bytes, igual ao `mem_gb` do `metrics.jsonl`.
- **Se não couber:** o primeiro recurso é baixar `loss_chunk_size` em `model_overrides` (ex.: 2048; §8).
  O `TrainConfig` exige `seq_len × micro_batch` igual em todos os estágios, e o estágio de 4K já usa
  `micro_batch` 1. Repita o portão depois de qualquer mudança.

Para escolher as alavancas de velocidade, rode a matriz completa: cada alavanca exata sozinha, todas
juntas e as políticas de checkpoint × `checkpoint_frac`. Ela grava em `runs/bench_matrix/bench.jsonl` e
termina com a tabela do `summarize_bench.py`:

```bash
bash scripts/bench_matrix.sh                 # PRESET=b0_4, SEQS="4096x1 2048x2", GRAD_ACCUM=8
```

Use a melhor configuração exata que passa no critério de memória. Se for diferente do
`model_overrides` de `configs/b0_4_pretrain.yaml`, atualize o config e o teste
`test_b0_4_config_values`.

### 4.2 Prazo

Para 10B tokens e o MFU equivalente a 2K (3,16 GFLOP/token, pico BF16 denso de 18,2 TFLOPs da 3050
8GB), o prazo fica assim:

| tok/s | dias | meses | MFU a 2K |
|---:|---:|---:|---:|
| 1.200 | 96 | 3,2 | 21% |
| 1.500 | 77 | 2,5 | 26% |
| 1.800 | 64 | 2,1 | 31% |
| 2.100 | 55 | 1,8 | 36% |

A estimativa de partida é ~1.500–2.000 tok/s. Ela vem dos ~4–5 TFLOP/s efetivos medidos na 3050 do
notebook (§10), escalados pelos 25% a mais de SMs e pelo clock maior do desktop. O número real sai do
benchmark (rode com `--grad-accum 16` para diluir o passo do Muon) e da validação.

### 4.3 Validação curta (treino real)

```bash
uv run python -m sore.train --config configs/b0_4_validate.yaml 2>&1 | tee runs/b0_4_validate.log
```

20 passos em cada estágio (2048×2 e 4096×1) sobre a amostra PT, ~2 h. Critério: nenhum passo pulado,
`loss` caindo, `mem_gb` ≤ 7,0 nos dois estágios e o `tok_s` dos passos estáveis (ignore o 1º passo, que
compila) perto do benchmark. Apague `runs/b0_4_validate/` depois.

### 4.4 Ablação de lr (s150m)

`lr_muon` e `lr_adamw` (3e-3) nunca foram validados. Com o `match_rms_adamw` o lr do Muon tem a escala
do AdamW, e o 0,02 da literatura vale para outra escala. `configs/s150m_ablation.yaml` usa os mesmos
dados, tokenizer e batch (262.144 tokens por passo) do `b0_4`, com 5e8 tokens, ~1,5–2 dias por corrida:

```bash
for LR in 1e-3 3e-3 6e-3; do
  uv run python -m sore.train --config configs/s150m_ablation.yaml \
      --set lr_muon=$LR --set lr_adamw=$LR --set out_dir=runs/ablation/lr$LR
done
```

Compare o `lm_loss` final (média dos últimos 10% dos passos, como em §11) e grave o melhor lr em
`configs/b0_4_pretrain.yaml`. Se o melhor for uma das pontas (1e-3 ou 6e-3), estenda a varredura para
aquele lado antes de decidir.

## 5. Lançamento

O jeito recomendado é o `scripts/launch_pretrain.sh`. Ele termina a geração de dados (com novas
tentativas), confere se cada fonte encheu a cota e se o `eos_id`/vocab do config batem com o
`manifest.json` dos dados, e treina, retomando do `last.pt` a cada queda (até 20 vezes):

```bash
tmux new -s pretrain
bash scripts/launch_pretrain.sh        # logs em runs/b0_4_pretrain/{launch,train}.log
# Ctrl-b d para destacar; `tmux attach -t pretrain` para voltar
```

Para retomar sozinho depois de uma queda de energia, ponha no `crontab -e` do usuário:

```
@reboot PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin setsid nohup bash /caminho/do/soreRNN/scripts/launch_pretrain.sh
```

O lançador usa um lock: um `@reboot` com outra cópia rodando sai sem fazer nada. Os caminhos podem mudar
por variáveis de ambiente (`CONFIG`, `DATA_DIR`, `TOKENIZER_DIR`, `TOTAL_TOKENS`, `RUN_DIR`; ver o
cabeçalho do script). Eles precisam casar com `data_paths` e `out_dir` do config.

Para rodar só o treino, à mão:

```bash
uv run python -m sore.train --config configs/b0_4_pretrain.yaml 2>&1 | tee -ia runs/b0_4_pretrain/train.log
```

O `-i` do `tee` é obrigatório. O Ctrl-C vai para o grupo inteiro (uv, python e tee). Sem `-i` o `tee`
morre, o próximo `print` do trainer dá `BrokenPipeError` e o processo cai sem salvar o `last.pt`.

O 1º passo é lento porque é nele que o `torch.compile` compila cada Block. A troca para 4K **não**
recompila: N = `seq_len × micro_batch` = 4096 nos dois estágios e o nº de documentos do micro-batch já
entra dinâmico. A única variante extra é a de um micro-batch com um documento só (`cu_seqlens = [0, N]`),
compilada uma vez quando aparece pela 1ª vez. O `tok_s`/`mfu` desses passos é ruído.

Se `runs/b0_4_pretrain/last.pt` já existir, o comando manual recusa começar do zero
(`FileExistsError`) para não sobrescrever o treino: use `--resume` (§6) ou outro `out_dir`.

## 6. Retomada

```bash
uv run python -m sore.train --config configs/b0_4_pretrain.yaml --resume runs/b0_4_pretrain/last.pt 2>&1 | tee -ia runs/b0_4_pretrain/train.log
```

A retomada restaura modelo, otimizadores, RNG, cursor de dados, estágio e contadores, e continua
exatamente do último checkpoint. O que veio depois dele é refeito. Para voltar a um snapshot, passe o
arquivo dele em `--resume`.

**Parada planejada:** dê Ctrl-C uma vez (ou `kill -TERM` no PID do `uv`, que repassa o sinal ao
Python). O passo em andamento termina (~2–3 min a 1,5–2K tok/s), o `last.pt` é salvo, o log mostra
`[train] SIGINT: parada no passo N com last.pt salvo` e o processo sai normalmente (código 0; o
lançador então não retoma). Sob `uv run`, um Ctrl-C chega duas vezes ao Python, por isso sinais
repetidos em até 1 s contam como um só. Um 2º Ctrl-C/SIGTERM depois disso interrompe na hora, sem salvar.

## 7. O que monitorar em `metrics.jsonl`

Cada passo grava uma linha JSON em `runs/b0_4_pretrain/metrics.jsonl`:

| Campo | O que é | Esperado / alerta |
|---|---|---|
| `step`, `tokens` | passo (1-based) e tokens consumidos | crescem 1 e 262.144 por linha |
| `loss`, `lm_loss`, `mtp_loss` | `lm + λ·mtp`, médias ponderadas por token | caem. O `mtp_loss` fica acima do `lm_loss` (prevê t+2). O `loss` cai também porque o λ cai |
| `mtp_weight` | λ do MTP no passo: linear de `mtp_weight_start` (padrão: o `mtp_weight` do modelo, 0,3) até `mtp_weight_end` (0,1) no último passo | 0,3 → 0,1 |
| `spike` | depois do warmup, `loss > (1 + spike_threshold) × EMA` do loss (beta 0,98; `spike_threshold` 0,3). O passo com spike não entra na EMA | raro. Gera `[spike] passo N: ...` no `train.log`. Vários seguidos: investigar e, se for o caso, voltar a um snapshot |
| `grad_norm` | norma antes do clip (1,0) | estável depois do warmup. Subida contínua ou picos frequentes ≫ 1 indicam instabilidade |
| `skipped` | passo pulado por grad norm não finito (`loss`/`grad_norm` viram `null`) | raro. Vários seguidos: parar e investigar |
| `lr_muon` | lr do Muon (`initial_lr × wsd`), antes da escala `0,2·√max(A, B)` do `match_rms_adamw` | sobe por 1.000 passos, platô, decai a partir do passo ~30.517 |
| `tok_s`, `mfu` | throughput do passo e MFU (`flops_per_token × tok_s / pico`, pico 18,2 TFLOPs) | estáveis. Queda indica throttling térmico ou outro processo na GPU (`nvidia-smi`) |
| `mem_gb` | `max_memory_allocated` desde o início do processo (só sobe) | ≤ 7,0. Sobe na troca para 4K |
| `seq_len` | contexto do estágio | 2048 até 7B tokens, depois 4096 |

Resumo dos últimos 50 passos, mantendo o último registro de cada passo:

```bash
uv run python - <<'EOF'
import json
recs = {}
for line in open("runs/b0_4_pretrain/metrics.jsonl"):
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

Acompanhe também `tail -f runs/b0_4_pretrain/train.log` e `nvidia-smi dmon -s pucm` (potência, clocks
e memória). Num treino de meses, a temperatura importa: com a GPU acima de ~83 °C os clocks caem e o
`tok_s` acompanha. Limpe filtros e ventoinhas e, se preciso, limite a potência com
`sudo nvidia-smi -pl <watts>`: perde-se pouco throughput e ganha-se folga térmica.

## 8. Problemas conhecidos

- **Workaround do Inductor (torch 2.14.1).** Com `mix_order_reduction` ligado, a fusão quebra no 1º
  backward do SoreLM (`AssertionError "509 v.s. 510"`). O trainer (e o `bench.py`) desliga a fusão
  automaticamente em `compile_model` (`sore/train/trainer.py`). Remova quando houver fix upstream.
- **Flex com T < 128 usa o sdpa.** O flex compilado falha no Inductor com T=127 ("Invalid FlexAttention
  decode kernel options"). A `GatedAttention` cai no sdpa denso abaixo de `FLEX_MIN_LEN` = 128, o que não
  afeta o treino (N = 4096). O BlockMask do flex é montado uma vez por forward no `SoreLM` e
  compartilhado pelas camadas de atenção e pelo MTP.
- **Na retomada, `lr_muon`, `lr_adamw` e `weight_decay` vêm do YAML.** O trainer reescreve o
  `initial_lr` e o `weight_decay` de cada param group a partir do config e imprime `[resume] aviso: ...`
  para cada valor que mudou em relação ao checkpoint. Os demais hiperparâmetros dos param groups (betas,
  eps, momentum do Muon) voltam do checkpoint.
- **`metrics.jsonl` pode repetir passos depois de crash + retomada.** Os passos após o último checkpoint
  são refeitos e gravados de novo. Para cada `step`, vale o último registro.
- **Crash sem sinal perde até `ckpt_every_tokens` de trabalho.** Queda de energia, OOM ou `kill -9` não
  passam pelo handler de SIGINT/SIGTERM. No `b0_4` isso é até 1e7 tokens, ~38 passos, ~1,5–2 h. Um
  nobreak evita a maior parte dessas quedas.
- **Memória transitória da loss.** Com `fused_ce` (padrão do `b0_4`) a cabeça de LM e a cross-entropy
  rodam num kernel só do FLA, por chunks de `loss_chunk_size` (8192) tokens. Se o estágio de 4K der
  OOM, baixe `loss_chunk_size` em `model_overrides` (ex.: 2048; é campo de execução, aceito também em
  `--resume`) e repita o portão.
- **Compile por Block e graph breaks.** O `chunk_gated_delta_rule` do FLA é `torch.compiler.disable`.
  O `compile_model` compila cada Block in-place (tronco e MTP) e deixa o grad checkpoint por fora, porque
  o `torch.utils.checkpoint` desliga o Dynamo. Sobra um graph break inofensivo por bloco GDN, em volta do
  kernel do FLA.
- **N estático no compile.** O `compile_model` usa `dynamic=False`, porque o lowering do flex no Inductor
  falha com Q_LEN dinâmico (`CantSplit`). Só o nº de documentos (`len(cu_seqlens)`) é dinâmico.
- **Sem FP8 na 3050.** O FP8 (`fp8: rowwise`) exige sm89+ (4090, Hopper); na 3050 (sm86) o trainer recusa
  com `RuntimeError`. Na 4090 ele também não compensou (`history/relatorio_dia1_rtx4090.md`).

## 9. Amostra PT para testes

Os portões de qualidade (§11) e a validação (§4.3) precisam de texto de verdade: o shard sintético de
`make_synthetic_shard.py` é aprendido quase por completo. A amostra tem 3e8 tokens de mC4 português
(`allenai/c4`, `multilingual/c4-pt*.json.gz`, ODC-BY), tokenizados com o tokenizer do Qwen3.5 e
remapeados **provisoriamente** para 65.536 ids: EOS = 0, UNK = 1 e os 65.534 ids mais frequentes da
amostra viram 2, 3, ... Não é o vocabulário do pré-treino (§2), mas cabe no vocab de qualquer preset.

```bash
uv run python scripts/make_text_sample.py --out data/sample_pt --n-tokens 3e8
```

Precisa de rede (baixa 3 arquivos de ~200 MB em `data/raw/`, que o `prune_tokenizer.py` também usa).
Saída em `data/sample_pt/` (ignorada pelo git, menos `doc_lengths.json` e `meta.json`): `sample_000.bin`
(+`.json`, `uint16`), `remap.npy`, `doc_lengths.json` (até 100.000 comprimentos de documento, com EOS,
por reservoir sampling) e `meta.json`. Use `eos_id=0`. Para reproduzir exatamente a amostra de
2026-10-24, fixe o tokenizer com `--revision 2fc06364715b967f1860aea9cf38778875588b17` (o
`tokenizer_revision` do `meta.json`) e confira o `sha256sum data/sample_pt/sample_000.bin`.

## 10. Referência: benchmark na RTX 3050 Laptop 6GB (WSL2)

Medido em 2026-10-24 no notebook, antes da reestruturação: `s150m` com vocab 65.536 (164,3M parâmetros,
1,35 GFLOP/token a 2048), seq 2048 × micro_batch 1, `--grad-accum 2 --grad-checkpoint --compile`,
comprimentos de documento de `data/sample_pt/doc_lengths.json`, mediana de 8 passos depois de 2 de
warmup:

| Configuração | tok/s | TFLOP/s efetivos | VRAM pico (alocado / reservado) | data | fwd_bwd | clip | optimizer |
|---|---:|---:|---:|---:|---:|---:|---:|
| `s150m` (164,3M) | 3.036 | 4,1 | 2,99 / 3,34 GB | 0,006 s | 1,055 s | 0,019 s | 0,282 s |
| fatia do `b0_8` (`n_layers=4`, 221,4M) | 2.616 | 4,9 | 4,06 / 4,64 GB | 0,006 s | 1,136 s | 0,024 s | 0,431 s |

Fases em segundos por passo (2 × 2048 tokens). O MFU publicado na época usava um pico de 35 TFLOPs,
alto demais para BF16 com acumulação fp32 numa 3050 de notebook (~12 TFLOPs), então os TFLOP/s
efetivos são a referência útil. Leituras do perfil (`--profile`): a GPU não fica esperando a CPU (`idle`
≤ 3%), o matmul domina (57–68% do tempo de GPU), e o `optimizer` (Muon) pesa 21–28% do passo com
`--grad-accum 2`. No treino ele se dilui nos 64 micro-batches. As categorias de kernel vêm de
`sore/train/profiling.py` (tabela `_RULES`).

## 11. Campanha de qualidade

Toda alavanca de velocidade **exata** (que não deveria mudar a matemática) precisa de uma corrida
curta no `s150m` (`configs/s150m_quality.yaml`: 2e7 tokens da amostra PT, 305 passos de 65.536 tokens,
`compile: true`) comparada com duas referências. Uma alavanca **com perda** só entra com o gap de perda
dentro do limite.

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

- Cada corrida grava em `runs/quality/NOME`. Uma corrida que já terminou (`last.pt` e o último passo no
  `metrics.jsonl`) é pulada; uma interrompida é retomada de `last.pt`.
- `ref_s0` (seed 0) e `ref_s1` (seed 1) rodam com todas as alavancas da lista **desligadas** (`ALL_OFF`
  no topo do script). A comparação final só sai com as duas referências terminadas; o `compare_runs.py`
  imprime uma tabela markdown (perda final, gap, limiar, PASSA/REPROVA) e sai com 1 se alguma candidata
  reprova.
- Na 3050 de 8GB o `micro_batch` 2 a 2048 cabe. Se estourar a memória, use
  `EXTRA_SET="model_overrides.grad_checkpoint=true"` em **todas** as corridas, refs incluídas (apague
  `runs/quality` antes, para não misturar configurações).

**Como uma alavanca entra.** No `scripts/run_quality_gates.sh`: (1) acrescente a flag "desligada" da
alavanca a `ALL_OFF` (um `KEY=VALUE` por elemento, vale para todas as corridas); (2) acrescente a linha
`NOME:override` ligada em `RUNS`, por exemplo `"fused_ce:model_overrides.fused_ce=true"`. As overrides
de uma linha ficam depois de `ALL_OFF` e vencem. Várias overrides numa linha ou no `EXTRA_SET` se
separam por `;`. Alavanca com perda: o nome também vai em `LOSSY`. Mude `ALL_OFF` e as referências já
rodadas passam a ser de outra configuração: apague `runs/quality/ref_s*` e rode de novo.

**Ensaio curto** (valida o encanamento, não a qualidade; ~5 min, quase tudo é o `torch.compile`):

```bash
EXTRA_SET="total_tokens=2e5;warmup_steps=1;stages=[{until_tokens: 2e5, seq_len: 2048, micro_batch: 2}]" \
  bash scripts/run_quality_gates.sh --only ref_s0
```

`EXTRA_SET` vale para todas as corridas, separado por `;`. O `warmup_steps=1` é necessário: com 3
passos o warmup de 30 passos passaria do início do decay. Com `--set` direto:
`python -m sore.train --config configs/s150m_quality.yaml --set seed=1 --set model_overrides.fused_ce=true`;
`model_overrides.X` faz merge com o `model_overrides` do YAML, e chave desconhecida continua dando erro.
Apague `runs/quality/` depois do ensaio: a corrida curta deixa um `last.pt` que a campanha real
retomaria como se fosse dela.
