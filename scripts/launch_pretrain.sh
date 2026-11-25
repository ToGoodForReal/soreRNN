#!/usr/bin/env bash
# Pré-treino do b0_4 (docs/RUNBOOK.md §5): termina a geração de dados (com novas tentativas), confere os shards e
# o config contra eles e treina, retomando do last.pt se o processo cair. Para retomar sozinho depois de
# uma queda de energia, ponha no crontab do usuário:
#   @reboot PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin setsid nohup bash /caminho/do/repo/scripts/launch_pretrain.sh
#
# Variáveis de ambiente (padrões = configs/b0_4_pretrain.yaml):
#   CONFIG        config do treino (padrão: configs/b0_4_pretrain.yaml)
#   DATA_DIR      saída do make_pretrain_data.py; precisa casar com o `data_paths` do config (padrão: data/pretrain)
#   TOKENIZER_DIR tokenizer podado do prune_tokenizer.py (padrão: data/tokenizer)
#   TOTAL_TOKENS  tokens a gerar (padrão: 10e9, o `total_tokens` do config)
#   RUN_DIR       `out_dir` do config, onde ficam last.pt e os logs (padrão: runs/b0_4_pretrain)
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO" || exit 1
CONFIG=${CONFIG:-configs/b0_4_pretrain.yaml}
DATA_DIR=${DATA_DIR:-data/pretrain}
TOKENIZER_DIR=${TOKENIZER_DIR:-data/tokenizer}
TOTAL_TOKENS=${TOTAL_TOKENS:-10e9}
RUN_DIR=${RUN_DIR:-runs/b0_4_pretrain}
DATA_TRIES=5
TRAIN_TRIES=20
mkdir -p "$RUN_DIR" "$DATA_DIR"
LOG=$RUN_DIR/launch.log
DATA_LOG=$DATA_DIR/make_pretrain_data.log
CKPT=$RUN_DIR/last.pt

# No boot a GPU pode demorar a aparecer: espera até 15 min
for i in $(seq 1 90); do nvidia-smi >/dev/null 2>&1 && break; sleep 10; done

# uma cópia por vez: um @reboot com outra já rodando sai sem fazer nada
exec 9>"$RUN_DIR/launch.lock"
if ! flock -n 9; then echo "[$(date)] outra cópia do lançador já está rodando" >>"$LOG"; exit 0; fi

exec >>"$LOG" 2>&1
echo "[$(date)] lançador iniciado (repo $REPO, config $CONFIG)"

data_done() { grep -q "^\[pretrain\] total" "$DATA_LOG" 2>/dev/null; }

# 1. geração de dados: retomável, então uma queda só refaz o arquivo em curso
for try in $(seq 1 $DATA_TRIES); do
    data_done && break
    echo "[$(date)] geração de dados: tentativa $try/$DATA_TRIES"
    uv run python scripts/make_pretrain_data.py --tokenizer-dir "$TOKENIZER_DIR" --out "$DATA_DIR" \
        --total-tokens "$TOTAL_TOKENS" >>"$DATA_LOG" 2>&1
    echo "[$(date)] geração saiu com código $?"
    sleep 30
done
if ! data_done; then
    echo "[$(date)] ABORTADO: a geração não terminou após $DATA_TRIES tentativas; ver $DATA_LOG"
    exit 1
fi

# 2. conferência dos shards e do config contra o manifest dos dados
if ! uv run python - "$DATA_DIR" "$CONFIG" <<'EOF'
import json, sys
from pathlib import Path
from sore.model.config import get_preset
from sore.train.config import load_train_config

out, cfg = Path(sys.argv[1]), load_train_config(Path(sys.argv[2]))
m = json.loads((out / "manifest.json").read_text())
ok = True
for name, w in m["mixture"].items():
    p = json.loads((out / name / "progress.json").read_text())
    quota = int(w * m["total_tokens"])
    status = "ok" if p["tokens"] >= quota * 0.99 else ("esgotada" if p["exhausted"] else "INCOMPLETA")
    ok &= status != "INCOMPLETA"
    print(f"  {name}: {p['tokens']:,} / {quota:,} ({status})")
    if list((out / name).glob("*.tmp")):
        print(f"  {name}: shard .tmp órfão"); ok = False
vocab = get_preset(cfg.model_preset, **cfg.model_overrides).vocab_size
if cfg.eos_id != m["eos_id"]:
    print(f"  eos_id do config ({cfg.eos_id}) != eos_id dos dados ({m['eos_id']})"); ok = False
if m["vocab_size"] > vocab:
    print(f"  vocab dos dados ({m['vocab_size']}) > vocab do modelo ({vocab})"); ok = False
if cfg.total_tokens > m["total_tokens"]:
    print(f"  aviso: total_tokens do config ({cfg.total_tokens:,}) > tokens gerados ({m['total_tokens']:,}): "
          "o stream repete épocas")
sys.exit(0 if ok else 1)
EOF
then
    echo "[$(date)] ABORTADO: dados incompletos ou config incompatível com eles"
    exit 1
fi
echo "[$(date)] dados conferidos"

# 3. pré-treino: retoma do last.pt a cada queda, até TRAIN_TRIES vezes
for try in $(seq 1 $TRAIN_TRIES); do
    resume=()
    [ -f "$CKPT" ] && resume=(--resume "$CKPT")
    echo "[$(date)] pré-treino: tentativa $try/$TRAIN_TRIES ${resume[*]:-(do zero)}"
    uv run python -m sore.train --config "$CONFIG" "${resume[@]}" >>"$RUN_DIR/train.log" 2>&1
    code=$?
    echo "[$(date)] pré-treino saiu com código $code"
    if [ $code -eq 0 ]; then
        echo "[$(date)] pré-treino concluído (ou parado de propósito com SIGTERM/SIGINT)"
        exit 0
    fi
    sleep 120
done
echo "[$(date)] ABORTADO: o pré-treino caiu $TRAIN_TRIES vezes; ver $RUN_DIR/train.log"
exit 1
