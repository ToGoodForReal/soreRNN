#!/usr/bin/env bash
# Pré-treino de 50B do b0_8: termina a geração de dados (com novas tentativas), confere os shards e
# treina, retomando do last.pt se o processo cair. Fica no ext4 (não na partição de dados) para o
# @reboot do crontab achá-lo mesmo quando a partição ainda não montou:
#   @reboot PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin setsid nohup bash .../scripts/launch_pretrain.sh
set -u
W=/mnt/windows/soreRNN
REPO=$(cd "$(dirname "$0")/.." && pwd)
LOG=$W/launch_after_data.log
CKPT=$W/runs/b0_8_pretrain/last.pt
DATA_TRIES=5
TRAIN_TRIES=20
BOOT_LOG=$HOME/.cache/soreRNN_launch_boot.log
mkdir -p "$(dirname "$BOOT_LOG")"

# No boot a partição e a GPU podem demorar: espera até 15 min por cada uma. Uma queda de energia
# deixa o NTFS "sujo" e ele não monta sem reparo (ntfsfix, como root): isso fica registrado aqui.
echo "[$(date)] lançador iniciado" >>"$BOOT_LOG"
for i in $(seq 1 90); do [ -d "$W/runs" ] && break; sleep 10; done
if [ ! -d "$W/runs" ]; then
    echo "[$(date)] $W não montada após 15 min (NTFS sujo após queda de energia?); treino não retomado" >>"$BOOT_LOG"
    exit 1
fi
for i in $(seq 1 90); do nvidia-smi >/dev/null 2>&1 && break; sleep 10; done

# uma cópia por vez: um @reboot com outra já rodando sai sem fazer nada
exec 9>"$W/launch_after_data.lock"
if ! flock -n 9; then echo "[$(date)] outra cópia do lançador já está rodando" >>"$BOOT_LOG"; exit 0; fi

exec >>"$LOG" 2>&1
cd "$REPO" || exit 1
echo "[$(date)] lançador iniciado (repo $REPO)"

data_done() { grep -q "^\[pretrain\] total" "$W/make_pretrain_data.log" 2>/dev/null; }

# 1. geração de dados: retomável, então uma queda só refaz o arquivo em curso
for try in $(seq 1 $DATA_TRIES); do
    data_done && break
    echo "[$(date)] geração de dados: tentativa $try/$DATA_TRIES"
    uv run python scripts/make_pretrain_data.py --tokenizer-dir data/tokenizer --out "$W/pretrain" \
        --total-tokens 50e9 >>"$W/make_pretrain_data.log" 2>&1
    echo "[$(date)] geração saiu com código $?"
    sleep 30
done
if ! data_done; then
    echo "[$(date)] ABORTADO: a geração não terminou após $DATA_TRIES tentativas; ver make_pretrain_data.log"
    exit 1
fi

# 2. conferência dos shards
if ! python3 - "$W/pretrain" <<'EOF'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
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
sys.exit(0 if ok else 1)
EOF
then
    echo "[$(date)] ABORTADO: alguma fonte incompleta"
    exit 1
fi
echo "[$(date)] dados conferidos"

# 3. pré-treino: retoma do last.pt a cada queda, até TRAIN_TRIES vezes
for try in $(seq 1 $TRAIN_TRIES); do
    resume=()
    [ -f "$CKPT" ] && resume=(--resume "$CKPT")
    echo "[$(date)] pré-treino: tentativa $try/$TRAIN_TRIES ${resume[*]:-(do zero)}"
    uv run python -m sore.train --config configs/b0_8_pretrain.yaml "${resume[@]}" >>"$W/b0_8_pretrain.log" 2>&1
    code=$?
    echo "[$(date)] pré-treino saiu com código $code"
    if [ $code -eq 0 ]; then
        echo "[$(date)] pré-treino concluído (ou parado de propósito com SIGTERM/SIGINT)"
        exit 0
    fi
    sleep 120
done
echo "[$(date)] ABORTADO: o pré-treino caiu $TRAIN_TRIES vezes; ver b0_8_pretrain.log"
exit 1
