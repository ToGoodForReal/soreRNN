#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
#  soreRNN-LM — Runner de Treinamento UNIFICADO
#  Pipeline: PRE-TREINO -> SFT -> DPO -> AVAL  (cada etapa salva seu checkpoint)
# ==============================================================================
#  Uso:
#     scripts/train.sh [MODE] [MODEL] [PASSOS_PRE]
#       MODE   = pretrain | sft | dpo | all | test   (default: all)
#       MODEL  = 150m | 300m                         (default: 150m)
#       PASSOS = passos de pré-treino                (default: por modelo)
#
#  Orçamento de dados (tokens) configurável por variável de ambiente:
#       PRE_TOKENS   (default 200000000)   SFT_TOKENS (default 8000000)
#     Ex.: PRE_TOKENS=1000000000 scripts/train.sh all 300m
# ==============================================================================

MODE="${1:-all}"
MODEL="${2:-150m}"
cd "$(dirname "$0")/.."
mkdir -p build data checkpoints logs
PYTHON_BIN="python3"; [ -f ".venv/bin/python" ] && PYTHON_BIN=".venv/bin/python"

PRE_TOKENS="${PRE_TOKENS:-200000000}"
SFT_TOKENS="${SFT_TOKENS:-8000000}"

if [ "$MODEL" = "300m" ]; then
  BIN="train_lm_300m";  STEPS="${3:-150000}"; MICRO=1; ACCUM=16; SEQ=512  # micro=1 cabe em 8GB (~6GB VRAM, ~3250 tok/s)
  PRETRAIN_DATA="data/pretrain_interleaved.bin"; SFT_DATA="data/sft_500m.bin"; SFT_BATCH=1
else
  BIN="train_lm_150m";  STEPS="${3:-100000}"; MICRO=4; ACCUM=4; SEQ=1024
  PRETRAIN_DATA="data/pretrain_75pt_25en.bin";  SFT_DATA="data/sft_chat_pt.bin"; SFT_BATCH=4
fi
VAL_DATA="data/val.bin"; DPO_DATA="data/dpo_pairs.bin"
PRE_CKPT="checkpoints/sore_lm_${MODEL}_pretrain.bin"
SFT_CKPT="checkpoints/sore_lm_${MODEL}_sft.bin"
DPO_CKPT="checkpoints/sore_lm_${MODEL}_dpo.bin"

echo "=============================================================="
echo " soreRNN-LM  MODE=$MODE  MODEL=$MODEL (bin=$BIN)"
echo " Pre-steps=$STEPS  micro=$MICRO accum=$ACCUM seq=$SEQ (batch efetivo=$((MICRO*ACCUM)))"
echo " Tokens: pre=$PRE_TOKENS  sft=$SFT_TOKENS"
echo "=============================================================="

build_targets=()
case "$MODE" in pretrain|all|test) build_targets+=("$BIN");; esac
case "$MODE" in sft|all|dpo|test)  build_targets+=(train_sft);; esac
case "$MODE" in dpo|all)           build_targets+=(train_dpo);; esac
[ ${#build_targets[@]} -eq 0 ] && build_targets=("train_lm_150m")
echo -e "\n[Build] ${build_targets[*]}"
if command -v cmake >/dev/null 2>&1; then
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/dev/null
  cmake --build build --target "${build_targets[@]}" -j"$(nproc)"
else
  echo " -> cmake não encontrado; utilizando binários já compilados em build/."
  for t in "${build_targets[@]}"; do
    if [ ! -x "build/$t" ]; then
      echo "[Erro] Binário 'build/$t' não encontrado. Instale o cmake (sudo pacman -S cmake) para compilar."
      exit 1
    fi
  done
fi

do_pretrain() {
  echo -e "\n[Fase 1/3: PRÉ-TREINO (puro)]"
  if [ ! -f "$PRETRAIN_DATA" ]; then
    echo " -> Gerando pré-treino + val held-out (dedup + interleave + EOS, tok PT)..."
    "$PYTHON_BIN" scripts/prepare_4b_pretrain.py \
        --out "$PRETRAIN_DATA" --val_out "$VAL_DATA" --tokens "$PRE_TOKENS" --val_tokens 1000000
  fi
  # ATENÇÃO: train_lm_150m = [pretrain_steps][sft_steps][micro][accum][seq]
  #          train_lm_300m = [pretrain_steps][micro][accum][seq]
  # Mantemos SFT interno DESLIGADO (sft_steps=0) -> SFT roda depois, uma vez só.
  # RESUME=1 -> retoma do checkpoint mais recente (best.bin ou pretrain.bin, ou RESUME_CKPT)
  RESUME_ARG="${RESUME_CKPT:-}"
  if [ -z "$RESUME_ARG" ] && [ "${RESUME:-0}" = "1" ]; then
    BEST_CKPT="checkpoints/sore_lm_${MODEL}_best.bin"
    if [ -f "$BEST_CKPT" ] && { [ ! -f "$PRE_CKPT" ] || [ "$BEST_CKPT" -nt "$PRE_CKPT" ]; }; then
      RESUME_ARG="$BEST_CKPT"
    elif [ -f "$PRE_CKPT" ]; then
      RESUME_ARG="$PRE_CKPT"
    fi
  fi
  if [ -n "$RESUME_ARG" ] && [ -f "$RESUME_ARG" ]; then
    echo " -> Retomando do checkpoint: $RESUME_ARG"
  fi
  if [ "$MODEL" = "300m" ]; then
    ./build/train_lm_300m "$STEPS" "$MICRO" "$ACCUM" "$SEQ" $RESUME_ARG 2>&1 | tee -a "logs/pretrain_${MODEL}.log"
  else
    ./build/train_lm_150m "$STEPS" 0 "$MICRO" "$ACCUM" "$SEQ" $RESUME_ARG 2>&1 | tee -a "logs/pretrain_${MODEL}.log"
  fi
}

do_sft() {
  echo -e "\n[Fase 2/3: SFT (máscara de loss + validação)]"
  [ -f "$PRE_CKPT" ] || { echo "[Erro] Falta $PRE_CKPT (rode pretrain antes)."; exit 1; }
  if [ ! -f "$SFT_DATA" ]; then
    "$PYTHON_BIN" scripts/prepare_500m_sft.py --out "$SFT_DATA" --val_out data/sft_val.bin --tokens "$SFT_TOKENS"
  fi
  ./build/train_sft "$PRE_CKPT" "$SFT_DATA" "$SFT_CKPT" 1500 "$SFT_BATCH" 512 data/sft_val.bin 2>&1 | tee "logs/sft_${MODEL}.log"
}

do_dpo() {
  echo -e "\n[Fase 3/3: DPO (otimização por preferência)]"
  [ -f "$SFT_CKPT" ] || { echo "[Erro] Falta $SFT_CKPT (rode sft antes)."; exit 1; }
  if [ ! -f "$DPO_DATA" ]; then
    "$PYTHON_BIN" scripts/prepare_dpo_dataset.py --out "$DPO_DATA" --count 20000
  fi
  ./build/train_dpo "$SFT_CKPT" "$DPO_DATA" "$DPO_CKPT" 600 0.1 2>&1 | tee "logs/dpo_${MODEL}.log"
}

do_test() {
  echo -e "\n[AVALIAÇÃO]"
  CK="$DPO_CKPT"; [ -f "$CK" ] || CK="$SFT_CKPT"; [ -f "$CK" ] || CK="$PRE_CKPT"
  if [ -f "$CK" ]; then "$PYTHON_BIN" scripts/evaluate.py --checkpoint "$CK"; else echo "Nenhum checkpoint para avaliar."; fi
}

case "$MODE" in
  pretrain) do_pretrain;;
  sft)      do_sft;;
  dpo)      do_dpo;;
  test)     do_test;;
  all)      do_pretrain; do_sft; do_dpo; do_test;;
  *) echo "MODE inválido: use pretrain|sft|dpo|all|test"; exit 1;;
esac
echo -e "\n[CONCLUÍDO] $MODE / $MODEL. Checkpoints em ./checkpoints/"
echo "Conversar: python3 scripts/chat.py --checkpoint $DPO_CKPT"
