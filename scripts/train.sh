#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
#  soreRNN-LM — Runner de Treinamento UNIFICADO
#  Arquitetura: Conv1D(K) + RG-LRU + GELU MLP + Weight Tying  |  C++20/CUDA
# ==============================================================================
#  Uso:
#     scripts/train.sh [MODE] [MODEL] [PASSOS]
#       MODE   = pretrain | sft | dpo | all | test   (default: all)
#       MODEL  = 150m | 300m                         (default: 150m)
#       PASSOS = passos de pre-treino                 (default: depende do modelo)
#
#  Exemplos:
#     scripts/train.sh pretrain 150m 10000
#     scripts/train.sh all 300m 15000
#     scripts/train.sh test 150m
# ==============================================================================

MODE="${1:-all}"
MODEL="${2:-150m}"
cd "$(dirname "$0")/.."
mkdir -p build data checkpoints logs

PYTHON_BIN="python3"
[ -f ".venv/bin/python" ] && PYTHON_BIN=".venv/bin/python"

# ---- Hiperparâmetros por modelo (ajustados p/ RTX 3050 8GB) ----------------
if [ "$MODEL" = "300m" ]; then
  BIN="train_lm_300m";  STEPS="${3:-15000}"; MICRO=2; ACCUM=4; SEQ=512
  PRETRAIN_DATA="data/pretrain_interleaved.bin"; SFT_DATA="data/sft_500m.bin"
else
  BIN="train_lm_150m";  STEPS="${3:-10000}"; MICRO=4; ACCUM=1; SEQ=1024
  PRETRAIN_DATA="data/pretrain_75pt_25en.bin"; SFT_DATA="data/sft_chat_pt.bin"
fi
VAL_DATA="data/val.bin"; DPO_DATA="data/dpo_pairs.bin"
PRE_CKPT="checkpoints/sore_lm_${MODEL}_pretrain.bin"
SFT_CKPT="checkpoints/sore_lm_${MODEL}_sft.bin"
DPO_CKPT="checkpoints/sore_lm_${MODEL}_dpo.bin"

echo "=============================================================="
echo " soreRNN-LM  MODE=$MODE  MODEL=$MODEL  (bin=$BIN)"
echo " Passos=$STEPS  Micro=$MICRO  Accum=$ACCUM  SeqLen=$SEQ"
echo "=============================================================="

# ---- Build (compila os alvos necessários para o modo) ----------------------
build_targets=()
case "$MODE" in
  pretrain|all|test) build_targets+=("$BIN");;
esac
case "$MODE" in
  sft|all|dpo|test) build_targets+=(train_sft);;
esac
case "$MODE" in
  dpo|all) build_targets+=(train_dpo);;
esac
[ ${#build_targets[@]} -eq 0 ] && build_targets=("train_lm_150m")
echo -e "\n[Build] ${build_targets[*]}"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build --target "${build_targets[@]}" -j"$(nproc)"

# ---- Fases ------------------------------------------------------------------
do_pretrain() {
  echo -e "\n[Fase: PRÉ-TREINO]"
  if [ ! -f "$PRETRAIN_DATA" ]; then
    echo " -> Gerando pré-treino + val held-out (dedup + interleave fino + EOS)..."
    "$PYTHON_BIN" scripts/prepare_4b_pretrain.py \
        --out "$PRETRAIN_DATA" --val_out "$VAL_DATA" --tokens 50000000 --val_tokens 1000000
  fi
  ./build/"$BIN" "$STEPS" "$MICRO" "$ACCUM" "$SEQ" 2>&1 | tee "logs/pretrain_${MODEL}_$(date +%H%M%S).log"
}

do_sft() {
  echo -e "\n[Fase: SFT (máscara de loss + val)]"
  if [ ! -f "$SFT_DATA" ]; then
    echo " -> Gerando SFT (Alpaca-PT/Dolly + CoT, masking 65535)..."
    "$PYTHON_BIN" scripts/prepare_500m_sft.py --out "$SFT_DATA" --val_out data/sft_val.bin --tokens 5000000
  fi
  ./build/train_sft "$PRE_CKPT" "$SFT_DATA" "$SFT_CKPT" 300 2 256 2>&1 | tee "logs/sft_${MODEL}.log"
}

do_dpo() {
  echo -e "\n[Fase: DPO (preferência)]"
  if [ ! -f "$DPO_DATA" ]; then
    "$PYTHON_BIN" scripts/prepare_dpo_dataset.py --out "$DPO_DATA" --count 5000
  fi
  ./build/train_dpo "$SFT_CKPT" "$DPO_DATA" "$DPO_CKPT" 300 0.1 2>&1 | tee "logs/dpo_${MODEL}.log"
}

do_test() {
  echo -e "\n[Fase: AVALIAÇÃO]"
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

echo -e "\n[CONCLUÍDO] $MODE / $MODEL. Use: python3 scripts/chat.py --checkpoint <ckpt>"
