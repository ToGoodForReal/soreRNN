#!/bin/bash
set -e

# ==============================================================================
# Pipeline de Produção soreRNN-LM: Pré-Treino Massivo (4B Tok) + SFT (500M Tok)
# Otimizado para NVIDIA RTX 3050 8GB VRAM (CUDA 13.3)
# ==============================================================================

PRETRAIN_DATA="data/pretrain_4b.bin"
SFT_DATA="data/sft_500m.bin"
VAL_DATA="data/val.bin"

# Hiperparâmetros de Pré-Treino (Adaptados para 8GB VRAM)
PRETRAIN_STEPS=100000   # ~1.64B tokens
SFT_STEPS=2500          # Ajuste fino com loss masking
MICRO_BATCH=4
GRAD_ACCUM=4            # 4 * 4 * 1024 = 16,384 tokens/passo otimizador
SEQ_LEN=1024

echo "================================================================="
echo "   INICIANDO PIPELINE DE PRODUÇÃO soreRNN-LM v2 (152.3M PARAMS)  "
echo "================================================================="
echo " GPU Target: NVIDIA RTX 3050 8GB VRAM"
echo " Micro-Batch: ${MICRO_BATCH} | Grad Accum: ${GRAD_ACCUM} | Seq Len: ${SEQ_LEN}"
echo " Tokens/Passo: $(( MICRO_BATCH * GRAD_ACCUM * SEQ_LEN )) tokens"
echo " Passos Pré-Treino: ${PRETRAIN_STEPS} | Passos SFT: ${SFT_STEPS}"
echo "================================================================="

# 1. Compilação
echo -e "\n[1/3] Compilando binários nativos C++/CUDA..."
cmake --build build -j$(nproc) --target train_lm_150m train_sft train_dpo

# 2. Execução do Pré-Treino (Fase 1)
if [ -f "$PRETRAIN_DATA" ]; then
    echo -e "\n[2/3] Executando Fase 1: Pré-Treino Massivo em ${PRETRAIN_DATA}..."
    ./build/train_lm_150m ${PRETRAIN_STEPS} ${SFT_STEPS} ${MICRO_BATCH} ${GRAD_ACCUM} ${SEQ_LEN}
else
    echo -e "\n[Aviso] Arquivo ${PRETRAIN_DATA} não encontrado. Execute primeiro:"
    echo "  python3 scripts/prepare_4b_pretrain.py --out ${PRETRAIN_DATA}"
fi

# 3. Execução do SFT Especializado (Fase 2)
if [ -f "$SFT_DATA" ]; then
    echo -e "\n[3/3] Executando Fase 2: SFT Multi-Tarefa com Loss Masking e Validação..."
    SFT_VAL="data/sft_val.bin"
    ./build/train_sft "checkpoints/sore_lm_150m_pretrain.bin" "$SFT_DATA" "checkpoints/sore_lm_150m_sft.bin" ${SFT_STEPS} ${MICRO_BATCH} ${SEQ_LEN} "$SFT_VAL"
fi

echo -e "\n================================================================="
echo " [CONCLUÍDO] Pipeline de Produção Finalizado!"
echo " Para testar inferência e chat:"
echo "  python3 scripts/chat.py --ckpt checkpoints/sore_lm_150m_sft.bin"
echo "================================================================="
