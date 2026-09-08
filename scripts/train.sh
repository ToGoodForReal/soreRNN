#!/usr/bin/env bash
set -e

# ==============================================================================
#  soreRNN-LM v2 (152.3M Parâmetros): Treinamento em GPU L40S (48GB VRAM)
#  Arquitetura: Conv1D (K=4) + RG-LRU + GELU MLP + Weight Tying
# ==============================================================================

PRETRAIN_STEPS=${1:-10000}
SFT_STEPS=${2:-300}
BATCH_SIZE=${3:-32}
SEQ_LEN=${4:-1024}

mkdir -p checkpoints logs data

echo "================================================================="
echo "   INICIANDO PIPELINE DE TREINAMENTO DO soreRNN-LM v2 (152.3M)   "
echo "================================================================="
echo " Hardware Alvo: NVIDIA L40S (48GB VRAM) / Ada Lovelace (sm_89)   "
echo " Configuração:"
echo "   - Passos de Pré-Treino: $PRETRAIN_STEPS"
echo "   - Passos de SFT Chat:   $SFT_STEPS"
echo "   - Tamanho do Batch:     $BATCH_SIZE"
echo "   - Janela de Contexto:   $SEQ_LEN tokens"
echo "   - Tokens por Passo:     $(( BATCH_SIZE * SEQ_LEN ))"
echo "   - Total Tokens Alvo:    $(( PRETRAIN_STEPS * BATCH_SIZE * SEQ_LEN ))"
echo "================================================================="

# 1. Compilação da Engine C++20 / CUDA
echo -e "\n[1/3] Compilando engine soreRNN-LM v2 C++20 / CUDA..."
cmake -S . -B build
cmake --build build --target train_lm_150m -j

# 2. Verificação dos Datasets
echo -e "\n[2/3] Verificando integridade dos datasets binários..."
if [ ! -f "data/pretrain_75pt_25en.bin" ] || [ ! -f "data/sft_chat_pt.bin" ]; then
    echo " -> Datasets não encontrados. Executando streaming de 4B tokens..."
    python3 scripts/prepare_data.py --target_4b --tokens_sft 5000000
else
    echo " -> Datasets existentes encontrados em data/."
    ls -lh data/
fi

# 3. Execução do Treinamento
LOG_FILE="logs/train_$(date +%Y%m%d_%H%M%S).log"
echo -e "\n[3/3] Iniciando treinamento acelerado 100% em GPU..."
echo " -> Log de execução sendo gravado em: $LOG_FILE"

./build/train_lm_150m "$PRETRAIN_STEPS" "$SFT_STEPS" "$BATCH_SIZE" "$SEQ_LEN" 2>&1 | tee "$LOG_FILE"

echo -e "\n================================================================="
echo " Treinamento concluído com sucesso!"
echo " Modelo final salvo em: checkpoints/sore_lm_150m_final.bin"
echo "================================================================="

# 4. Teste Rápido de Inferência
echo -e "\n[Teste de Inferência O(1)] Testando geração de resposta com soreRNN-v2..."
python3 scripts/chat.py --checkpoint checkpoints/sore_lm_150m_final.bin \
    --prompt "<|user|>\nOlá, quem é você?\n<|assistant|>\n" \
    --max_tokens 50 --temp 0.7 || true
