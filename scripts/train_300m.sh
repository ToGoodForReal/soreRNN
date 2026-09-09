#!/usr/bin/env bash
set -e

# ==============================================================================
#  soreRNN-LM (300M Parâmetros): Treinamento em GPU NVIDIA GeForce RTX 3050 (8GB VRAM)
#  Arquitetura: 18 Camadas, D=1280, d_mlp=3200, RG-LRU + TF32 + Gradient Accumulation
# ==============================================================================

PRETRAIN_STEPS=${1:-15000}
MICRO_BATCH=${2:-2}
GRAD_ACCUM=${3:-16}
SEQ_LEN=${4:-512}

mkdir -p checkpoints logs data

EFFECTIVE_BATCH=$(( MICRO_BATCH * GRAD_ACCUM ))
TOKENS_PER_STEP=$(( EFFECTIVE_BATCH * SEQ_LEN ))
TOTAL_TOKENS_TARGET=$(( PRETRAIN_STEPS * TOKENS_PER_STEP ))

echo "================================================================="
echo "   INICIANDO PIPELINE DE TREINAMENTO DO soreRNN-LM 300M          "
echo "================================================================="
echo " Hardware: NVIDIA GeForce RTX 3050 (8GB VRAM) / Ampere (sm_86)"
echo " Otimizações Ativas: Tensor Cores TF32 + Timescale Init + Grad Accum"
echo " Configuração:"
echo "   - Passos de Pré-Treino:   $PRETRAIN_STEPS"
echo "   - Micro-Batch (GPU):      $MICRO_BATCH"
echo "   - Gradient Accumulation:  $GRAD_ACCUM passos"
echo "   - Batch Efetivo Total:    $EFFECTIVE_BATCH sequências"
echo "   - Janela de Contexto:     $SEQ_LEN tokens"
echo "   - Tokens por Passo:       $TOKENS_PER_STEP tokens"
echo "   - Total Tokens Alvo:      $TOTAL_TOKENS_TARGET tokens"
echo "================================================================="

# 1. Compilar executável do 300M
echo -e "\n[1/3] Compilando engine soreRNN-LM 300M..."
cmake -S . -B build
cmake --build build --target train_lm_300m -j

PYTHON_BIN="python3"
if [ -f ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
fi

# 2. Verificação dos Datasets (Prioriza Interleaving Dinâmico Anti-Esquecimento)
echo -e "\n[2/3] Verificando integridade dos datasets de treino..."
DATASET_FILE="data/pretrain_interleaved.bin"
if [ ! -f "$DATASET_FILE" ]; then
    if [ -f "data/pretrain_75pt_25en.bin" ]; then
        DATASET_FILE="data/pretrain_75pt_25en.bin"
        echo " -> Usando dataset existente: $DATASET_FILE"
    else
        echo " -> Dataset intercalado não encontrado. Gerando stream dinâmico anti-esquecimento..."
        $PYTHON_BIN scripts/prepare_interleaved_data.py --out data/pretrain_interleaved.bin --tokens 500000000
        DATASET_FILE="data/pretrain_interleaved.bin"
    fi
else
    echo " -> Dataset intercalado encontrado: $DATASET_FILE"
    ls -lh "$DATASET_FILE"
fi

# 3. Executar o Treinamento do Modelo de 300M
LOG_FILE="logs/train_300m_$(date +%Y%m%d_%H%M%S).log"
echo -e "\n[3/3] Iniciando treinamento acelerado com Tensor Cores TF32..."
echo " -> Log de execução sendo gravado em: $LOG_FILE"

./build/train_lm_300m "$PRETRAIN_STEPS" "$MICRO_BATCH" "$GRAD_ACCUM" "$SEQ_LEN" 2>&1 | tee "$LOG_FILE"

echo -e "\n================================================================="
echo " Treinamento do soreRNN-LM 300M concluído com sucesso!"
echo " Modelo final salvo em: checkpoints/sore_lm_300m_final.bin"
echo "================================================================="
