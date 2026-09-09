#!/usr/bin/env bash
# ==============================================================================
# PIPELINE DE TESTE: SFT MULTI-TAREFA (CONVERSAÇÃO + MATEMÁTICA CoT + LÓGICA)
# Executa de forma isolada e leve (~2.2 GB VRAM) em paralelo com o pré-treino principal.
# ==============================================================================
set -e

mkdir -p build data checkpoints logs

echo "================================================================="
echo "   soreRNN-LM: TESTE DE SFT MULTI-TAREFA EM PARALELO"
echo "================================================================="

# 1. Compilar executável dedicado train_sft (não afeta o train_lm_150m em execução)
echo -e "\n[Passo 1/4] Compilando executável dedicado train_sft..."
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build --target train_sft -j4

# 2. Gerar dataset SFT multi-tarefa balanceado
echo -e "\n[Passo 2/4] Gerando dataset SFT multi-tarefa (Conversação + Math CoT + Lógica)..."
if [ ! -f "data/sft_multitask_test.bin" ]; then
    python3 scripts/prepare_sft_multitask.py --out data/sft_multitask_test.bin --tokens 1500000
else
    echo " -> Dataset 'data/sft_multitask_test.bin' já existe. Usando versão existente."
fi

# 3. Executar Fine-Tuning Multi-Tarefa (150 passos, B=2, T=256 -> consome apenas 2.2 GB VRAM)
echo -e "\n[Passo 3/4] Executando SFT Multi-Tarefa (150 passos rápidos)..."
./build/train_sft \
    checkpoints/sore_lm_150m_pretrain.bin \
    data/sft_multitask_test.bin \
    checkpoints/sore_lm_150m_sft_multitask.bin \
    150 2 256 data/sft_multitask_val.bin 2>&1 | tee logs/sft_multitask_run.log

# 4. Avaliação Comparativa Automatizada
echo -e "\n[Passo 4/4] Avaliando modelo pós-SFT Multi-Tarefa..."
python3 scripts/test_multitask_evaluation.py \
    --checkpoint checkpoints/sore_lm_150m_sft_multitask.bin 2>&1 | tee logs/multitask_sft_test_report.txt

echo -e "\n================================================================="
echo " [SUCESSO] Teste concluído! Relatório salvo em logs/multitask_sft_test_report.txt"
echo "================================================================="
