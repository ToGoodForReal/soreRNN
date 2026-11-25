#!/usr/bin/env bash
# Matriz de benchmark e perfilamento de um preset (padrão: b0_4 na RTX 3050 8GB; docs/RUNBOOK.md §4.1).
#
# Variáveis de ambiente:
#   PRESET            preset do bench.py (padrão: b0_4)
#   OUT_DIR           diretório de saída (padrão: runs/bench_matrix)
#   PRESET_OVERRIDES  overrides extras passados ao bench.py (padrão vazio; ex: "--override n_layers=4" no ensaio)
#   SEQS              pares seq_len x micro_batch (padrão: "4096x1 2048x2", os estágios do b0_4)
#   GRAD_ACCUM        micro-batches por passo no bench (padrão: 8)
#   FP8               1 roda também a melhor exata com FP8 rowwise (só GPUs sm89+, como a 4090; padrão: 0)
#
# Matriz por seq:
#   1. base (todas as alavancas desligadas, full, com --profile);
#   2. cada alavanca exata sozinha (fused_ce, gdn_path, ckpt_selective, ckpt_ffn);
#   3. todas as exatas juntas;
#   4. todas as exatas x cada política de checkpoint x checkpoint_frac {1.0, 0.5, 0.25};
#   5. a melhor configuração x --profile (e x fp8 rowwise com FP8=1; tensorwise + compile é recusado:
#      gera grad NaN no torch 2.14.1 / torchao 0.18). A RTX 3050 (sm86) não tem FP8.
#
# No fim, imprime a tabela sumarizada via scripts/summarize_bench.py.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"

PRESET=${PRESET:-b0_4}
OUT_DIR=${OUT_DIR:-runs/bench_matrix}
PRESET_OVERRIDES=${PRESET_OVERRIDES:-}
SEQS=${SEQS:-"4096x1 2048x2"}
GRAD_ACCUM=${GRAD_ACCUM:-8}
FP8=${FP8:-0}

mkdir -p "$OUT_DIR"
BENCH_JSONL="$OUT_DIR/bench.jsonl"

echo "=== soreRNN — matriz de benchmark do preset $PRESET ==="
echo "Saída: $BENCH_JSONL"
echo "Sequências: $SEQS"
echo "Overrides extras: ${PRESET_OVERRIDES:-nenhum}"
echo "Grad accum: $GRAD_ACCUM | FP8: $FP8"
echo "================================================"

run_bench() {
    local profile_flag="$1"
    shift
    local seq_len="$1"
    local mb="$2"
    shift 2

    # Monta comando do benchmark com os parâmetros fixos da matriz
    # OOM registra linha em BENCH_JSONL e sai com 1; continuamos a matriz
    local cmd=(
        uv run python scripts/bench.py
        --preset "$PRESET"
        --seq-len "$seq_len"
        --micro-batch "$mb"
        --grad-accum "$GRAD_ACCUM"
        --steps 6
        --warmup 2
        --compile
        --grad-checkpoint
        --doc-lengths data/sample_pt/doc_lengths.json
        --out "$BENCH_JSONL"
    )
    if [ -n "$PRESET_OVERRIDES" ]; then
        read -r -a EXTRA_PO <<< "$PRESET_OVERRIDES"
        cmd+=("${EXTRA_PO[@]}")
    fi
    if [ -n "$profile_flag" ]; then
        cmd+=("$profile_flag")
    fi
    cmd+=("$@")

    echo ">> Executando: ${cmd[*]}"
    "${cmd[@]}" || echo "[bench_matrix] Execução encerrou com código $? (ex: OOM registrado)"
}

ALL_EXACT_OFF=(
    "--override" "fused_ce=false"
    "--override" "gdn_kernel_gate=false"
    "--override" "gdn_fla_conv=false"
    "--override" "fused_norm_gate=false"
    "--override" "checkpoint_policy=full"
    "--override" "checkpoint_frac=1.0"
)

ALL_EXACT_ON=(
    "--override" "fused_ce=true"
    "--override" "gdn_kernel_gate=true"
    "--override" "gdn_fla_conv=true"
    "--override" "fused_norm_gate=true"
)

for SEQ in $SEQS; do
    SEQ_LEN="${SEQ%x*}"
    MB="${SEQ#*x}"
    echo ""
    echo "======================================================================"
    echo ">> Iniciando matriz para seq_len=$SEQ_LEN, micro_batch=$MB"
    echo "======================================================================"

    # 1. Base (todas as alavancas desligadas, checkpoint full, com --profile)
    echo ""
    echo "--- 1. Base (todas as alavancas desligadas, com --profile) ---"
    run_bench "--profile" "$SEQ_LEN" "$MB" "${ALL_EXACT_OFF[@]}"

    # 2. Cada alavanca exata sozinha
    echo ""
    echo "--- 2a. Alavanca sozinha: fused_ce ---"
    run_bench "" "$SEQ_LEN" "$MB" \
        "--override" "fused_ce=true" \
        "--override" "gdn_kernel_gate=false" \
        "--override" "gdn_fla_conv=false" \
        "--override" "fused_norm_gate=false" \
        "--override" "checkpoint_policy=full" \
        "--override" "checkpoint_frac=1.0"

    echo ""
    echo "--- 2b. Alavanca sozinha: gdn_path ---"
    run_bench "" "$SEQ_LEN" "$MB" \
        "--override" "fused_ce=false" \
        "--override" "gdn_kernel_gate=true" \
        "--override" "gdn_fla_conv=true" \
        "--override" "fused_norm_gate=true" \
        "--override" "checkpoint_policy=full" \
        "--override" "checkpoint_frac=1.0"

    echo ""
    echo "--- 2c. Alavanca sozinha: ckpt_selective ---"
    run_bench "" "$SEQ_LEN" "$MB" \
        "--override" "fused_ce=false" \
        "--override" "gdn_kernel_gate=false" \
        "--override" "gdn_fla_conv=false" \
        "--override" "fused_norm_gate=false" \
        "--override" "checkpoint_policy=selective" \
        "--override" "checkpoint_frac=1.0"

    echo ""
    echo "--- 2d. Alavanca sozinha: ckpt_ffn ---"
    run_bench "" "$SEQ_LEN" "$MB" \
        "--override" "fused_ce=false" \
        "--override" "gdn_kernel_gate=false" \
        "--override" "gdn_fla_conv=false" \
        "--override" "fused_norm_gate=false" \
        "--override" "checkpoint_policy=ffn" \
        "--override" "checkpoint_frac=1.0"

    # 3. Todas as exatas juntas (com full checkpoint)
    echo ""
    echo "--- 3. Todas as exatas juntas (checkpoint full, frac 1.0) ---"
    run_bench "" "$SEQ_LEN" "$MB" \
        "${ALL_EXACT_ON[@]}" \
        "--override" "checkpoint_policy=full" \
        "--override" "checkpoint_frac=1.0"

    # 4. Todas as exatas x cada política de checkpoint x checkpoint_frac {1.0, 0.5, 0.25}
    echo ""
    echo "--- 4. Todas as exatas x políticas de checkpoint x checkpoint_frac ---"
    for POL in "selective" "ffn" "full"; do
        for FRAC in "1.0" "0.5" "0.25"; do
            # Full 1.0 já foi rodado no passo 3
            if [ "$POL" = "full" ] && [ "$FRAC" = "1.0" ]; then
                continue
            fi
            echo ">> Política: $POL, frac: $FRAC"
            run_bench "" "$SEQ_LEN" "$MB" \
                "${ALL_EXACT_ON[@]}" \
                "--override" "checkpoint_policy=$POL" \
                "--override" "checkpoint_frac=$FRAC"
        done
    done

    # 5. A melhor configuração exata x --profile (e x fp8 rowwise com FP8=1)
    echo ""
    echo "--- 5. Melhor configuração exata para seq_len=$SEQ_LEN, micro_batch=$MB ---"
    BEST_CMD=$(uv run python scripts/summarize_bench.py "$BENCH_JSONL" --best-exact "$SEQ_LEN" "$MB")

    if [ -n "$BEST_CMD" ]; then
        echo "Melhor exata encontrada: $BEST_CMD"
        read -r -a BEST_ARR <<< "$BEST_CMD"
        echo ">> Rodando profiling da melhor configuração exata..."
        run_bench "--profile" "$SEQ_LEN" "$MB" "${BEST_ARR[@]}"

        if [ "$FP8" = "1" ]; then
            echo ">> Rodando melhor configuração com FP8 rowwise..."
            run_bench "" "$SEQ_LEN" "$MB" --fp8 rowwise "${BEST_ARR[@]}"
        fi
    else
        echo "Aviso: Nenhuma configuração exata válida encontrada para seq $SEQ_LEN x $MB (possível OOM em todas)."
    fi
done

echo ""
echo "======================================================================"
echo ">> Resumo final da matriz de benchmark:"
echo "======================================================================"
uv run python scripts/summarize_bench.py "$BENCH_JSONL"
