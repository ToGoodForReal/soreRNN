#!/usr/bin/env bash
# Protocolo de dia 1 na RTX 4090: matriz de benchmark e perfilamento do preset b0_8.
#
# Variáveis de ambiente:
#   OUT_DIR           diretório de saída (padrão: runs/day1)
#   PRESET_OVERRIDES  overrides extras passados ao bench.py (padrão vazio; ex: "--override n_layers=4" no ensaio)
#   SEQS              pares seq_len x micro_batch (padrão: "8192x1 4096x2")
#   FP8_FLAGS         flags extras de FP8 (padrão vazio; "--fp8-emulate" no ensaio na 3050)
#
# Matriz por seq:
#   1. base (todas as alavancas desligadas, full, com --profile);
#   2. cada alavanca exata sozinha (fused_ce, gdn_path, ckpt_selective, ckpt_ffn);
#   3. todas as exatas juntas;
#   4. todas as exatas x cada política de checkpoint x checkpoint_frac {1.0, 0.5, 0.25};
#   5. a melhor configuração x --profile e x fp8 {tensorwise, rowwise} (com $FP8_FLAGS).
#
# No fim, imprime a tabela sumarizada via scripts/summarize_bench.py.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"

OUT_DIR=${OUT_DIR:-runs/day1}
PRESET_OVERRIDES=${PRESET_OVERRIDES:-}
SEQS=${SEQS:-"8192x1 4096x2"}
FP8_FLAGS=${FP8_FLAGS:-}

mkdir -p "$OUT_DIR"
BENCH_JSONL="$OUT_DIR/bench.jsonl"

echo "=== soreRNN — Protocolo de Dia 1 na RTX 4090 ==="
echo "Saída: $BENCH_JSONL"
echo "Sequências: $SEQS"
echo "Overrides extras: ${PRESET_OVERRIDES:-nenhum}"
echo "Flags FP8: ${FP8_FLAGS:-nenhuma}"
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
        --preset b0_8
        --seq-len "$seq_len"
        --micro-batch "$mb"
        --grad-accum 8
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
    "${cmd[@]}" || echo "[day1_bench] Execução encerrou com código $? (ex: OOM registrado)"
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

    # 5. A melhor configuração exata x --profile e x fp8 {tensorwise, rowwise}
    echo ""
    echo "--- 5. Melhor configuração exata para seq_len=$SEQ_LEN, micro_batch=$MB ---"
    BEST_CMD=$(python3 scripts/summarize_bench.py "$BENCH_JSONL" --best-exact "$SEQ_LEN" "$MB")

    if [ -n "$BEST_CMD" ]; then
        echo "Melhor exata encontrada: $BEST_CMD"
        read -r -a BEST_ARR <<< "$BEST_CMD"
        fp8_extra=()
        if [ -n "$FP8_FLAGS" ]; then
            read -r -a fp8_extra <<< "$FP8_FLAGS"
        fi

        echo ">> Rodando profiling da melhor configuração exata..."
        run_bench "--profile" "$SEQ_LEN" "$MB" "${BEST_ARR[@]}"

        echo ">> Rodando melhor configuração com FP8 tensorwise..."
        run_bench "" "$SEQ_LEN" "$MB" --fp8 tensorwise "${fp8_extra[@]}" "${BEST_ARR[@]}"

        echo ">> Rodando melhor configuração com FP8 rowwise..."
        run_bench "" "$SEQ_LEN" "$MB" --fp8 rowwise "${fp8_extra[@]}" "${BEST_ARR[@]}"
    else
        echo "Aviso: Nenhuma configuração exata válida encontrada para seq $SEQ_LEN x $MB (possível OOM em todas)."
    fi
done

echo ""
echo "======================================================================"
echo ">> Resumo final da matriz de benchmark:"
echo "======================================================================"
uv run python scripts/summarize_bench.py "$BENCH_JSONL"
