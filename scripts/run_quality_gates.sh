#!/usr/bin/env bash
# Campanha de qualidade das alavancas de velocidade (configs/s150m_quality.yaml, ~20M tokens cada).
#
# Uso: bash scripts/run_quality_gates.sh [--only NOME]
#   --only NOME   roda só essa corrida (as demais são ignoradas; a comparação final usa as que existirem)
# Variáveis de ambiente:
#   EXTRA_SET     overrides extras para TODAS as corridas, separadas por `;` (ensaios curtos), ex.:
#                 EXTRA_SET="total_tokens=2e5;warmup_steps=1;stages=[{until_tokens: 2e5, seq_len: 2048, micro_batch: 2}]"
#   QUALITY_DIR   raiz das corridas (padrão runs/quality)
#
# Cada corrida grava em $QUALITY_DIR/NOME e é pulada se já terminou (last.pt + último passo no
# metrics.jsonl); se parou no meio, é retomada de last.pt. No fim chama scripts/compare_runs.py.
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=configs/s150m_quality.yaml
QUALITY_DIR=${QUALITY_DIR:-runs/quality}

# ----------------------------------------------------------------------------------------------
# PONTO DE EXTENSÃO 1 (cada task de alavanca mexe aqui): ALL_OFF = overrides que DESLIGAM todas as
# alavancas da lista, um `KEY=VALUE` por elemento. Vale para toda corrida (refs e candidatas), então
# a candidata difere da referência só pela alavanca que a sua linha liga. Começa vazio.
# Ex. de uma task de alavanca: ALL_OFF+=("model_overrides.fused_ce=false")
ALL_OFF=(
)

# PONTO DE EXTENSÃO 2: as corridas, `NOME:override;override;...` (separador `;`, pois valores como
# stages=[{a: 1, b: 2}] têm vírgula e espaço). A alavanca liga a sua flag na própria linha, que vem
# depois de ALL_OFF e por isso vence. Candidata exata compara com |gap| <= max(ruído, 0,5%).
# Ex.: "fused_ce:model_overrides.fused_ce=true"
RUNS=(
  "ref_s0:seed=0"
  "ref_s1:seed=1"
)

# PONTO DE EXTENSÃO 3: nomes de candidatas COM perda (gap <= 1%); só vale se a exata ficar < 15k tok/s.
LOSSY=(
)
# ----------------------------------------------------------------------------------------------

ONLY=""
case "${1:-}" in
  "") ;;
  --only) ONLY=${2:?"--only precisa de um NOME"} ;;
  *) echo "uso: $0 [--only NOME]" >&2; exit 2 ;;
esac

# `;` separa as overrides; um elemento vazio (ex. EXTRA_SET vazio) não vira --set
split_semicolon() { local IFS=';'; read -r -a SPLIT <<< "${1:-}"; }
split_semicolon "${EXTRA_SET:-}"
EXTRA=("${SPLIT[@]}")

# total de passos como o TrainConfig calcula, para saber se a corrida terminou
total_steps() { uv run python -c '
import sys
from pathlib import Path
from sore.train.config import load_train_config, parse_overrides
print(load_train_config(Path(sys.argv[1]), parse_overrides(sys.argv[2:])).total_steps)' "$CONFIG" "$@"; }

finished() { # $1 = out_dir, $2 = total_steps
  [[ -f "$1/last.pt" && -s "$1/metrics.jsonl" ]] || return 1
  [[ "$(tail -n 1 "$1/metrics.jsonl" | uv run python -c 'import json,sys; print(json.loads(sys.stdin.read())["step"])')" == "$2" ]]
}

# monta NAME, OUT, ARGS (os KEY=VALUE) e STEPS de uma entrada de RUNS. Ordem: out_dir, ALL_OFF, a
# linha, EXTRA_SET (a última vence: EXTRA_SET manda e a linha da alavanca desfaz o ALL_OFF dela).
prepare() {
  local entry=$1 s
  NAME=${entry%%:*}; OUT="$QUALITY_DIR/$NAME"
  split_semicolon "${entry#*:}"
  ARGS=()
  for s in "out_dir=$OUT" "${ALL_OFF[@]}" "${SPLIT[@]}" "${EXTRA[@]}"; do
    [[ -n "$s" ]] && ARGS+=("$s")
  done
  STEPS=$(total_steps "${ARGS[@]}")
}

ran=()
for entry in "${RUNS[@]}"; do
  prepare "$entry"
  [[ -z "$ONLY" || "$ONLY" == "$NAME" ]] || continue
  sets=(); for s in "${ARGS[@]}"; do sets+=(--set "$s"); done
  if finished "$OUT" "$STEPS"; then
    echo "[quality] $NAME: já terminou ($STEPS passos), pulando"
  elif [[ -f "$OUT/last.pt" ]]; then
    echo "[quality] $NAME: retomando de $OUT/last.pt"
    uv run python -m sore.train --config "$CONFIG" "${sets[@]}" --resume "$OUT/last.pt"
  else
    echo "[quality] $NAME: treinando ($STEPS passos)"
    uv run python -m sore.train --config "$CONFIG" "${sets[@]}"
  fi
  ran+=("$NAME")
done
[[ ${#ran[@]} -gt 0 ]] || { echo "[quality] nenhuma corrida chamada '$ONLY' em RUNS" >&2; exit 2; }

# comparação: só com as duas referências terminadas (parciais dariam um ruído enganoso);
# candidatas = as demais corridas terminadas
cands=(); refs_done=0
for entry in "${RUNS[@]}"; do
  prepare "$entry"
  finished "$OUT" "$STEPS" || continue
  case "$NAME" in
    ref_s0|ref_s1) refs_done=$((refs_done + 1)) ;;
    *) cands+=("$OUT") ;;
  esac
done
if [[ $refs_done -lt 2 ]]; then
  echo "[quality] ref_s0 e ref_s1 ainda não terminaram: comparação adiada"
  exit 0
fi
lossy=(); for name in "${LOSSY[@]}"; do lossy+=(--lossy "$name"); done
uv run python scripts/compare_runs.py --ref "$QUALITY_DIR/ref_s0" --ref2 "$QUALITY_DIR/ref_s1" \
  "${lossy[@]}" "${cands[@]}"
