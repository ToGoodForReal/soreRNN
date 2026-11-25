#!/usr/bin/env bash
# Prepara a máquina de treino (Linux nativo ou WSL; hoje o desktop com a RTX 3050 8GB, docs/RUNBOOK.md §1).
#
# Uso: bash scripts/setup_worker.sh [--check]
#   (sem flag)  confere a GPU, instala o uv se faltar, `uv sync` e os testes @cuda/@fla
#   --check     só valida (GPU, uv, ambiente sincronizado, torch com CUDA, fla); não instala nada
#
# Pressupõe o repositório já presente (git bundle ou push da branch; ver docs/RUNBOOK.md §1). A amostra PT
# (data/sample_pt/) só gera um aviso se faltar: ela é feita depois, com o ambiente pronto (docs/RUNBOOK.md §2).
# Termina com 0 se tudo está pronto.
set -euo pipefail
cd "$(dirname "$0")/.."

CHECK=0
case "${1:-}" in
  "") ;;
  --check) CHECK=1 ;;
  *) echo "uso: $0 [--check]" >&2; exit 2 ;;
esac
command -v uv >/dev/null || export PATH="$HOME/.local/bin:$PATH"  # onde o instalador do uv coloca o binário

fail() { echo "[setup] ERRO: $*" >&2; exit 1; }

# 1. GPU: o driver precisa expor CUDA >= 13.0 (os wheels torch 2.14.1 do uv.lock são cu130)
command -v nvidia-smi >/dev/null || fail "nvidia-smi não encontrado (driver NVIDIA ausente; no WSL, driver do Windows)"
smi=$(nvidia-smi)  # em variável: com pipefail, um `head` fechando o pipe daria SIGPIPE
# drivers novos escrevem "CUDA UMD Version: 13.4" no lugar de "CUDA Version: 13.4"
cuda=$(sed -n 's/.*CUDA \(UMD \)\{0,1\}Version: *\([0-9]*\.[0-9]*\).*/\2/p' <<< "$smi" | head -n 1)
[[ -n "$cuda" ]] || fail "não consegui ler 'CUDA Version' da saída do nvidia-smi"
major=${cuda%%.*}
[[ "$major" -ge 13 ]] || fail "driver com CUDA $cuda; é preciso >= 13.0 (atualize o driver NVIDIA)"
echo "[setup] GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n 1) | CUDA $cuda"

# 2. uv
if ! command -v uv >/dev/null; then
  [[ $CHECK -eq 0 ]] || fail "uv não encontrado (rode sem --check para instalar)"
  echo "[setup] instalando o uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "[setup] $(uv --version)"

# 3. dados e dependências
if [[ ! -f data/sample_pt/sample_000.bin ]]; then
  echo "[setup] aviso: data/sample_pt/sample_000.bin ausente; gere com scripts/make_text_sample.py (docs/RUNBOOK.md §2)" >&2
fi
if [[ $CHECK -eq 1 ]]; then
  uv sync --frozen --check >/dev/null 2>&1 || fail "ambiente desatualizado em relação ao uv.lock (rode sem --check)"
else
  uv sync --frozen
fi

# 4. o torch enxerga a GPU e o FLA importa. Explícito porque os testes @cuda/@fla são PULADOS (e o
# pytest sai com 0) quando falta CUDA ou FLA, justamente o que este script existe para pegar
uv run --no-sync python -c '
import sys
import torch
if not torch.cuda.is_available():
    sys.exit("torch.cuda.is_available() é False: driver/CUDA quebrado para o torch cu130")
try:
    import fla  # noqa: F401
except ImportError as e:
    sys.exit(f"flash-linear-attention (fla) não importa: {e}")
print(f"[setup] torch {torch.__version__} vê {torch.cuda.get_device_name()}; fla importa")
' || fail "torch sem CUDA ou fla ausente (mensagem acima)"

if [[ $CHECK -eq 1 ]]; then
  echo "[setup] --check: GPU, uv, ambiente, CUDA no torch e fla OK"
  exit 0
fi

# 5. os testes @cuda/@fla têm de EXECUTAR: nenhum pulado, pelo menos um rodando
rc=0; out=$(uv run pytest -q -rs -m "cuda or fla" 2>&1) || rc=$?
echo "$out"
[[ $rc -eq 0 ]] || fail "pytest -m 'cuda or fla' falhou (exit $rc)"
if grep -Eq '[0-9]+ skipped' <<< "$out"; then
  fail "testes @cuda/@fla foram pulados (ver linhas SKIPPED acima); eles têm de executar"
fi
grep -Eq '[0-9]+ passed' <<< "$out" || fail "nenhum teste @cuda/@fla executou"
echo "[setup] pronto"
