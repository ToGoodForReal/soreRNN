"""Scripts de campanha (bash), exercitados num mini-repo temporário com `uv`/`nvidia-smi` falsos."""
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None or os.name == "nt", reason="precisa de bash (Linux/WSL)")

# `uv` falso: `uv run python ...` vai para o python do teste; `python -m sore.train` vira um treino
# de mentira que escreve metrics.jsonl + last.pt (FAKE_STOP_AT=NOME:N para num passo N < total)
FAKE_UV = """#!/usr/bin/env bash
set -eo pipefail
case "$1" in
  sync) exit "${FAKE_SYNC_RC:-0}" ;;
  run) shift ;;
  *) exit 0 ;;
esac
[[ "$1" == "--no-sync" ]] && shift
if [[ "$1" == pytest ]]; then echo "$FAKE_PYTEST_OUT"; exit "${FAKE_PYTEST_RC:-0}"; fi
if [[ "$1 $2 $3" == "python -m sore.train" ]]; then
  for a in "$@"; do [[ "$a" == out_dir=* ]] && out=${a#out_dir=}; done
  name=$(basename "$out"); total=305
  [[ "${FAKE_STOP_AT:-}" == "$name:"* ]] && total=${FAKE_STOP_AT#*:}
  mkdir -p "$out"; : > "$out/last.pt"
  for ((i = 1; i <= total; i++)); do echo "{\\"step\\": $i, \\"lm_loss\\": 2.0}"; done >> "$out/metrics.jsonl"
  exit 0
fi
shift
if [[ -n "${FAKE_PYCHECK:-}" && "$1" == -c ]]; then  # a checagem torch/fla do setup_worker
  [[ "$FAKE_PYCHECK" == ok ]] && exit 0
  echo "torch.cuda.is_available() é False" >&2; exit 1
fi
exec "$REAL_PY" "$@"
"""


def _fake_bin(tmp_path: Path, smi_cuda: str = "13.4") -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for name, body in (("uv", FAKE_UV), ("nvidia-smi", f'#!/bin/sh\necho "| NVIDIA-SMI x   CUDA Version: {smi_cuda} |"\n')):
        p = bin_dir / name
        p.write_text(body)
        p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return bin_dir


def _env(tmp_path: Path, **extra: str) -> dict[str, str]:
    env = {**os.environ, "PATH": f"{_fake_bin(tmp_path)}{os.pathsep}{os.environ['PATH']}", "REAL_PY": sys.executable}
    env.update(extra)
    return env


def _mini_repo(tmp_path: Path, lever_run: bool = True) -> Path:
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    (root / "configs").mkdir()
    for rel in ("scripts/run_quality_gates.sh", "scripts/compare_runs.py", "scripts/setup_worker.sh",
                "configs/s150m_quality.yaml"):
        shutil.copy(REPO / rel, root / rel)
    gates = root / "scripts" / "run_quality_gates.sh"
    if lever_run:  # uma candidata de mentira na lista RUNS
        gates.write_text(gates.read_text().replace('  "ref_s1:seed=1"\n', '  "ref_s1:seed=1"\n  "lever:seed=0"\n'))
    return root


def _run(root: Path, script: str, env: dict[str, str], *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([BASH, f"scripts/{script}", *args], cwd=root, env=env, capture_output=True, text=True)


def test_gates_all_runs_finish_and_compare(tmp_path):
    root = _mini_repo(tmp_path)
    res = _run(root, "run_quality_gates.sh", _env(tmp_path))
    assert res.returncode == 0, res.stderr
    assert "| lever (exact) |" in res.stdout and "PASSA" in res.stdout


def test_gates_unfinished_candidate_fails_loudly(tmp_path):
    root = _mini_repo(tmp_path)
    res = _run(root, "run_quality_gates.sh", _env(tmp_path, FAKE_STOP_AT="lever:10"))
    assert res.returncode != 0
    assert "lever não terminou" in res.stderr and "PASSA" not in res.stdout


def test_gates_only_ignores_other_unfinished_runs(tmp_path):
    root = _mini_repo(tmp_path)
    res = _run(root, "run_quality_gates.sh", _env(tmp_path), "--only", "ref_s0")
    assert res.returncode == 0 and "comparação adiada" in res.stdout


def test_setup_check_rejects_old_cuda_driver(tmp_path):
    root = _mini_repo(tmp_path)
    env = _env(tmp_path)
    (tmp_path / "bin" / "nvidia-smi").write_text('#!/bin/sh\necho "CUDA Version: 12.4"\n')
    res = _run(root, "setup_worker.sh", env, "--check")
    assert res.returncode == 1 and "13.0" in res.stderr


def _setup_root(tmp_path: Path) -> Path:
    root = _mini_repo(tmp_path)
    (root / "data" / "sample_pt").mkdir(parents=True)
    (root / "data" / "sample_pt" / "sample_000.bin").write_bytes(b"")
    return root


def test_setup_fails_when_torch_has_no_cuda(tmp_path):
    root = _setup_root(tmp_path)
    res = _run(root, "setup_worker.sh", _env(tmp_path, FAKE_PYCHECK="nocuda"), "--check")
    assert res.returncode == 1 and "CUDA" in res.stderr


def test_setup_fails_when_cuda_tests_are_skipped(tmp_path):
    root = _setup_root(tmp_path)
    ok = _env(tmp_path, FAKE_PYCHECK="ok", FAKE_PYTEST_OUT="..\n2 passed in 1s")
    assert _run(root, "setup_worker.sh", ok).returncode == 0
    # o pytest sai com 0 quando tudo é pulado: o script precisa pegar isso pelo resumo
    skipped = _env(tmp_path, FAKE_PYCHECK="ok", FAKE_PYTEST_OUT="ss\n2 skipped in 1s")
    res = _run(root, "setup_worker.sh", skipped)
    assert res.returncode == 1 and "pulados" in res.stderr
    none_ran = _env(tmp_path, FAKE_PYCHECK="ok", FAKE_PYTEST_OUT="no tests ran in 0.1s")
    assert _run(root, "setup_worker.sh", none_ran).returncode == 1
