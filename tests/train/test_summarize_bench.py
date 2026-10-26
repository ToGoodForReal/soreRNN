"""Testes para o sumarizador de benchmarks (scripts/summarize_bench.py)."""
from pathlib import Path
import json

import pytest

from scripts.summarize_bench import format_table, summarize_bench


def test_summarize_bench_orders_and_flags(tmp_path: Path):
    """Testa que summarize_bench ordena por tok/s desc, coloca OOM no fim e marca 15K e 11.8K."""
    bench_file = tmp_path / "bench.jsonl"
    
    # 3 entradas: uma acima de 15k, uma entre 11.8k e 15k, e uma OOM
    entry_mid = {
        "preset": "b0_8",
        "overrides": {"fused_ce": True},
        "seq_len": 4096,
        "micro_batch": 2,
        "grad_accum": 8,
        "tok_s": 12500.0,
        "mfu": 0.415,
        "mem_reserved_gb": 18.5,
        "mem_gb": 16.2,
    }
    entry_high = {
        "preset": "b0_8",
        "overrides": {"fused_ce": True, "gdn_path": "native"},
        "seq_len": 4096,
        "micro_batch": 2,
        "grad_accum": 8,
        "tok_s": 15800.0,
        "mfu": 0.524,
        "mem_reserved_gb": 19.1,
        "mem_gb": 17.0,
    }
    entry_oom = {
        "oom": True,
        "preset": "b0_8",
        "overrides": {"compile": False},
        "seq_len": 8192,
        "micro_batch": 2,
        "peak_tflops": 165.0,
    }
    
    # Grava fora de ordem propositalmente (mid, oom, high)
    with bench_file.open("w") as f:
        f.write(json.dumps(entry_mid) + "\n")
        f.write(json.dumps(entry_oom) + "\n")
        f.write(json.dumps(entry_high) + "\n")

    table = summarize_bench(bench_file)
    lines = [line.strip() for line in table.splitlines() if line.strip() and not line.startswith("|-")]
    
    # Header + separador + 3 linhas de dados = pelo menos 4 linhas
    # Linhas de dados (após header)
    data_lines = [l for l in lines if not l.startswith("| Config") and not l.startswith("|---")]
    assert len(data_lines) == 3

    # Ordem: high (15.8k) primeiro, mid (12.5k) segundo, OOM por último
    assert "15,800" in data_lines[0] or "15800" in data_lines[0]
    assert "12,500" in data_lines[1] or "12500" in data_lines[1]
    assert "OOM" in data_lines[2]

    # Flags: meta 15K e piso 11.8K marcados
    assert ">= 15k" in data_lines[0].lower() or "meta" in data_lines[0].lower() or ">= 15.000" in data_lines[0]
    assert ">= 11.8k" in data_lines[1].lower() or "piso" in data_lines[1].lower() or ">= 11.800" in data_lines[1]
    assert "oom" in data_lines[2].lower()


def test_summarize_bench_cli(tmp_path: Path, capsys):
    """Testa execução via CLI do summarize_bench e identificação de throughput abaixo do piso."""
    from scripts.summarize_bench import main

    bench_file = tmp_path / "bench.jsonl"
    entry_low = {
        "preset": "b0_8",
        "seq_len": 4096,
        "micro_batch": 2,
        "tok_s": 8000.0,
        "mfu": 0.26,
    }
    with bench_file.open("w") as f:
        f.write(json.dumps(entry_low) + "\n")

    ret = main([str(bench_file)])
    assert ret == 0
    captured = capsys.readouterr()
    assert "< 11.800" in captured.out
    assert "8,000" in captured.out or "8000" in captured.out


def test_summarize_bench_cli_missing_file():
    """Testa que arquivo inexistente retorna código de erro 1."""
    from scripts.summarize_bench import main

    ret = main(["non_existent_file.jsonl"])
    assert ret == 1


def test_bench_write_oom_record(tmp_path: Path):
    """Testa que write_oom_record de scripts/bench.py grava a linha esperada no formato R9."""
    from scripts.bench import write_oom_record

    out_file = tmp_path / "oom_bench.jsonl"
    write_oom_record(
        out_path=out_file,
        preset="b0_8",
        overrides={"checkpoint_policy": "full"},
        seq_len=8192,
        micro_batch=2,
        peak_tflops=165.0,
    )
    assert out_file.exists()
    line = json.loads(out_file.read_text().strip())
    assert line["oom"] is True
    assert line["preset"] == "b0_8"
    assert line["overrides"] == {"checkpoint_policy": "full"}
    assert line["seq_len"] == 8192
    assert line["micro_batch"] == 2
    assert line["peak_tflops"] == 165.0


def test_find_best_exact(tmp_path: Path):
    """Testa busca da melhor configuração exata (maior tok/s, sem OOM, sem FP8)."""
    from scripts.summarize_bench import find_best_exact

    bench_file = tmp_path / "bench.jsonl"
    e1 = {"seq_len": 4096, "micro_batch": 2, "tok_s": 12000.0, "fp8": "off", "overrides": {"a": 1}}
    e2 = {"seq_len": 4096, "micro_batch": 2, "tok_s": 14000.0, "fp8": "off", "overrides": {"b": 2}}
    e_fp8 = {"seq_len": 4096, "micro_batch": 2, "tok_s": 17000.0, "fp8": "rowwise", "overrides": {"b": 2}}
    e_oom = {"seq_len": 4096, "micro_batch": 2, "oom": True}
    e_diff_seq = {"seq_len": 8192, "micro_batch": 1, "tok_s": 15000.0, "fp8": "off"}

    with bench_file.open("w") as f:
        for entry in [e1, e2, e_fp8, e_oom, e_diff_seq]:
            f.write(json.dumps(entry) + "\n")

    best = find_best_exact(bench_file, 4096, 2)
    assert best is not None
    assert best["tok_s"] == 14000.0
    assert best["overrides"] == {"b": 2}

    best_8k = find_best_exact(bench_file, 8192, 1)
    assert best_8k is not None
    assert best_8k["tok_s"] == 15000.0


def test_summarize_bench_cli_best_exact(tmp_path: Path, capsys):
    """Testa que summarize_bench.py --best-exact imprime os argumentos --override da melhor corrida."""
    from scripts.summarize_bench import main

    bench_file = tmp_path / "bench.jsonl"
    e1 = {"seq_len": 4096, "micro_batch": 2, "tok_s": 12000.0, "fp8": "off", "overrides": {"fused_ce": False}}
    e2 = {"seq_len": 4096, "micro_batch": 2, "tok_s": 16000.0, "fp8": "off", "overrides": {"fused_ce": True, "checkpoint_policy": "selective"}}
    with bench_file.open("w") as f:
        f.write(json.dumps(e1) + "\n")
        f.write(json.dumps(e2) + "\n")

    ret = main([str(bench_file), "--best-exact", "4096", "2"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "--override checkpoint_policy=selective --override fused_ce=True" == captured.out.strip()



def test_b0_8_fp8_ab_configs_load():
    """Valida que os arquivos de configuração de A/B de FP8 são válidos para o TrainConfig."""
    import yaml
    from sore.train.config import TrainConfig

    for cfg_name in ["configs/b0_8_fp8_ab_bf16.yaml", "configs/b0_8_fp8_ab_fp8.yaml"]:
        cfg_path = Path(cfg_name)
        assert cfg_path.exists()
        raw = yaml.safe_load(cfg_path.read_text())
        cfg = TrainConfig(**raw)
        assert cfg.model_preset == "b0_8"
        assert cfg.total_tokens == int(3e8)
        assert cfg.stages[0].seq_len == 4096
        assert cfg.stages[0].micro_batch == 2



