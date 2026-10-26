"""Configuração do pytest: markers `cuda`/`fla` com skip automático e fixture `device`."""
import importlib.util

import pytest
import torch


def _has_fla() -> bool:
    # find_spec não importa o pacote; o import real fica em try/except
    # porque o FLA depende do Triton e pode falhar na importação.
    if importlib.util.find_spec("fla") is None:
        return False
    try:
        import fla  # noqa: F401
    except Exception:
        return False
    return True


def pytest_configure(config):
    config.addinivalue_line("markers", "cuda: exige GPU CUDA (pulado sem torch.cuda.is_available())")
    config.addinivalue_line("markers", "fla: exige GPU CUDA e flash-linear-attention importável")
    config.addinivalue_line("markers", "fp8: exige GPU CUDA com capacidade >= (8, 9)")


def pytest_collection_modifyitems(config, items):
    has_cuda = torch.cuda.is_available()
    has_fla = has_cuda and _has_fla()
    has_fp8 = has_cuda and torch.cuda.get_device_capability() >= (8, 9)
    skip_cuda = pytest.mark.skip(reason="sem GPU CUDA")
    skip_fla = pytest.mark.skip(reason="sem GPU CUDA ou sem flash-linear-attention")
    skip_fp8 = pytest.mark.skip(reason="sem GPU CUDA com capacidade >= (8, 9)")
    for item in items:
        if "cuda" in item.keywords and not has_cuda:
            item.add_marker(skip_cuda)
        if "fla" in item.keywords and not has_fla:
            item.add_marker(skip_fla)
        if "fp8" in item.keywords and not has_fp8:
            item.add_marker(skip_fp8)


@pytest.fixture
def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
