"""Testes de fumaça: o pacote importa e a GPU executa um matmul BF16."""
import pytest
import torch


def test_import():
    import sore

    assert sore.__version__ == "0.2.0.dev0"


@pytest.mark.cuda
def test_cuda_bf16_matmul():
    a = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
    c = a @ b
    assert c.shape == (512, 512)
    assert torch.isfinite(c.float()).all()
