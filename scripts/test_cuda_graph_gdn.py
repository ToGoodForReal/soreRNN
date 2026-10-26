"""Experimento empírico de viabilidade de CUDA Graphs em Block GDN com FLA varlen."""
import torch
from sore.model.config import get_preset
from sore.model.block import Block

def test_compile_reduce_overhead():
    print("=== Teste 1: torch.compile(mode='reduce-overhead') ===")
    cfg = get_preset("tiny", gdn_backend="fla", gdn_kernel_gate=True, gdn_fla_conv=True, fused_norm_gate=True)
    device = torch.device("cuda")
    block = Block(cfg, 0, "gdn").to(device)
    compiled_block = torch.compile(block, mode="reduce-overhead")

    b, t, d = 1, 512, cfg.d_model
    x = torch.randn(b, t, d, device=device, dtype=torch.bfloat16)
    cu1 = torch.tensor([0, 256, 512], device=device, dtype=torch.long)
    cu1_cpu = cu1.cpu()

    print("Tentando forward 1 com cu_seqlens [0, 256, 512]...")
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out1 = compiled_block(x, cu_seqlens=cu1, cu_seqlens_cpu=cu1_cpu)
        print("Forward 1 concluído com sucesso!")
    except Exception as e:
        print(f"Erro no Forward 1: {type(e).__name__}: {e}")

    print("Tentando forward 2 com cu_seqlens diferente [0, 128, 512]...")
    cu2 = torch.tensor([0, 128, 512], device=device, dtype=torch.long)
    cu2_cpu = cu2.cpu()
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out2 = compiled_block(x, cu_seqlens=cu2, cu_seqlens_cpu=cu2_cpu)
        print("Forward 2 concluído com sucesso!")
    except Exception as e:
        print(f"Erro no Forward 2: {type(e).__name__}: {e}")

def test_manual_cuda_graph():
    print("\n=== Teste 2: torch.cuda.CUDAGraph() manual ===")
    cfg = get_preset("tiny", gdn_backend="fla", gdn_kernel_gate=True, gdn_fla_conv=True, fused_norm_gate=True)
    device = torch.device("cuda")
    block = Block(cfg, 0, "gdn").to(device)

    b, t, d = 1, 512, cfg.d_model
    static_x = torch.randn(b, t, d, device=device, dtype=torch.bfloat16)
    static_cu = torch.tensor([0, 256, 512], device=device, dtype=torch.long)
    static_cu_cpu = static_cu.cpu()

    # Warmup
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _ = block(static_x, cu_seqlens=static_cu, cu_seqlens_cpu=static_cu_cpu)
    torch.cuda.current_stream().wait_stream(s)

    # Capture
    g = torch.cuda.CUDAGraph()
    print("Iniciando captura de CUDA graph com static_cu = [0, 256, 512]...")
    try:
        with torch.cuda.graph(g, stream=s):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                static_out = block(static_x, cu_seqlens=static_cu, cu_seqlens_cpu=static_cu_cpu)
        print("Captura do CUDA graph bem sucedida!")
    except Exception as e:
        print(f"Erro durante captura: {type(e).__name__}: {e}")
        return

    # Replay com mesmos dados
    print("Replay com mesma entrada...")
    try:
        g.replay()
        torch.cuda.synchronize()
        print("Replay com mesma entrada OK!")
    except Exception as e:
        print(f"Erro no replay: {type(e).__name__}: {e}")

    # E agora com cu_seqlens dinâmico / atualizado in-place:
    print("\nAtualizando static_cu in-place para [0, 128, 512] (novo particionamento)...")
    static_cu.copy_(torch.tensor([0, 128, 512], device=device, dtype=torch.long))
    try:
        g.replay()
        torch.cuda.synchronize()
        replayed_out = static_out.clone()
        print("Replay executou sem erro no driver CUDA.")
        
        # Testar corretude numérica contra execução real
        with torch.autocast("cuda", dtype=torch.bfloat16):
            expected_out = block(static_x, cu_seqlens=static_cu, cu_seqlens_cpu=torch.tensor([0, 128, 512]))
        diff = (replayed_out - expected_out).abs().max().item()
        print(f"Diferença máxima absoluta entre Replay e Saída Correta: {diff:.6f}")
        if diff > 1e-3:
            print("FALHA DE CORRETUDE: O replay do CUDA Graph NÃO produziu a computação correta para o novo cu_seqlens!")
            print("Causa fundamental: no CUDA graph, os nós dos kernels Triton são gravados com os grids e ponteiros de chunk fixados na captura.")
        else:
            print("Resultado bateu!")
    except Exception as e:
        print(f"Erro no replay com cu_seqlens alterado: {type(e).__name__}: {e}")

def test_make_graphed_callables():
    print("\n=== Teste 3: torch.cuda.make_graphed_callables ===")
    cfg = get_preset("tiny", gdn_backend="fla", gdn_kernel_gate=True, gdn_fla_conv=True, fused_norm_gate=True)
    device = torch.device("cuda")
    block = Block(cfg, 0, "gdn").to(device)

    b, t, d = 1, 512, cfg.d_model
    x1 = torch.randn(b, t, d, device=device, dtype=torch.bfloat16)
    cu1 = torch.tensor([0, 256, 512], device=device, dtype=torch.long)
    cu1_cpu = cu1.cpu()

    class AutocastBlock(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x, cu_seqlens, cu_seqlens_cpu):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return self.m(x, cu_seqlens=cu_seqlens, cu_seqlens_cpu=cu_seqlens_cpu)

    wrapped = AutocastBlock(block)

    try:
        graphed_block = torch.cuda.make_graphed_callables(
            wrapped,
            sample_args=(x1, cu1, cu1_cpu),
        )
        print("make_graphed_callables concluído!")
        
        # Chamada 1 (mesmo cu_seqlens)
        o1 = graphed_block(x1, cu1, cu1_cpu)
        print("Execução 1 com sample_args OK!")
        
        # Chamada 2 (cu_seqlens com outra contagem de documentos ou valores)
        cu2 = torch.tensor([0, 128, 512], device=device, dtype=torch.long)
        cu2_cpu = cu2.cpu()
        o2 = graphed_block(x1, cu2, cu2_cpu)
        print("Chamada 2 com novo cu_seqlens OK!")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            expected = block(x1, cu_seqlens=cu2, cu_seqlens_cpu=cu2_cpu)
        diff = (o2 - expected).abs().max().item()
        print(f"Diferença máxima Chamada 2 vs esperado: {diff:.6f}")
    except Exception as e:
        print(f"Erro em make_graphed_callables: {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_compile_reduce_overhead()
    test_manual_cuda_graph()
    test_make_graphed_callables()
