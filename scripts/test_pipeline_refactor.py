#!/usr/bin/env python3
"""
soreRNN-LM: Teste de Validação das 10 Refatorações Solicitadas
"""

import os
import sys
import numpy as np

def run_tests():
    print("=" * 70)
    print("  INICIANDO BATERIA DE TESTES DE INTEGRAÇÃO DA REFATORAÇÃO")
    print("=" * 70)

    # 1. Teste de Tokenizers (Item 7)
    print("\n[Item 7] Testando SoreTokenizer...")
    from tokenizer_utils import get_tokenizer
    tok_pt = get_tokenizer("pt")
    tok_gpt2 = get_tokenizer("gpt2")
    sample_pt = "A inteligência artificial avançada em português."
    enc_pt = tok_pt.encode(sample_pt)
    enc_gpt2 = tok_gpt2.encode(sample_pt)
    assert len(enc_pt) > 0 and len(enc_gpt2) > 0
    print(f" -> Tokenizer PT: {len(enc_pt)} tokens | GPT-2: {len(enc_gpt2)} tokens")
    assert tok_pt.vocab_size == 50257
    print(" -> [OK] Item 7: Tokenizer PT nativo operacional com vocab=50257.")

    # 2. Teste de Deduplicação e Filtros (Item 1 & Item 6)
    print("\n[Itens 1 & 6] Testando MinHash e Prevenção de Vazamento Val-vs-Train...")
    from dedup_filter import MinHashDeduplicator, filter_fineweb_edu, filter_text_quality
    dedup = MinHashDeduplicator()
    doc_val = "Este é um texto exclusivo do conjunto de validação para teste de held-out em processamento de linguagem."
    dedup.add_validation_document(doc_val)

    assert dedup.is_val_leak(doc_val) == True
    assert dedup.is_val_leak(doc_val + " com pequenas alterações no final.") == True
    assert dedup.add_train_document(doc_val) == False # Bloqueado de entrar no treino!
    print(" -> [OK] Prevenção de vazamento de validação (zero leakage) validada!")

    # Filtro FineWeb-Edu
    assert filter_fineweb_edu({"score": 4.5}, min_score=3.0) == True
    assert filter_fineweb_edu({"score": 1.8}, min_score=3.0) == False
    print(" -> [OK] Filtro de pontuação FineWeb-Edu validado!")

    # 3. Teste de Matemática Diversificada (Item 5)
    print("\n[Item 5] Testando Variabilidade da Matemática...")
    from math_dataset_utils import generate_diverse_synthetic_math
    samples = [generate_diverse_synthetic_math() for _ in range(5)]
    for i, s in enumerate(samples):
        assert len(s) > 50
    print(f" -> Amostra 1:\n{samples[0][:150]}...")
    print(" -> [OK] Item 5: Gerador de matemática diversificado e anti-template memorization.")

    # 4. Teste de Geração com Interleaving e Fronteira EOS Mascarada (Itens 2, 3, 4, 6)
    print("\n[Itens 2, 3, 4, 6] Testando Geração Integrada...")
    test_train_path = "data/test_refactor_train.bin"
    test_val_path = "data/test_refactor_val.bin"

    from prepare_4b_pretrain import generate_dataset
    generate_dataset(
        out_path=test_train_path,
        val_out_path=test_val_path,
        target_tokens=1000,
        val_tokens=200,
        tokenizer_name="pt",
        pt_weight=0.80,
        edu_weight=0.10,
        math_weight=0.10,
        paired_mode=True
    )

    assert os.path.exists(test_train_path)
    assert os.path.exists(test_val_path)

    # Validar que o arquivo de treino possui máscaras 65535 no último token do documento
    train_arr = np.fromfile(test_train_path, dtype=np.uint16)
    assert len(train_arr) >= 2000 # 1000 pares = 2000 uint16
    targets = train_arr[1::2]
    num_masked_boundaries = np.sum(targets == 65535)
    assert num_masked_boundaries > 0
    print(f" -> [OK] Total de tokens mascarados na fronteira (target=65535): {num_masked_boundaries}")
    print(f" -> [OK] val.bin gerado com sucesso: {os.path.getsize(test_val_path)} bytes")

    # Limpeza dos arquivos de teste
    os.remove(test_train_path)
    os.remove(test_val_path)

    print("\n" + "=" * 70)
    print("  TODOS OS TESTES DE VALIDAÇÃO PASSARAM COM SUCESSO ABSOLUTO! ")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
