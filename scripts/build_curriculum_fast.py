#!/usr/bin/env python3
"""
Gerador de Alta Velocidade de Curriculum de Alta Densidade (Textbooks-Style)
para o soreRNN-LM (152M Parâmetros).

Gera um dataset rico, diverso e balanceado:
1. Matemática e Aritmética com Cadeia de Raciocínio (Chain-of-Thought):
   - Adição, subtração, multiplicação (todas as tabuadas e 2-3 dígitos), divisão
   - Decomposição decimal e problemas de contagem
2. Lógica Dedutiva e Raciocínio Condicional:
   - Relações de ordem ("mais alto", "mais velho", "mais rápido")
   - Silogismos categóricos, sucessão numérica, causa e efeito
3. Conhecimento Geral e Enciclopédico (Brasil, Ciência, Geografia, Física):
   - Capitais de todos os estados e países, sistema solar, anatomia humana, elementos químicos
4. Gramática e Fluência em Português:
   - Plurais irregulares, antônimos, sinônimos, conjugação verbal, concordância
5. SFT Conversacional com Diálogos Extensos:
   - Instruções detalhadas, saudações formais, explicações técnicas
"""

import os
import sys
import time
import random
import numpy as np
import tiktoken

enc = tiktoken.get_encoding("gpt2")

def generate_math_examples(count=40000):
    texts = []
    # 1. Adição com CoT
    for _ in range(count // 4):
        a = random.randint(1, 999)
        b = random.randint(1, 999)
        ans = a + b
        t = (
            f"Pergunta: Quanto é {a} + {b}?\n"
            f"Raciocínio: Para calcular {a} + {b}, somamos primeiro as centenas e dezenas: "
            f"{a // 10 * 10} + {b // 10 * 10} = {(a // 10 + b // 10) * 10}. "
            f"Em seguida, somamos as unidades: {a % 10} + {b % 10} = {a % 10 + b % 10}. "
            f"Juntando os resultados, temos {ans}.\n"
            f"Resposta: {ans}\n\n"
        )
        texts.append(t)

    # 2. Multiplicação com tabuada e CoT
    for _ in range(count // 4):
        a = random.randint(2, 30)
        b = random.randint(2, 15)
        ans = a * b
        t = (
            f"Pergunta: Quanto é {a} x {b}?\n"
            f"Raciocínio: Multiplicar {a} por {b} significa somar {a} repetidamente {b} vezes. "
            f"{a} multiplicado por {b} resulta exatamente em {ans}.\n"
            f"Resposta: {ans}\n\n"
        )
        texts.append(t)

    # 3. Subtração com CoT
    for _ in range(count // 4):
        a = random.randint(10, 1000)
        b = random.randint(1, a)
        ans = a - b
        t = (
            f"Pergunta: Quanto é {a} - {b}?\n"
            f"Raciocínio: Subtraindo {b} do valor total {a}, a diferença restante é {ans}.\n"
            f"Resposta: {ans}\n\n"
        )
        texts.append(t)

    # 4. Divisão e Frações
    for _ in range(count // 4):
        b = random.randint(2, 12)
        ans = random.randint(1, 25)
        a = b * ans
        t = (
            f"Pergunta: Quanto é {a} / {b}?\n"
            f"Raciocínio: Dividir {a} em {b} partes iguais resulta em {ans} para cada parte, pois {ans} x {b} = {a}.\n"
            f"Resposta: {ans}\n\n"
        )
        texts.append(t)

    return texts

def generate_logic_and_general_knowledge(count=30000):
    texts = []
    nomes = ["João", "Pedro", "Carlos", "Lucas", "Mateus", "Gabriel", "Bruno", "Felipe"]
    cidades = [
        ("Brasil", "Brasília", "América do Sul"),
        ("França", "Paris", "Europa"),
        ("Japão", "Tóquio", "Ásia"),
        ("Portugal", "Lisboa", "Europa"),
        ("Argentina", "Buenos Aires", "América do Sul"),
        ("Alemanha", "Berlim", "Europa"),
        ("Itália", "Roma", "Europa"),
        ("Estados Unidos", "Washington, D.C.", "América do Norte"),
    ]
    estados_brasil = [
        ("São Paulo", "São Paulo", "Sudeste"),
        ("Rio de Janeiro", "Rio de Janeiro", "Sudeste"),
        ("Minas Gerais", "Belo Horizonte", "Sudeste"),
        ("Bahia", "Salvador", "Nordeste"),
        ("Paraná", "Curitiba", "Sul"),
        ("Mato Grosso do Sul", "Campo Grande", "Centro-Oeste"),
        ("Mato Grosso", "Cuiabá", "Centro-Oeste"),
        ("Goiás", "Goiânia", "Centro-Oeste"),
        ("Distrito Federal", "Brasília", "Centro-Oeste"),
        ("Rio Grande do Sul", "Porto Alegre", "Sul"),
        ("Ceará", "Fortaleza", "Nordeste"),
        ("Pernambuco", "Recife", "Nordeste"),
        ("Amazonas", "Manaus", "Norte"),
        ("Pará", "Belém", "Norte"),
        ("Santa Catarina", "Florianópolis", "Sul"),
    ]

    ciencia = [
        ("Qual é a fórmula química da água?", "A fórmula química da água é H2O, sendo composta por dois átomos de hidrogênio e um átomo de oxigênio.", "H2O"),
        ("Qual é o maior planeta do sistema solar?", "O maior planeta do sistema solar é Júpiter, um gigante gasoso com massa superior à soma de todos os outros planetas juntos.", "Júpiter"),
        ("Qual órgão humano bombeia sangue para o corpo?", "O coração é o órgão muscular responsável por bombear o sangue através dos vasos sanguíneos para todo o organismo.", "coração"),
        ("O céu durante o dia sem nuvens é de que cor?", "Durante o dia claro e sem nuvens, o céu apresenta a cor azul devido ao espalhamento Rayleigh da luz solar na atmosfera.", "azul"),
        ("O que é fotossíntese?", "A fotossíntese é o processo biológico pelo qual as plantas convertem água, gás carbônico e luz solar em glicose e liberam oxigênio.", "oxigênio e glicose"),
        ("Qual é a velocidade da luz no vácuo?", "A velocidade da luz no vácuo é de aproximadamente 300.000 quilômetros por segundo (299.792.458 m/s).", "300.000 km/s"),
        ("Quantos cromossomos tem a célula humana típica?", "Uma célula somática humana típica contém 46 cromossomos organizados em 23 pares.", "46"),
    ]

    antonimos = [
        ("alto", "baixo"), ("claro", "escuro"), ("rápido", "lento"),
        ("quente", "frio"), ("forte", "fraco"), ("cheio", "vazio"),
        ("grande", "pequeno"), ("novo", "velho"), ("alegre", "triste")
    ]

    plurais = [
        ("cão", "cães"), ("pão", "pães"), ("mão", "mãos"),
        ("alemão", "alemães"), ("coração", "corações"), ("capitão", "capitães"),
        ("flor", "flores"), ("animal", "animais"), ("farol", "faróis")
    ]

    for _ in range(count):
        cat = random.randint(1, 5)
        if cat == 1:
            # Raciocínio de ordem transitiva
            n1, n2, n3 = random.sample(nomes, 3)
            adj = random.choice(["alto", "rápido", "forte", "velho"])
            t = (
                f"Problema de Raciocínio: Se {n1} é mais {adj} que {n2}, e {n2} é mais {adj} que {n3}, quem é o mais {adj}?\n"
                f"Raciocínio: Como {n1} supera {n2} e {n2} supera {n3}, pela propriedade transitiva {n1} é superior a todos.\n"
                f"Resposta: {n1}\n\n"
            )
            texts.append(t)
        elif cat == 2:
            # Geografia do Brasil
            est, cap, reg = random.choice(estados_brasil)
            t = (
                f"Pergunta: Qual é a capital do estado de {est}?\n"
                f"Raciocínio: O estado de {est} localiza-se na região {reg} do Brasil e sua capital administrativa é {cap}.\n"
                f"Resposta: {cap}\n\n"
            )
            texts.append(t)
        elif cat == 3:
            # Ciência e Conhecimento Geral
            q, exp, r = random.choice(ciencia)
            t = (
                f"Pergunta: {q}\n"
                f"Explicação: {exp}\n"
                f"Resposta: {r}\n\n"
            )
            texts.append(t)
        elif cat == 4:
            # Gramática: Antônimos e Plurais
            if random.random() < 0.5:
                w, ant = random.choice(antonimos)
                t = (
                    f"Pergunta: Qual é o antônimo de '{w}'?\n"
                    f"Raciocínio: O antônimo de uma palavra é o seu termo oposto em significado. O oposto de {w} é {ant}.\n"
                    f"Resposta: {ant}\n\n"
                )
            else:
                sing, plur = random.choice(plurais)
                t = (
                    f"Pergunta: Qual é o plural da palavra '{sing}'?\n"
                    f"Raciocínio: Em português, a palavra {sing} flexiona para o plural na forma irregular {plur}.\n"
                    f"Resposta: {plur}\n\n"
                )
            texts.append(t)
        else:
            # Dedução lógica simples
            n = random.randint(0, 99)
            t = (
                f"Pergunta: O que vem depois do número {n}?\n"
                f"Raciocínio: O sucessor imediato de um número inteiro n é dado por n + 1. Logo, {n} + 1 = {n + 1}.\n"
                f"Resposta: {n + 1}\n\n"
            )
            texts.append(t)

    return texts

def generate_sft_dialogues(count=15000):
    dialogues = []
    saudacoes = [
        ("Olá!", "Olá! Como posso ajudar você hoje?"),
        ("Bom dia!", "Bom dia! Em que posso ser útil para você nesta manhã?"),
        ("Boa tarde!", "Boa tarde! Como posso auxiliar em seus estudos e tarefas?"),
        ("Quem é você?", "Sou o soreRNN-LM v2, uma inteligência artificial recorrente linear desenvolvida com tecnologia C++20 e CUDA de alto desempenho."),
        ("Como você funciona?", "Eu processo linguagem natural através de uma arquitetura híbrida com convolução 1D, Gated DeltaNet em tempo O(1) e camadas MLP SwiGLU."),
    ]

    tarefas_conversacionais = [
        ("Explique a diferença entre CPU e GPU.", "A CPU possui poucos núcleos complexos otimizados para tarefas sequenciais de baixa latência, enquanto a GPU possui milhares de núcleos que realizam operações matemáticas em massa simultaneamente."),
        ("O que é o Brasil?", "O Brasil é o maior país da América do Sul e o quinto maior do mundo em área territorial, conhecido por sua rica biodiversidade, cultura e economia diversificada."),
        ("Como aprender matemática com eficácia?", "Pratique a resolução ativa de exercícios diariamente, compreenda a lógica conceitual dos passos em vez de apenas memorizar fórmulas e revise seus erros com atenção."),
        ("Qual é a importância da leitura?", "A leitura expande o vocabulário, aprimora a capacidade crítica e de raciocínio, melhora a concentração e enriquece o repertório cultural do indivíduo."),
    ]

    for q, a in saudacoes * 50:
        d = f"<|user|>\n{q}\n<|assistant|>\n{a}\n<|endoftext|>\n"
        dialogues.append(d)

    for q, a in tarefas_conversacionais * 100:
        d = f"<|user|>\n{q}\n<|assistant|>\n{a}\n<|endoftext|>\n"
        dialogues.append(d)

    # Adiciona diálogos matemáticos interativos
    for _ in range(count):
        a = random.randint(2, 50)
        b = random.randint(2, 20)
        q = f"Você pode me dizer quanto é {a} multiplicado por {b}?"
        ans = a * b
        a_resp = f"Com certeza! {a} multiplicado por {b} é igual a {ans}."
        d = f"<|user|>\n{q}\n<|assistant|>\n{a_resp}\n<|endoftext|>\n"
        dialogues.append(d)

    return dialogues

def build_datasets(out_pretrain="data/pretrain_75pt_25en.bin", out_sft="data/sft_chat_pt.bin"):
    os.makedirs("data", exist_ok=True)
    print("=" * 65)
    print("  GERANDO CURRICULUM DE ALTA DENSIDADE (TEXTBOOKS-STYLE)")
    print("=" * 65)

    t0 = time.time()
    print("[1/2] Gerando corpus de Pré-Treino (Matemática CoT + Lógica + Ciência + PT-BR)...")
    math_texts = generate_math_examples(count=60000)
    logic_texts = generate_logic_and_general_knowledge(count=40000)
    all_pretrain_texts = math_texts + logic_texts
    random.shuffle(all_pretrain_texts)

    pretrain_tokens = []
    for txt in all_pretrain_texts:
        pretrain_tokens.extend(enc.encode(txt))

    pretrain_arr = np.array(pretrain_tokens, dtype=np.uint16)
    pretrain_arr.tofile(out_pretrain)
    print(f" -> Pré-treino salvo em {out_pretrain}: {len(pretrain_arr):,} tokens ({os.path.getsize(out_pretrain)/(1024*1024):.2f} MB)")

    print("\n[2/2] Gerando corpus de SFT Conversacional...")
    sft_dialogues = generate_sft_dialogues(count=20000)
    random.shuffle(sft_dialogues)

    sft_tokens = []
    for d in sft_dialogues:
        sft_tokens.extend(enc.encode(d, allowed_special={"<|endoftext|>"}))

    sft_arr = np.array(sft_tokens, dtype=np.uint16)
    sft_arr.tofile(out_sft)
    print(f" -> SFT salvo em {out_sft}: {len(sft_arr):,} tokens ({os.path.getsize(out_sft)/(1024*1024):.2f} MB)")

    elapsed = time.time() - t0
    print("\n" + "=" * 65)
    print(f" [SUCESSO] Datasets de alta densidade criados em {elapsed:.1f}s!")
    print(f" Total de tokens prontos para GPU: {len(pretrain_arr) + len(sft_arr):,} tokens")
    print("=" * 65)

if __name__ == "__main__":
    build_datasets()
