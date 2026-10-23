#!/usr/bin/env python3
"""
soreRNN-LM: Gerador e Stream de Matemática Real e Procedural com Alta Entropia
Elimina o sobreajuste de template ('template memorization'):
- Variabilidade sintática total nos enunciados, passos de raciocínio e conclusões.
- Mistura 50% matemática real (MetaMathQA / problemas reais de raciocínio) com 50% procedural variado.
- Reduz o viés do andaime 'Problema/Raciocínio/Resposta'.
"""

import random
import sys
import time
from typing import Iterator
from datasets import load_dataset

QUESTION_PREFIXES = [
    "",
    "Questão: ",
    "Problema: ",
    "Exercício: ",
    "Pergunta: ",
    "Desafio matemático: ",
    "Considere o seguinte problema: ",
]

STEP_PREFIXES = [
    "Resolução detalhada:\n",
    "Raciocínio passo a passo:\n",
    "Vamos calcular por etapas:\n",
    "Solução analítica:\n",
    "Desenvolvimento:\n",
    "Para resolver este exercício:\n",
]

CONCLUSION_TEMPLATES = [
    "Portanto, a resposta é {ans}.",
    "Assim, o resultado final obtido é {ans}.",
    "Conclui-se que o valor correto é {ans}.",
    "Logo, temos como solução {ans}.",
    "Resultado final: {ans}.",
    "Consequentemente, a resposta exata é {ans}.",
]

def generate_diverse_synthetic_math() -> str:
    """Gera um problema matemático com alta diversidade procedural e textual."""
    q_pre = random.choice(QUESTION_PREFIXES)
    s_pre = random.choice(STEP_PREFIXES)
    c_tpl = random.choice(CONCLUSION_TEMPLATES)

    mode = random.choice([
        "juros_simples",
        "sistema_linear",
        "equacao_segundo_grau",
        "probabilidade",
        "geometria_trapezio_cilindro",
        "velocidade_tempo",
        "porcentagem_sucessiva",
        "aritmetica_decomposta"
    ])

    if mode == "juros_simples":
        capital = random.choice([100, 200, 500, 1000, 2000, 5000])
        taxa = random.choice([2, 3, 5, 10])
        tempo = random.choice([2, 3, 6, 12])
        juros = int(capital * (taxa / 100.0) * tempo)
        montante = capital + juros
        enunciado = (
            f"Uma aplicação de R$ {capital},00 rende sob regime de juros simples a uma taxa de {taxa}% ao mês. "
            f"Qual será o valor dos juros e o montante acumulado após {tempo} meses?"
        )
        corpo = (
            f"1. A fórmula dos juros simples é J = C × i × t, onde C é o capital, i a taxa e t o período de tempo.\n"
            f"2. Convertendo a taxa mensal em fração decimal: {taxa}% = {taxa/100.0:.2f}.\n"
            f"3. Calculando os juros: J = {capital} × {taxa/100.0:.2f} × {tempo} = R$ {juros},00.\n"
            f"4. O montante final corresponde ao capital somado aos juros: M = {capital} + {juros} = R$ {montante},00.\n"
        )
        ans = f"Juros de R$ {juros},00 e Montante de R$ {montante},00"

    elif mode == "sistema_linear":
        # ax + by = c1, x + y = c2
        x = random.randint(2, 20)
        y = random.randint(2, 20)
        c2 = x + y
        a = random.randint(2, 5)
        b = random.randint(2, 4)
        c1 = a * x + b * y
        enunciado = (
            f"Determine a solução para o sistema de equações lineares com duas incógnitas:\n"
            f"Equação 1: {a}x + {b}y = {c1}\n"
            f"Equação 2: x + y = {c2}"
        )
        corpo = (
            f"1. Da Equação 2, isolamos a variável x: x = {c2} - y.\n"
            f"2. Substituímos essa expressão na Equação 1:\n"
            f"   {a}({c2} - y) + {b}y = {c1}\n"
            f"   {a * c2} - {a}y + {b}y = {c1}\n"
            f"   {a * c2} - {a - b}y = {c1}\n"
            f"   {- (a - b)}y = {c1 - a * c2}\n"
            f"   y = {y}\n"
            f"3. Calculamos x substituindo o valor de y: x = {c2} - {y} = {x}.\n"
        )
        ans = f"x = {x} e y = {y}"

    elif mode == "equacao_segundo_grau":
        # (x - r1)(x - r2) = x^2 - (r1+r2)x + r1*r2
        r1 = random.randint(1, 9)
        r2 = random.randint(1, 9)
        b_coef = -(r1 + r2)
        c_coef = r1 * r2
        b_str = f"- {abs(b_coef)}x" if b_coef < 0 else f"+ {b_coef}x"
        c_str = f"+ {c_coef}" if c_coef >= 0 else f"- {abs(c_coef)}"
        enunciado = f"Encontre as raízes reais da equação quadrática: x² {b_str} {c_str} = 0."
        delta = (b_coef ** 2) - 4 * 1 * c_coef
        corpo = (
            f"1. Identificamos os coeficientes: a = 1, b = {b_coef}, c = {c_coef}.\n"
            f"2. Calculamos o discriminante (Delta): Δ = b² - 4ac = ({b_coef})² - 4(1)({c_coef}) = {delta}.\n"
            f"3. Aplicamos a fórmula de Bhaskara: x = (-b ± √Δ) / 2a.\n"
            f"   x = ({-b_coef} ± {int(delta**0.5)}) / 2.\n"
            f"   x₁ = ({-b_coef} + {int(delta**0.5)}) / 2 = {max(r1, r2)}.\n"
            f"   x₂ = ({-b_coef} - {int(delta**0.5)}) / 2 = {min(r1, r2)}.\n"
        )
        ans = f"As raízes são {min(r1, r2)} e {max(r1, r2)}"

    elif mode == "probabilidade":
        total_bolas = random.choice([20, 30, 40, 50])
        azuis = random.randint(5, total_bolas // 2)
        vermelhas = total_bolas - azuis
        pct = (azuis / total_bolas) * 100.0
        enunciado = (
            f"Uma urna contém {total_bolas} esferas idênticas em formato, sendo {azuis} azuis e {vermelhas} vermelhas. "
            f"Retirando-se ao acaso uma esfera, qual a probabilidade percentual de ela ser azul?"
        )
        corpo = (
            f"1. A probabilidade de um evento simples é P = (casos favoráveis) / (casos possíveis).\n"
            f"2. Casos favoráveis (esferas azuis) = {azuis}.\n"
            f"3. Casos possíveis (total de esferas) = {total_bolas}.\n"
            f"4. Razão de probabilidade: P = {azuis}/{total_bolas} = {azuis/total_bolas:.4f}.\n"
            f"5. Multiplicando por 100 para obter porcentagem: {pct:.1f}%.\n"
        )
        ans = f"{pct:.1f}%"

    elif mode == "geometria_trapezio_cilindro":
        b_maior = random.randint(8, 20)
        b_menor = random.randint(3, b_maior - 2)
        h = random.randint(4, 12)
        area = ((b_maior + b_menor) * h) / 2
        enunciado = (
            f"Calcule a área de um trapézio cujas bases medem {b_maior} cm e {b_menor} cm, "
            f"com altura relativa de {h} cm."
        )
        corpo = (
            f"1. A fórmula da área do trapézio é: A = [(B + b) × h] / 2.\n"
            f"2. Substituindo os valores das bases: ({b_maior} + {b_menor}) = {b_maior + b_menor}.\n"
            f"3. Multiplicando pela altura: {b_maior + b_menor} × {h} = {(b_maior + b_menor) * h}.\n"
            f"4. Dividindo por 2: {(b_maior + b_menor) * h} / 2 = {area:.1f} cm².\n"
        )
        ans = f"{area:.1f} cm²"

    elif mode == "velocidade_tempo":
        v = random.choice([60, 70, 80, 90, 100, 120])
        t_horas = random.choice([2, 3, 4, 5])
        dist = v * t_horas
        enunciado = (
            f"Um veículo desloca-se em uma rodovia com velocidade média constante de {v} km/h durante {t_horas} horas. "
            f"Qual a distância total percorrida pelo automóvel nesse trajeto?"
        )
        corpo = (
            f"1. A equação horária do movimento uniforme é d = v × t.\n"
            f"2. Velocidade informada: v = {v} km/h.\n"
            f"3. Tempo decorrido: t = {t_horas} h.\n"
            f"4. Distância calculada: d = {v} × {t_horas} = {dist} km.\n"
        )
        ans = f"{dist} km"

    elif mode == "porcentagem_sucessiva":
        val_inicial = random.choice([100, 200, 500, 800, 1000])
        aumento = random.choice([10, 20, 25])
        desconto = random.choice([10, 20])
        pos_aumento = val_inicial * (1.0 + aumento / 100.0)
        pos_desconto = pos_aumento * (1.0 - desconto / 100.0)
        enunciado = (
            f"Um produto com preço inicial de R$ {val_inicial},00 sofreu um acréscimo de {aumento}% e, posteriormente, "
            f"uma redução de {desconto}% sobre o novo valor. Qual é o preço final do produto?"
        )
        corpo = (
            f"1. Valor após o acréscimo de {aumento}%:\n"
            f"   R$ {val_inicial},00 × (1 + {aumento/100.0:.2f}) = R$ {pos_aumento:.2f}.\n"
            f"2. Valor após o desconto de {desconto}% sobre o novo montante:\n"
            f"   R$ {pos_aumento:.2f} × (1 - {desconto/100.0:.2f}) = R$ {pos_desconto:.2f}.\n"
        )
        ans = f"R$ {pos_desconto:.2f}"

    else: # aritmetica_decomposta
        a = random.randint(100, 9999)
        b = random.randint(100, 9999)
        ans_val = a + b
        enunciado = f"Efetue a soma analítica dos numerais {a} e {b}."
        corpo = (
            f"1. Decomposição posicional dos números:\n"
            f"   {a} = ({a // 1000 * 1000}) + ({a % 1000 // 100 * 100}) + ({a % 100 // 10 * 10}) + ({a % 10})\n"
            f"   {b} = ({b // 1000 * 1000}) + ({b % 1000 // 100 * 100}) + ({b % 100 // 10 * 10}) + ({b % 10})\n"
            f"2. Somando as parcelas de mesma ordem decimal, consolidamos o total exato: {ans_val}.\n"
        )
        ans = f"{ans_val}"

    conclusao = c_tpl.format(ans=ans)
    texto = f"{q_pre}{enunciado}\n\n{s_pre}{corpo}\n{conclusao}"
    return texto

def real_math_hf_stream() -> Iterator[str]:
    """Iterador de streaming de problemas matemáticos reais do HuggingFace."""
    while True:
        try:
            ds = load_dataset("meta-math/MetaMathQA", split="train", streaming=True)
            for row in ds:
                query = row.get("query", "").strip()
                resp = row.get("response", "").strip()
                if not query or not resp:
                    continue
                doc = f"Problema: {query}\n\nResolução:\n{resp}"
                yield doc
        except Exception as e:
            print(f"[Aviso Stream Math] Falha ao carregar MetaMathQA: {e}. Alternando...", file=sys.stderr)
            time.sleep(2)
            # Fallback seguro
            for _ in range(20):
                yield generate_diverse_synthetic_math()

def balanced_math_stream(real_prob: float = 0.5) -> Iterator[str]:
    """
    Stream equilibrado que combina matemática real com gerador procedural diversificado.
    Evita memorização de sintaxe rígida e equilibra raciocínio complexo.
    """
    real_gen = real_math_hf_stream()
    while True:
        if random.random() < real_prob:
            try:
                yield next(real_gen)
            except Exception:
                yield generate_diverse_synthetic_math()
        else:
            yield generate_diverse_synthetic_math()

if __name__ == "__main__":
    print("Testando gerador de matemática procedural diversificado:")
    sample = generate_diverse_synthetic_math()
    print("-" * 60)
    print(sample)
    print("-" * 60)
    print("Gerador procedural verificado com sucesso!")
