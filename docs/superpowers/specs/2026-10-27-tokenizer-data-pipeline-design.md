# soreRNN — Design: Tokenizador Qwen3.5 Podado (96K) e Pipeline de Dados PT-BR

Data: 2026-10-27  
Status: Aprovado para Planejamento  
Alvo: Plano 2 da Arquitetura soreRNN v2 (`docs/superpowers/specs/2026-10-23-sorernn-v2-design.md`, §4 e §5)

---

## 1. Visão Geral e Motivação

O modelo `b0_8` (~0.8B parâmetros) adota a estratégia de pré-treino do zero em corpus enriquecido de português, destilação de logits de teacher (`Qwen/Qwen3.5-0.8B`), e arquitetura recorrente moderna (Gated DeltaNet + Atenção Global).

O vocabulário padrão do Qwen3.5 possui **248.320 tokens**, o que consumiria ~318M parâmetros apenas na camada de embedding linear com `d_model = 1280` (quase 40% de todo o orçamento de parâmetros do modelo).

Este subsistema estabelece:
1. **Poda Controlada do Vocabulário Qwen3.5:** Redução de 248.320 para exatamente **98.304 tokens** (96K, múltiplo de 128 e 256), fechado sob regras de merge BPE e mantendo 100% de compatibilidade com bytes UTF-8 e tokens especiais;
2. **Matrizes de Mapeamento para Destilação de Logits (KD):** Tabelas `old_to_new` e `new_to_old` que permitem projetar os logits do teacher de 248K para o subvocabulário de 96K sem desalinhamento;
3. **Pipeline de Dados em Streaming Multiplexado:** Ingestão das fontes HF em streaming com filtros heurísticos rigorosos de qualidade, deduplicação MinHash LSH local e empacotamento em shards binários `uint32` de 1 GiB ($2^{28}$ tokens), perfeitamente alinhados com janelas de 8192 tokens do `StreamDataset`.

---

## 2. Poda do Tokenizador com Fechamento BPE

### 2.1 Modelo BPE e Grafo de Dependências

O tokenizador do Qwen3.5 opera sobre bytes UTF-8 com o algoritmo Byte-Pair Encoding (BPE). Cada regra de merge unifica dois tokens existentes $A$ e $B$ para formar um token composto $C$:
$$(A, B) \longrightarrow C$$

Para que o tokenizador resultante seja matematicamente consistente e nunca gere falhas em tempo de execução:
- Se um token $C$ pertence ao vocabulário podado, os tokens $A$ e $B$ **obrigatoriamente** também devem pertencer ao vocabulário;
- Por indução, o fechamento transitivo de ancestrais $\text{Ancestrais}(C)$ deve estar contido no vocabulário podado:
  $$\text{Ancestrais}(C) = \{A, B\} \cup \text{Ancestrais}(A) \cup \text{Ancestrais}(B)$$
- A tabela de merges podada deve conter uma regra $(A, B) \to C$ se e somente se $\{A, B, C\} \subset \text{Vocab}_{96k}$.

### 2.2 Conjunto Obrigatório Protegido ($S_{\text{must}}$)

Antes de considerar qualquer frequência de texto, os seguintes elementos são blindados contra poda:
1. **Fallback de Bytes:** Todos os 256 bytes elementares (`<0x00>` até `<0xFF>`). Nenhum byte possui ancestrais;
2. **Tokens Especiais do Qwen3.5:**
   - ChatML: `<|im_start|>`, `<|im_end|>`;
   - Reasoning: `<think>`, `</think>`;
   - Tool Calling: `<tool_call>`, `</tool_call>`, `<tool_response>`;
   - Visão (reservados): `<|vision_start|>`, `<|vision_end|>`, `<|image_pad|>`;
   - Fim de texto / Delimitador de documentos: `<|endoftext|>`.
3. **Fechamento dos Especiais:** Caso algum token especial seja composto por merges na estrutura interna do BPE, todos os seus ancestrais são automaticamente incorporados a $S_{\text{must}}$.

### 2.3 Algoritmo de Seleção Gulosa com Fechamento

Dado o vocabulário alvo $V = 98.304$:
1. **Coleta de Frequências:** Amostramos ~15M tokens de texto real seguindo a mistura acordada (75% PT, 15% EN STEM, 10% Código) e tokenizamos com o modelo original do Qwen3.5. Calculamos a contagem de ocorrências $\text{freq}(t)$ para cada token $t \in [0, 248319]$.
2. **Inicialização:** $S \leftarrow S_{\text{must}}$.
3. **Fila de Prioridade:** Ordenamos os tokens restantes por frequência decrescente: $t_1, t_2, \dots, t_M$.
4. **Iteração de Inclusão:**
   - Para cada token $t_i$:
     - Se $t_i \in S$, continue.
     - Calcule o conjunto de adições necessárias: $\Delta(t_i) = \{t_i\} \cup (\text{Ancestrais}(t_i) \setminus S)$.
     - Se $|S| + |\Delta(t_i)| \le V$:
       - $S \leftarrow S \cup \Delta(t_i)$.
     - Se $|S| == V$, encerre a seleção.
5. **Preenchimento Residual:** Se a lista de frequências for esgotada e $|S| < V$, selecionamos os tokens com menor profundidade de merge cujos ancestrais já estejam satisfeitos até atingir exatamente 98.304 tokens.

### 2.4 Mapeamento para Destilação de Logits (KD)

Para permitir que o modelo teacher (`Qwen3.5-0.8B`, vocab 248.320) destile conhecimento para o student (vocab 98.304):
- Atribuímos aos 98.304 tokens selecionados novos identificadores contíguos $0, 1, \dots, 98303$, preservando posições canônicas para tokens especiais (ex: `<|endoftext|>` como `eos_id = 98303`).
- Exportamos os tensores de mapeamento:
  - `new_to_old` ($\text{shape } [98304]$, `torch.int64`):
    $$\text{student\_logits} = \text{teacher\_logits}[:, \text{new\_to\_old}]$$
  - `old_to_new` ($\text{shape } [248320]$, `torch.int64`): preenchido com `-1` para tokens descartados e `novo_id` para tokens preservados.

---

## 3. Pipeline de Dados de Pré-Treino

### 3.1 Composição e Fontes do Corpus

O pipeline ingere fontes públicas via streaming (`datasets.load_dataset(..., streaming=True)`) com balanceamento estocástico ponderado:

| Categoria | Proporção | Fontes Primárias | Justificativa |
|---|---|---|---|
| **PT Web Filtrada** | 65% | FineWeb-2 (`por_Latn`), CulturaX PT | Grande volume web em português de alta cobertura |
| **PT Alta Qualidade** | 10% | Wikipedia PT (`wikimedia/wikipedia`), SciELO, Corpus Carolina | Precisão gramatical, fatos enciclopédicos e artigos acadêmicos |
| **EN STEM / Edu** | 15% | FineWeb-Edu, Cosmopedia, OpenWebMath | Raciocínio matemático, conceitos científicos e didática |
| **Código & Estruturado** | 10% | The Stack v2 / StarCoder (Python, SQL, Shell, JSON, HTML) | Habilidade de programação, sintaxe estrita e tool calling |

### 3.2 Filtros de Qualidade Heurísticos (`sore.data.filters`)

Cada documento de texto submetido ao pipeline é avaliado sequencialmente:
1. **Comprimento Mínimo:** Texto com pelo menos 150 caracteres e pelo menos 30 palavras (descarta fragmentos, títulos isolados e menus web);
2. **Densidade Alfanumérica:** Razão entre caracteres alfanuméricos e caracteres totais $\ge 0.70$ (elimina dumps de memória, logs binários e ruído de caracteres especiais);
3. **Repetição de Linhas:** Proporção de linhas duplicadas no documento $\le 30\%$;
4. **Repetição de n-gramas:** Razão de repetição de 5-gramas de palavras $\le 20\%$ (bloqueia loops de SEO e preenchimento de palavras-chave);
5. **Stopwords PT (somente para fontes PT):** Presença de no mínimo 3 stopwords frequentes em português (*que, para, com, não, uma, por, sobre, etc.*), descartando textos categorizados incorretamente como português;
6. **Higienização de Boilerplate:** Remoção de tags HTML residuais, assinaturas markdown de badges e espaços em branco duplicados.

### 3.3 Deduplicação Local com MinHash LSH (`sore.data.dedup`)

Para eliminar documentos idênticos ou quase idênticos sem exigir armazenamento global em disco:
- Representação por 5-gramas de palavras;
- Assinatura MinHash com $K = 128$ funções hash pseudo-aleatórias de 64 bits;
- Índice LSH com janela circular deslizante de $100.000$ documentos em memória;
- Documentos com similaridade de Jaccard estimada $\ge 0.80$ são rejeitados pelo pipeline.

### 3.4 Sharding Binário e Alinhamento com `StreamDataset`

- **Tamanho do Shard:** Exatamente $2^{28} = 268.435.456$ tokens por arquivo.
- **Propriedades Matemáticas de Alinhamento:**
  - $268.435.456 = 32.768 \times 8192$ (múltiplo exato da maior janela de contexto dos estágios de treino, `unit_len = 8192`);
  - Todo shard contém exatamente 32.768 unidades inteiras;
  - Zero descarte ou fragmentação de tokens ao alternar entre estágios de 4096 e 8192 tokens;
  - Tamanho exato em disco por shard em `uint32`: **1.073.741.824 bytes (1 GiB exato)**.
- **Empacotamento de Documentos:**
  - O delimitador `<|endoftext|>` (`eos_id = 98303`) é anexado imediatamente ao final de cada documento aceito;
  - Quando a contagem atinge $2^{28}$ tokens, o shard é fechado e gravado no disco com seu arquivo descritor sidecar `.json`.
- **Sidecar JSON (`shard_NNNNN.json`):**
  ```json
  {
    "n_tokens": 268435456,
    "eos_id": 98303,
    "unit_len": 8192,
    "sources": {
      "pt_web": 0.65,
      "pt_hq": 0.10,
      "en_edu": 0.15,
      "code": 0.10
    },
    "sha256": "..."
  }
  ```

---

## 4. Estrutura de Módulos e Componentes

```
sore/
├── data/
│   ├── __init__.py
│   ├── prune_tokenizer.py    # Algoritmo de poda de merges BPE, grafo e exportador HF
│   ├── filters.py            # Filtros heurísticos de qualidade e limpeza
│   ├── dedup.py              # MinHash LSH local em streaming
│   └── pipeline.py           # Ingestão multiplexada, tokenização e sharder binário
scripts/
├── prune_tokenizer.py        # CLI para rodar a calibração e exportar o tokenizer 96k
└── prepare_data.py           # CLI para disparar geração de shards para pré-treino
tests/
└── data/
    ├── test_prune_tokenizer.py
    ├── test_filters.py
    ├── test_dedup.py
    └── test_pipeline.py
```

---

## 5. Portões de Qualidade e Critérios de Aceite

O subsistema é considerado aprovado quando satisfizer os 5 portões abaixo:

1. **Portão de Integridade BPE:**
   - 100% dos pares de merge preservados têm ancestrais presentes no vocabulário;
   - Tamanho final do vocabulário é rigorosamente 98.304 tokens.
2. **Portão de Cobertura UTF-8:**
   - Qualquer sequência arbitrária de bytes UTF-8 é codificável sem geração de token `UNK` (garantido pelos 256 bytes elementares).
3. **Portão de Fertilidade em Português:**
   - Calculada como $\text{Fertilidade} = \frac{\text{total de tokens}}{\text{total de palavras}}$ em uma amostra de teste de 10.000 sentenças em português;
   - Degradação máxima permitida:
     $$\frac{\text{Fertilidade}_{96k}}{\text{Fertilidade}_{248k}} \le 1{,}03 \quad (+3\% \text{ no máximo})$$
4. **Portão de Roundtrip:**
   - Para textos normais, acentuados, JSON, Python e tool calling:
     $$\text{decode}(\text{encode}(T)) == T$$
5. **Portão de Sharding e Compatibilidade com Treino:**
   - Shards binários gerados são lidos pelo `StreamDataset` (`sore/train/data.py`) sem exceções;
   - Treino de fumaça de 10 passos do `b0_8` consome os shards binários com loss decrescente e sem NaNs.
