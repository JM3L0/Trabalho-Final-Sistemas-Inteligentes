# 🧠 Classificação de Gliomas Cerebrais com Machine Learning

> **Trabalho Final — Disciplina de Sistemas Inteligentes**  
> Estudo comparativo de técnicas de classificação (KNN, MLP e BernoulliNB) aplicadas ao diagnóstico de gliomas cerebrais utilizando dados clínicos e mutacionais do dataset TCGA.

---

## 📋 Sobre o Projeto

Gliomas são os tumores cerebrais primários mais comuns e apresentam dois subtipos com prognósticos radicalmente distintos:

- **LGG** (*Low-Grade Glioma*) — crescimento lento, melhor prognóstico
- **GBM** (*Glioblastoma Multiforme*) — altamente agressivo, prognóstico reservado

Este trabalho aplica e compara três algoritmos de classificação supervisionada para **diferenciar automaticamente LGG de GBM** com base em marcadores moleculares e dados clínicos. O pipeline implementa boas práticas de machine learning, incluindo pré-processamento robusto, busca de hiperparâmetros via GridSearchCV e avaliação com múltiplas métricas.

---

## 📊 Dataset

| Propriedade | Valor |
|---|---|
| **Fonte** | UCI Machine Learning Repository (ID 759) |
| **Nome** | TCGA Glioma |
| **Instâncias** | 839 pacientes |
| **Atributos** | 23 (1 contínuo, 2 categóricos, 20 binários moleculares) |
| **Tarefa** | Classificação binária (LGG = 0 vs GBM = 1) |

**Atributos principais:** Idade ao diagnóstico, Gênero, Raça e 20 marcadores moleculares binários (e.g., mutações IDH1, TP53, ATRX, CDKN2A, entre outros).

---

## 🏗️ Estrutura do Pipeline

O notebook está organizado em **7 etapas sequenciais**:

| Etapa | Descrição |
|---|---|
| **1 — Setup e Carregamento** | Instalação de dependências e carregamento do dataset via `ucimlrepo` |
| **2 — Pré-processamento** | `StandardScaler` para contínuos, `OneHotEncoder` para categóricos, remoção de nulos e duplicatas |
| **3 — Divisão dos Dados** | Hold-Out 70/30 estratificado + K-Fold Estratificado (5 folds) |
| **4 — Implementação dos Modelos** | `Pipeline` + `GridSearchCV` para KNN, MLP e BernoulliNB |
| **5 — Métricas de Avaliação** | Acurácia, Precisão, Recall, F1, Kappa de Cohen, AUC-ROC |
| **6 — Análise de Erros** | Análise de FP/FN com foco em mutações IDH1, TP53 e ATRX |
| **7 — Visualizações** | Distribuição de classes, heatmap de correlação, matrizes de confusão, curvas ROC e comparativo de métricas |

---

## 📈 Resultados

### Métricas no conjunto de teste (Hold-Out 30%)

| Métrica | KNN | MLP | BernoulliNB |
|---|---|---|---|
| **Acurácia** | **0.8730** | 0.8016 | 0.8651 |
| **Precisão** | 0.8246 | **0.8333** | 0.8000 |
| **Recall** | 0.8868 | 0.6604 | **0.9057** |
| **F1-Score** | **0.8545** | 0.7368 | 0.8496 |
| **Kappa** | **0.7421** | 0.5810 | 0.7281 |
| **AUC-ROC** | 0.9038 | 0.8874 | **0.9083** |

### Análise de Erros

| Modelo | Total de Erros | FP | FN | Erros c/ IDH1 Wildtype |
|---|---|---|---|---|
| **KNN** | **32** | 20 | 12 | 23 |
| MLP | 50 | 14 | **36** | 40 |
| BernoulliNB | 34 | 24 | **10** | 25 |

> 💡 **KNN** obteve o melhor equilíbrio geral (menor total de erros, maior F1 e Kappa). **BernoulliNB** destacou-se no AUC-ROC e no menor número de Falsos Negativos, o que é crítico no contexto clínico para não deixar casos de GBM passarem despercebidos.

---

## 🚀 Como Executar

O projeto foi desenvolvido no **Google Colab** — nenhuma instalação local é necessária.

### 1. Abrir no Google Colab

1. Acesse o arquivo `Trabalho_Final.ipynb` neste repositório
2. Clique no botão **"Open in Colab"** ou abra manualmente via:
   > **Colab → Arquivo → Abrir notebook → GitHub → cole a URL deste repositório**

### 2. Executar o Notebook

Após abrir no Colab:

1. Vá em **Ambiente de execução → Conectar**
2. Em seguida, clique em **Ambiente de execução → Executar tudo**

O notebook irá automaticamente:
- ✅ Instalar as dependências necessárias (`ucimlrepo`)
- ✅ Baixar e carregar o dataset TCGA Glioma
- ✅ Executar o pré-processamento e treinar os modelos
- ✅ Exibir todas as métricas e visualizações

---

## 📦 Dependências

Todas as bibliotecas são instaladas automaticamente pelo notebook. Abaixo a lista para referência:

| Biblioteca | Uso |
|---|---|
| `ucimlrepo` | Download automático do dataset UCI |
| `pandas` / `numpy` | Manipulação de dados |
| `scikit-learn` | Modelos, pipelines, métricas e GridSearchCV |
| `matplotlib` / `seaborn` | Visualizações |

> As demais dependências (`scikit-learn`, `pandas`, `matplotlib`, `seaborn`) já estão disponíveis no ambiente padrão do Google Colab.

---

## 🧬 Modelos e Hiperparâmetros

### KNN — K-Nearest Neighbors
```
n_neighbors : [3, 5, 7, 9, 11]
weights     : ['uniform', 'distance']
metric      : ['euclidean', 'manhattan', 'hamming']
```

### MLP — Multilayer Perceptron
```
hidden_layer_sizes : [(32,), (64,), (32,16), (64,32)]
activation         : ['relu', 'tanh']
solver             : ['adam', 'lbfgs']
alpha              : [0.0001, 0.001, 0.01]
learning_rate      : ['constant', 'adaptive']
```

### BernoulliNB — Naive Bayes Bernoulli
```
alpha      : [1e-9, 1e-3, 0.1, 0.5, 1.0]
binarize   : [None, 0.0, 0.5]
fit_prior  : [True, False]
```

A seleção do melhor conjunto de hiperparâmetros foi realizada via **GridSearchCV** com **K-Fold Estratificado (5 folds)**, otimizando o **F1-Score** para a classe GBM (positiva).

---

## 📁 Estrutura do Repositório

```
Trabalho-Final-Sistemas-Inteligentes/
│
├── Trabalho_Final.ipynb   # Notebook principal com todo o pipeline
└── README.md              # Documentação do projeto
```

---

## 👨‍💻 Disciplina

**Sistemas Inteligentes** — Trabalho Final  
Dataset: [TCGA Glioma — UCI ML Repository (ID 759)](https://archive.ics.uci.edu/dataset/759/glioma+grading+clinical+and+mutation+features+dataset)
