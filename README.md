# Projeto de Análise Exploratória e Pipeline de ML

Este projeto faz parte da atividade final da UC de Aprendizado de Máquina.

O foco atual é construir um pipeline simples e reprodutível para o dataset `data/dataset_graduacao_indicada.csv`, começando pela EDA e avançando para a preparação de dados.

## Estrutura do projeto

```text
.
|-- AGENTS.md
|-- atividade_final.pdf
|-- artifacts/
|-- data/
|   |-- dataset_graduacao_indicada.csv
|   |-- graducao indicada.txt
|   `-- processed/
|-- reports/
|   |-- eda.html
|   |-- preprocess.html
|   |-- model_report.html
|   `-- figures/
`-- src/
    |-- __init__.py
    |-- eda.py
    |-- preprocess.py
    |-- run_pipeline.py
    |-- utils/
    |   `-- report_utils.py
    `-- train.py
```

## Como rodar o pipeline completo

Na raiz do projeto, execute:

```bash
python -B -m src.run_pipeline
```

Esse comando executa em sequência:

- EDA
- preparação de dados
- treinamento de modelos

Parâmetros disponíveis:

- `--input`: caminho do arquivo CSV
- `--reports`: diretório dos relatórios HTML
- `--processed`: diretório dos datasets processados
- `--artifacts`: diretório dos artefatos
- `--test-size`: proporção do conjunto de teste
- `--random-state`: seed fixa para reprodutibilidade

## Como rodar a análise exploratória

Na raiz do projeto, execute:

```bash
python -B -m src.eda
```

Esse comando:

- lê a base em `data/dataset_graduacao_indicada.csv`
- gera o relatório em `reports/eda.html`
- gera os gráficos em `reports/figures/`

Se quiser informar outro arquivo de entrada ou outro diretório de saída:

```bash
python -B -m src.eda --input data/dataset_graduacao_indicada.csv --out reports
```

Parâmetros disponíveis:

- `--input`: caminho do arquivo CSV
- `--out`: pasta onde o relatório e as figuras serão salvos

## Como rodar a preparação de dados

Na raiz do projeto, execute:

```bash
python -B -m src.preprocess --input data/dataset_graduacao_indicada.csv --outdir data/processed --artifacts artifacts
```

Esse comando:

- separa `X` e `y`
- define colunas numéricas e categóricas
- faz split treino/teste com `random_state` fixo e estratificação
- aplica imputação simples, `StandardScaler` e `OneHotEncoder`
- salva o pipeline em `artifacts/preprocess_pipeline.joblib`
- salva os datasets processados em `data/processed/train.parquet` e `data/processed/test.parquet`
- gera o relatório em `reports/preprocess.html`

Parâmetros disponíveis:

- `--input`: caminho do arquivo CSV
- `--outdir`: pasta onde os datasets processados serão salvos
- `--artifacts`: pasta onde o pipeline será salvo
- `--test-size`: proporção do conjunto de teste
- `--random-state`: seed fixa para reprodutibilidade

## Como rodar o treinamento de modelos

Na raiz do projeto, execute:

```bash
python -B -m src.train --data_dir data/processed --artifacts artifacts --reports reports
```

Esse comando:

- carrega os datasets processados de treino e teste
- treina `LogisticRegression` e `RandomForest`
- calcula `accuracy`, `precision macro`, `recall macro` e `f1 macro`
- compara os resultados e escolhe o melhor modelo
- salva o melhor artefato em `artifacts/model.joblib`
- gera o relatório em `reports/model_report.html`

Parâmetros disponíveis:

- `--data_dir`: pasta com `train.parquet` e `test.parquet`
- `--artifacts`: pasta onde o melhor modelo será salvo
- `--reports`: pasta onde o relatório será salvo
- `--random-state`: seed fixa para reprodutibilidade

## Saídas geradas

Após rodar a análise exploratória:

- `reports/eda.html`
- `reports/figures/target_distribution.png`
- `reports/figures/curso_tecnico_distribution.png`
- `reports/figures/age_distribution.png`
- `reports/figures/feature_correlation.png`
- `reports/figures/preference_by_course.png`

Após rodar a preparação de dados:

- `reports/preprocess.html`
- `artifacts/preprocess_pipeline.joblib`
- `data/processed/train.parquet`
- `data/processed/test.parquet`

Após rodar o treinamento:

- `reports/model_report.html`
- `artifacts/model.joblib`

## Documentação das etapas

- EDA: `reports/eda.html`
- Preparação de dados: `reports/preprocess.html`
- Treinamento de modelos: `reports/model_report.html`
- Prompt base para relatórios: `context/html_report_prompt.md`

## O que a EDA cobre

- visão geral do dataset
- tipos de dados
- valores nulos
- linhas duplicadas
- estatísticas descritivas
- distribuição da variável alvo
- distribuição de `Curso_Tecnico`
- distribuição de idade
- correlação entre features
- médias de preferência por graduação indicada
- insights finais e conclusão

## O que o preprocessamento cobre

- separação entre variáveis explicativas e alvo
- identificação de colunas numéricas e categóricas
- imputação simples
- normalização de variáveis numéricas
- encoding de variáveis categóricas
- split treino/teste com seed fixa
- geração de artefatos carregáveis para a modelagem

## O que o treinamento cobre

- carregamento dos datasets processados
- treinamento de dois modelos de classificação
- comparação de métricas no conjunto de teste
- seleção do melhor modelo
- serialização do modelo final para as próximas etapas

## Dependências

Os scripts usam as bibliotecas:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `joblib`
- `pyarrow`

Se necessário, instale com:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib pyarrow
```

## Observação

O projeto foi ajustado para:

- usar a base oficial dentro da pasta `data/`
- gerar textos em português
- produzir gráficos mais coerentes visualmente para a entrega
- manter o preprocessamento reprodutível e sem vazamento de dados
