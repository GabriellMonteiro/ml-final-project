# Projeto de Análise Exploratória e Pipeline de ML

Este projeto faz parte da atividade final da UC de Aprendizado de Máquina.

O oco atual é a etapa de EDA sobre a base `data/dataset_graduacao_indicada.csv`, com geração de relatório em Markdown e gráficos automáticos.
f
## Estrutura do projeto

```text
.
|-- AGENTS.md
|-- atividade_final.pdf
|-- data/
|   |-- dataset_graduacao_indicada.csv
|   |-- graducao indicada.txt
|-- reports/
|   |-- eda.md
|   `-- figures/
`-- src/
    |-- __init__.py
    `-- eda.py
```

## Como rodar a análise exploratória

Na raiz do projeto, execute:

```bash
python -B -m src.eda
```

Esse comando:

- lê a base em `data/dataset_graduacao_indicada.csv`
- gera o relatório em `reports/eda.md`
- gera os gráficos em `reports/figures/`

## Comando com parâmetros

Se quiser informar outro arquivo de entrada ou outro diretório de saída:

```bash
python -B -m src.eda --input data/dataset_graduacao_indicada.csv --out reports
```

Parâmetros disponíveis:

- `--input`: caminho do arquivo CSV
- `--out`: pasta onde o relatório e as figuras serão salvos

## Saídas geradas

Após rodar a análise, os principais artefatos são:

- `reports/eda.md`
- `reports/figures/target_distribution.png`
- `reports/figures/curso_tecnico_distribution.png`
- `reports/figures/age_distribution.png`
- `reports/figures/feature_correlation.png`
- `reports/figures/preference_by_course.png`

## O que a análise verifica

A EDA atual cobre:

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

## Dependências

O script usa as bibliotecas:

- `pandas`
- `matplotlib`
- `seaborn`
- `numpy`

Se necessário, instale com:

```bash
pip install pandas matplotlib seaborn numpy
```

## Observação

O relatório atual foi ajustado para:

- usar a base oficial dentro da pasta `data/`
- gerar textos em português
- produzir gráficos mais coerentes visualmente para a entrega
