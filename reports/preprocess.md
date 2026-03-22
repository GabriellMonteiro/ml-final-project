# Relatório de Preparação de Dados

## Objetivo

Transformar o dataset bruto em uma base pronta para treino e teste, com um pipeline reprodutível e sem vazamento de dados.

## Arquivo de entrada

- Fonte: `data/dataset_graduacao_indicada.csv`
- Coluna alvo: `Graduacao_Indicada`

## Separação de variáveis

### Variáveis numéricas

- `Idade`
- `Anos_Para_Formar`
- `Gosta_Matematica`
- `Gosta_Programacao`
- `Gosta_Biologia`
- `Gosta_Fisica`
- `Gosta_Quimica`
- `Gosta_Arte_Design`
- `Gosta_Comunicacao`
- `Gosta_Negocios`
- `Gosta_Historia`
- `Gosta_Geografia`

### Variáveis categóricas

- `Curso_Tecnico`

## Estratégia de preprocessamento

O preprocessamento foi implementado com `ColumnTransformer` e `Pipeline` do scikit-learn.

### Etapas para variáveis numéricas

- imputação com `SimpleImputer(strategy="median")`
- padronização com `StandardScaler`

### Etapas para variáveis categóricas

- imputação com `SimpleImputer(strategy="most_frequent")`
- encoding com `OneHotEncoder(handle_unknown="ignore")`

## Split de treino e teste

- método: `train_test_split`
- proporção de teste: `20%`
- seed fixa: `42`
- estratificação: aplicada sobre a variável alvo `Graduacao_Indicada`

Essa abordagem mantém a distribuição das classes entre treino e teste e reduz risco de variação aleatória entre execuções.

## Controle de vazamento de dados

O pipeline é ajustado apenas no conjunto de treino:

- `fit_transform` em `X_train`
- `transform` em `X_test`

Isso evita que informações do conjunto de teste influenciem o preprocessamento aprendido.

## Saídas geradas

- pipeline salvo em `artifacts/preprocess_pipeline.joblib`
- treino processado salvo em `data/processed/train.parquet`
- teste processado salvo em `data/processed/test.parquet`

## Resultado final

- shape do treino processado: `(16000, 15)`
- shape do teste processado: `(4000, 15)`
- total de features transformadas: `14`
- coluna alvo preservada nos dois datasets processados

## Validações executadas

- pipeline carregável com `joblib`
- datasets processados sem valores nulos
- coluna alvo presente em `train.parquet` e `test.parquet`
- pipeline compatível com a próxima etapa de modelagem

## Como executar

```bash
python -B -m src.preprocess --input data/dataset_graduacao_indicada.csv --outdir data/processed --artifacts artifacts
```

## Observação

Como o dataset já estava limpo no EDA, o pipeline mantém a imputação simples por robustez e reprodutibilidade, mesmo sem valores ausentes na base atual.
