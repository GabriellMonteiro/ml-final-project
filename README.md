# Projeto de Análise Exploratória e Pipeline de ML

Este projeto faz parte da atividade final da UC de Aprendizado de Máquina.

O foco atual é construir um pipeline simples e reprodutível para o dataset `data/dataset_graduacao_indicada.csv`, começando pela EDA e avançando para a preparação de dados.

## Estrutura do projeto

```text
.
|-- AGENTS.md
|-- app/
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
- registro de experimentos no MLflow local

Parâmetros disponíveis:

- `--input`: caminho do arquivo CSV
- `--reports`: diretório dos relatórios HTML
- `--processed`: diretório dos datasets processados
- `--artifacts`: diretório dos artefatos
- `--test-size`: proporção do conjunto de teste
- `--random-state`: seed fixa para reprodutibilidade
- `--tracking-dir`: diretório local dos runs do MLflow
- `--experiment-name`: nome do experimento no MLflow
- `--logreg-c`: regularização da LogisticRegression
- `--logreg-max-iter`: iterações máximas da LogisticRegression
- `--rf-n-estimators`: quantidade de árvores da RandomForest
- `--rf-max-depth`: profundidade máxima da RandomForest

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
- registra parâmetros, métricas e artefatos em `mlruns/`

Parâmetros disponíveis:

- `--data_dir`: pasta com `train.parquet` e `test.parquet`
- `--artifacts`: pasta onde o melhor modelo será salvo
- `--reports`: pasta onde o relatório será salvo
- `--random-state`: seed fixa para reprodutibilidade
- `--tracking-dir`: diretório local dos runs do MLflow
- `--experiment-name`: nome do experimento no MLflow
- `--logreg-c`: regularização da LogisticRegression
- `--logreg-max-iter`: iterações máximas da LogisticRegression
- `--rf-n-estimators`: quantidade de árvores da RandomForest
- `--rf-max-depth`: profundidade máxima da RandomForest

Exemplo de run alternativo para comparação no MLflow:

```bash
python -B -m src.train --logreg-c 0.5 --logreg-max-iter 3000 --rf-n-estimators 500 --rf-max-depth 18
```

## Como visualizar os experimentos no MLflow

Depois de rodar o treinamento, execute:

```bash
mlflow ui --backend-store-uri mlruns
```

Em seguida, abra no navegador:

```text
http://127.0.0.1:5000
```

No UI voce deve encontrar:

- um run novo para cada execucao de treino
- parametros dos modelos avaliados
- metricas de teste
- artefatos do melhor modelo e do relatorio HTML

## Como rodar a API e a interface web

Na raiz do projeto, execute:

```bash
uvicorn app.main:app --reload
```

Para escolher a variante servida pela API localmente:

```bash
$env:MODEL_VARIANT="logreg"
uvicorn app.main:app --reload
```

ou

```bash
$env:MODEL_VARIANT="rf_compacto"
uvicorn app.main:app --reload
```

Depois abra no navegador:

```text
http://127.0.0.1:8000/
```

A interface principal agora fica na raiz da aplicação e permite responder o questionario sem usar o Swagger.

URLs principais:

- Interface web: `http://127.0.0.1:8000/`
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`
- Health check: `http://127.0.0.1:8000/health`

Documentacoes disponiveis:

- Swagger UI em `/docs`
- ReDoc em `/redoc`

Endpoints principais:

- `GET /`
- `POST /web/predict`
- `GET /health`
- `POST /predict`

Fluxo recomendado para teste manual:

1. Abra `http://127.0.0.1:8000/`
2. Preencha o formulario do questionario
3. Envie a solicitacao e confira a graduacao prevista, a probabilidade e o modelo ativo

Exemplo de payload para `POST /predict`:

```json
{
  "Idade": 45,
  "Curso_Tecnico": "Sim",
  "Anos_Para_Formar": 7,
  "Gosta_Matematica": 1,
  "Gosta_Programacao": 4,
  "Gosta_Biologia": 2,
  "Gosta_Fisica": 3,
  "Gosta_Quimica": 5,
  "Gosta_Arte_Design": 3,
  "Gosta_Comunicacao": 1,
  "Gosta_Negocios": 1,
  "Gosta_Historia": 5,
  "Gosta_Geografia": 1
}
```

Exemplo com `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d "{\"Idade\":45,\"Curso_Tecnico\":\"Sim\",\"Anos_Para_Formar\":7,\"Gosta_Matematica\":1,\"Gosta_Programacao\":4,\"Gosta_Biologia\":2,\"Gosta_Fisica\":3,\"Gosta_Quimica\":5,\"Gosta_Arte_Design\":3,\"Gosta_Comunicacao\":1,\"Gosta_Negocios\":1,\"Gosta_Historia\":5,\"Gosta_Geografia\":1}"
```

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
- `artifacts/deploy_logreg.joblib`
- `artifacts/deploy_rf_compacto.joblib`
- `mlruns/`

Após rodar a API:

- interface web em `http://127.0.0.1:8000/`
- documentação em `http://127.0.0.1:8000/docs`
- documentação alternativa em `http://127.0.0.1:8000/redoc`

## Documentação das etapas

- EDA: `reports/eda.html`
- Preparação de dados: `reports/preprocess.html`
- Treinamento de modelos: `reports/model_report.html`
- Prompt base para relatórios: `context/html_report_prompt.md`

## Deploy no Render

O projeto esta preparado para deploy no Render sem Docker, usando o arquivo `render.yaml`.

Arquivos principais para o deploy:

- `render.yaml`
- `requirements.txt`
- `app/main.py`
- `artifacts/preprocess_pipeline.joblib`
- `artifacts/deploy_logreg.joblib`
- `artifacts/deploy_rf_compacto.joblib`

Variavel de ambiente usada pela API:

- `MODEL_VARIANT`

Valores suportados:

- `logreg`
- `rf_compacto`

Valor padrao recomendado para a primeira publicacao:

- `MODEL_VARIANT=logreg`

Passos de deploy:

1. Suba o repositório no GitHub com os artefatos de deploy versionados.
2. No Render, crie um novo Blueprint ou Web Service a partir do repositório.
3. Confirme o uso do `render.yaml`.
4. Verifique se a env var `MODEL_VARIANT` esta definida como `logreg` ou `rf_compacto`.
5. Aguarde o build instalar as dependencias com `pip install -r requirements.txt`.
6. O start da aplicacao sera feito com `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.

Validacoes apos deploy:

- `GET /health`
- `GET /docs`
- `POST /predict`

Observacao:

- o artefato `artifacts/model.joblib` continua sendo o melhor modelo local do fluxo de treino
- o ambiente publicado no Render usa uma variante leve de deploy escolhida por `MODEL_VARIANT`

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
- `mlflow`
- `pyarrow`
- `fastapi`
- `uvicorn`
- `pydantic`
- `jinja2`
- `python-multipart`

Se necessário, instale com:

```bash
pip install -r requirements.txt
```

Ou, se preferir instalar manualmente:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib mlflow pyarrow fastapi uvicorn pydantic jinja2 python-multipart
```

## Observação

O projeto foi ajustado para:

- usar a base oficial dentro da pasta `data/`
- gerar textos em português
- produzir gráficos mais coerentes visualmente para a entrega
- manter o preprocessamento reprodutível e sem vazamento de dados
