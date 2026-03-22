# Projeto Final de ML com Pipeline, API e Deploy

Projeto da atividade final de Aprendizado de Máquina com foco em um fluxo simples de ponta a ponta: EDA, preprocessamento, treinamento, avaliação, persistência do modelo, API com FastAPI e deploy no Render.

O dataset usado está em `data/dataset_graduacao_indicada.csv` e o alvo previsto é `Graduacao_Indicada`.

## Objetivo do projeto

Entregar um pipeline de Machine Learning funcional e reproduzível para recomendação de graduação, seguindo um escopo MLOps simples:

- carregar dados
- fazer EDA
- limpar e transformar os dados
- testar `LogisticRegression` e `RandomForest`
- comparar métricas
- salvar artefatos
- disponibilizar predição via API
- publicar no Render

## Estrutura do repositório

```text
.
|-- app/
|   |-- main.py
|   |-- predictor.py
|   `-- templates/
|-- artifacts/
|   |-- preprocess_pipeline.joblib
|   |-- model.joblib
|   |-- deploy_logreg.joblib
|   `-- deploy_rf_compacto.joblib
|-- data/
|   |-- dataset_graduacao_indicada.csv
|   `-- processed/
|-- reports/
|   |-- eda.html
|   |-- preprocess.html
|   |-- model_report.html
|   `-- figures/
|-- src/
|   |-- eda.py
|   |-- preprocess.py
|   |-- train.py
|   `-- run_pipeline.py
|-- requirements.txt
`-- render.yaml
```

## Quick Start

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

### 2. Rodar o pipeline completo

```bash
python -B -m src.run_pipeline
```

Esse comando executa, na ordem:

1. EDA
2. preprocessamento
3. treinamento e comparação de modelos
4. geração de artefatos e relatórios
5. registro local no MLflow

### 3. Subir a API localmente

```bash
uvicorn app.main:app --reload
```

Abrir no navegador:

- Interface web: `http://127.0.0.1:8000/`
- Swagger: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`
- Health check: `http://127.0.0.1:8000/health`

## Esteira CI/CD

O projeto agora segue uma esteira MLOps simples, pensada para validação rápida e deploy funcional.

### O que a CI valida

- instalação das dependências do projeto
- sintaxe e importação dos módulos em `app/` e `src/`
- entrypoints principais com `--help`
- presença dos artefatos e arquivos obrigatórios versionados
- smoke tests da API com `GET /health`, `POST /predict`, `POST /web/predict` e cenário degradado com `MODEL_VARIANT` inválida

### Quando a CI roda

- em `pull_request` para `main`
- em `push` para `main`

### Como o deploy acontece

- a validação automática roda no GitHub Actions pelo workflow `ci`
- o deploy continua simples e fica a cargo do Render
- após merge na `main`, o Render pode publicar automaticamente usando o `render.yaml`

### Observação importante

- a esteira inicial não executa `python -m src.run_pipeline`
- o treino completo fica fora da CI para evitar regravação de artefatos, aumento de tempo e inconsistência em um fluxo que precisa ser simples

### Roadmap conceitual

Como evolução futura, a esteira pode incorporar:

- validação formal de dados
- treino automatizado controlado
- model registry
- staging antes de produção
- monitoramento de drift e performance
- rollback e retraining

## Evidências geradas pelo projeto

Após a execução do pipeline completo, o projeto produz:

- `reports/eda.html`: relatório da análise exploratória
- `reports/preprocess.html`: relatório da etapa de tratamento e transformação
- `reports/model_report.html`: comparação entre `LogisticRegression` e `RandomForest`
- `reports/figures/`: gráficos da EDA, do split e da comparação de métricas
- `data/processed/train.parquet` e `data/processed/test.parquet`
- `artifacts/preprocess_pipeline.joblib`
- `artifacts/model.joblib`: melhor modelo do fluxo local
- `artifacts/deploy_logreg.joblib` e `artifacts/deploy_rf_compacto.joblib`: variantes leves para publicação
- `mlruns/`: histórico local de experimentos no MLflow

## O que foi implementado no pipeline

### EDA

- verificação de tipos de dados
- verificação de valores nulos
- análise da distribuição da variável alvo
- correlação entre features
- gráficos e insights finais em HTML

### Preprocessamento

- separação entre `X` e `y`
- identificação de colunas numéricas e categóricas
- imputação simples
- `StandardScaler` para colunas numéricas
- `OneHotEncoder` para variáveis categóricas
- split treino/teste com estratificação e `random_state=42`

### Modelagem

- treinamento de `LogisticRegression`
- treinamento de `RandomForest`
- avaliação com:
  - `accuracy`
  - `precision_macro`
  - `recall_macro`
  - `f1_macro`
- escolha do melhor modelo priorizando `f1_macro` e, em empate, `accuracy`

## API de predição

A API foi implementada com FastAPI e serve tanto JSON quanto interface web.

### Endpoints principais

- `GET /`
- `GET /health`
- `POST /predict`
- `POST /web/predict`

### Contrato de entrada

O payload da API valida os dados antes da inferência:

- `Idade`: entre 14 e 100
- `Anos_Para_Formar`: entre 1 e 15
- preferências: de 1 a 5
- `Curso_Tecnico`: aceita `Sim`, `Nao` e `Não`, com normalização interna

### Exemplo de payload

```json
{
  "Idade": 45,
  "Curso_Tecnico": "Não",
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

### Exemplo com `curl`

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d "{\"Idade\":45,\"Curso_Tecnico\":\"Não\",\"Anos_Para_Formar\":7,\"Gosta_Matematica\":1,\"Gosta_Programacao\":4,\"Gosta_Biologia\":2,\"Gosta_Fisica\":3,\"Gosta_Quimica\":5,\"Gosta_Arte_Design\":3,\"Gosta_Comunicacao\":1,\"Gosta_Negocios\":1,\"Gosta_Historia\":5,\"Gosta_Geografia\":1}"
```

### Teste manual recomendado

1. Abrir `http://127.0.0.1:8000/`
2. Preencher o formulário
3. Enviar a predição
4. Conferir curso previsto, probabilidade e modelo ativo
5. Validar `GET /health` para confirmar carga dos artefatos

## MLflow

Depois do treinamento, os experimentos podem ser visualizados localmente com:

```bash
mlflow ui --backend-store-uri mlruns
```

Interface local:

```text
http://127.0.0.1:5000
```

No MLflow ficam registrados:

- parâmetros dos modelos
- métricas de teste
- artefatos do melhor modelo
- relatório HTML de treinamento

## Deploy no Render

O projeto está preparado para deploy sem Docker usando `render.yaml`.

Configuração atual:

- runtime Python
- instalação com `pip install -r requirements.txt`
- start com `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- health check em `/health`

### Variável de ambiente

- `MODEL_VARIANT`

Valores suportados:

- `logreg`
- `rf_compacto`

Valor padrão no deploy atual:

```text
MODEL_VARIANT=logreg
```

### Passos de publicação

1. Subir o repositório para o GitHub com os artefatos de deploy versionados.
2. Criar um novo serviço no Render usando o repositório.
3. Confirmar o uso do arquivo `render.yaml`.
4. Validar a env var `MODEL_VARIANT`.
5. Ativar auto-deploy da branch `main` no serviço do Render.
6. Aguardar o build e testar a aplicação publicada.

### Validações após deploy

- `GET /health`
- `GET /docs`
- `POST /predict`
- confirmação de que a publicação ocorreu após merge na `main`

## Comandos por etapa

### EDA

```bash
python -B -m src.eda
```

Com parâmetros:

```bash
python -B -m src.eda --input data/dataset_graduacao_indicada.csv --out reports
```

### Preprocessamento

```bash
python -B -m src.preprocess --input data/dataset_graduacao_indicada.csv --outdir data/processed --artifacts artifacts --reports reports
```

### Treinamento

```bash
python -B -m src.train --data_dir data/processed --artifacts artifacts --reports reports
```

Exemplo de experimento alternativo:

```bash
python -B -m src.train --logreg-c 0.5 --logreg-max-iter 3000 --rf-n-estimators 500 --rf-max-depth 18
```

### Pipeline completo

```bash
python -B -m src.run_pipeline
```

### API local com seleção de variante

PowerShell:

```powershell
$env:MODEL_VARIANT="logreg"
uvicorn app.main:app --reload
```

ou

```powershell
$env:MODEL_VARIANT="rf_compacto"
uvicorn app.main:app --reload
```

## Dependências principais

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

## Observação final

O projeto foi mantido propositalmente simples, com scripts diretos e foco em funcionalidade local, para atender ao escopo de um MLOps enxuto e pronto para demonstração.
