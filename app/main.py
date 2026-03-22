from __future__ import annotations
"""Aplicacao FastAPI para expor o modelo treinado do projeto."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field

from app.predictor import PredictorService

EDUCATIONAL_MESSAGE = (
    "Este conteudo e destinado apenas para fins educacionais. "
    "Os dados exibidos sao ilustrativos e podem nao corresponder a situacoes reais."
)


class PredictionRequest(BaseModel):
    """Schema de entrada do endpoint de predicao."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
                "Gosta_Geografia": 1,
            }
        }
    )

    Idade: int = Field(description="Idade do estudante.")
    Curso_Tecnico: str = Field(description="Se o estudante possui curso tecnico. Exemplo: 'Sim' ou 'Nao'.")
    Anos_Para_Formar: int = Field(description="Quantidade estimada de anos para concluir a graduacao.")
    Gosta_Matematica: int = Field(description="Nivel de afinidade com matematica, de 1 a 5.")
    Gosta_Programacao: int = Field(description="Nivel de afinidade com programacao, de 1 a 5.")
    Gosta_Biologia: int = Field(description="Nivel de afinidade com biologia, de 1 a 5.")
    Gosta_Fisica: int = Field(description="Nivel de afinidade com fisica, de 1 a 5.")
    Gosta_Quimica: int = Field(description="Nivel de afinidade com quimica, de 1 a 5.")
    Gosta_Arte_Design: int = Field(description="Nivel de afinidade com arte e design, de 1 a 5.")
    Gosta_Comunicacao: int = Field(description="Nivel de afinidade com comunicacao, de 1 a 5.")
    Gosta_Negocios: int = Field(description="Nivel de afinidade com negocios, de 1 a 5.")
    Gosta_Historia: int = Field(description="Nivel de afinidade com historia, de 1 a 5.")
    Gosta_Geografia: int = Field(description="Nivel de afinidade com geografia, de 1 a 5.")


class PredictionResponse(BaseModel):
    """Schema de resposta do endpoint de predicao."""

    prediction: str = Field(description="Graduacao indicada pelo modelo.")
    probability: float = Field(description="Probabilidade estimada da classe prevista.")
    model_name: str = Field(description="Nome do modelo carregado pela API.")


class HealthResponse(BaseModel):
    """Schema de resposta do endpoint de saude."""

    status: str = Field(description="Estado geral da API.")
    model_loaded: bool = Field(description="Indica se o modelo foi carregado com sucesso.")
    preprocess_loaded: bool = Field(description="Indica se o preprocessador foi carregado com sucesso.")
    model_name: str = Field(description="Nome do modelo ativo.")
    model_variant: str = Field(description="Variante do modelo selecionada por configuracao.")
    model_path: str = Field(description="Caminho do artefato escolhido para a API.")
    error: str | None = Field(default=None, description="Mensagem de erro de carga, quando existir.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega os artefatos antes da API comecar a atender requests."""
    predictor = PredictorService()
    predictor.load()
    app.state.predictor = predictor
    yield


app = FastAPI(
    title="API de Predicao de Graduacao",
    description=(
        "API local para inferencia do modelo treinado no projeto. "
        "A documentacao interativa esta disponivel no Swagger."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def get_predictor(request: Request) -> PredictorService:
    """Recupera a instancia compartilhada do helper de inferencia."""
    return request.app.state.predictor


@app.get("/", tags=["Geral"])
def read_root() -> dict[str, str]:
    """Exibe a mensagem educacional e indica onde consultar a documentacao."""
    return {
        "message": EDUCATIONAL_MESSAGE,
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "health_url": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Geral"])
def health_check(request: Request, response: Response) -> HealthResponse:
    """Informa o estado da aplicacao e dos artefatos carregados."""
    predictor = get_predictor(request)
    if not predictor.ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(**predictor.health_payload())


@app.post("/predict", response_model=PredictionResponse, tags=["Predicao"])
def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    """Executa uma predicao para um unico registro bruto."""
    predictor = get_predictor(request)
    if not predictor.ready:
        raise HTTPException(status_code=503, detail="Artefatos do modelo nao estao disponiveis.")

    result = predictor.predict(payload.model_dump())
    return PredictionResponse(
        prediction=result.prediction,
        probability=result.probability,
        model_name=result.model_name,
    )
