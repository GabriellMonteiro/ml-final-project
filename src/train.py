from __future__ import annotations
"""Script de treinamento, avaliacao e tracking de experimentos.

Treina os dois modelos exigidos pelo escopo, compara metricas no conjunto
de teste, salva o melhor artefato e registra a execucao no MLflow local.
"""

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

from src.utils.mlflow_utils import log_training_experiment
from src.utils.report_utils import build_structured_report

TARGET_COLUMN = "Graduacao_Indicada"
DEPLOY_RF_N_ESTIMATORS = 80
DEPLOY_RF_MAX_DEPTH = 10


def parse_args() -> argparse.Namespace:
    """Le os argumentos de linha de comando do script de treino."""
    parser = argparse.ArgumentParser(description="Executa o treinamento e a avaliacao dos modelos.")
    parser.add_argument(
        "--data_dir",
        default="data/processed",
        help="Diretorio com os arquivos train.parquet e test.parquet.",
    )
    parser.add_argument(
        "--artifacts",
        default="artifacts",
        help="Diretorio onde o melhor modelo sera salvo.",
    )
    parser.add_argument(
        "--reports",
        default="reports",
        help="Diretorio onde o relatorio de metricas sera salvo.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed fixa para reprodutibilidade.",
    )
    parser.add_argument(
        "--tracking-dir",
        default="mlruns",
        help="Diretorio local onde o MLflow vai salvar os runs.",
    )
    parser.add_argument(
        "--experiment-name",
        default="graduacao-indicada-classificacao",
        help="Nome do experimento no MLflow.",
    )
    parser.add_argument(
        "--logreg-c",
        type=float,
        default=1.0,
        help="Valor de regularizacao C da LogisticRegression.",
    )
    parser.add_argument(
        "--logreg-max-iter",
        type=int,
        default=2000,
        help="Numero maximo de iteracoes da LogisticRegression.",
    )
    parser.add_argument(
        "--rf-n-estimators",
        type=int,
        default=300,
        help="Quantidade de arvores da RandomForest.",
    )
    parser.add_argument(
        "--rf-max-depth",
        type=int,
        default=None,
        help="Profundidade maxima da RandomForest. Omitido usa crescimento livre.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    """Cria o diretorio caso ele ainda nao exista."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_figures_dir(reports_dir: Path) -> Path:
    """Garante a existencia da pasta de figuras do relatorio."""
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir


def load_splits(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega os datasets processados de treino e teste."""
    train_path = data_dir / "train.parquet"
    test_path = data_dir / "test.parquet"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Arquivos train.parquet e test.parquet sao obrigatorios.")

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    return train_df, test_df


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa as features da coluna alvo."""
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def build_models(
    random_state: int,
    logreg_c: float,
    logreg_max_iter: int,
    rf_n_estimators: int,
    rf_max_depth: int | None,
) -> dict[str, object]:
    """Monta os modelos exigidos pelo escopo da atividade."""
    return {
        "LogisticRegression": LogisticRegression(
            C=logreg_c,
            max_iter=logreg_max_iter,
            solver="lbfgs",
            class_weight="balanced",
            random_state=random_state,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        ),
    }


def evaluate_model(model: object, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, object]:
    """Calcula as metricas principais de classificacao no conjunto de teste."""
    predictions = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision_macro": precision_score(y_test, predictions, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, predictions, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, predictions, average="macro", zero_division=0),
        "classification_report": classification_report(y_test, predictions, zero_division=0),
    }


def choose_best_model(results: dict[str, dict[str, object]]) -> str:
    """Seleciona o melhor modelo priorizando F1 macro e depois accuracy."""
    ranking = sorted(
        results.items(),
        key=lambda item: (item[1]["f1_macro"], item[1]["accuracy"]),
        reverse=True,
    )
    return ranking[0][0]


def build_metrics_dataframe(results: dict[str, dict[str, object]]) -> pd.DataFrame:
    """Cria uma tabela consolidada com as metricas dos modelos."""
    return pd.DataFrame(
        [
            {
                "Modelo": model_name,
                "Accuracy": metrics["accuracy"],
                "Precision Macro": metrics["precision_macro"],
                "Recall Macro": metrics["recall_macro"],
                "F1 Macro": metrics["f1_macro"],
            }
            for model_name, metrics in results.items()
        ]
    )


def plot_model_comparison(metrics_df: pd.DataFrame, figures_dir: Path) -> Path:
    """Gera um grafico comparando as metricas dos modelos avaliados."""
    plot_df = metrics_df.set_index("Modelo")
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_df.plot(kind="bar", ax=ax, color=["#1d4ed8", "#0f766e", "#d97706", "#7c3aed"])
    ax.set_title("Comparacao de metricas dos modelos", fontsize=14, pad=14)
    ax.set_xlabel("Modelo")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Metrica")
    plt.xticks(rotation=0)
    fig.tight_layout()

    output_path = figures_dir / "model_metrics_comparison.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_report(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    results: dict[str, dict[str, object]],
    best_model_name: str,
    comparison_figure_path: Path,
    experiment_name: str,
    tracking_dir: Path,
    model_config: dict[str, object],
) -> str:
    """Monta o relatorio de treinamento em HTML."""
    best_metrics = results[best_model_name]
    sections = [
        {
            "title": "Visao geral",
            "blocks": [
                {
                    "type": "metrics_grid",
                    "items": [
                        {
                            "label": "Shape do treino",
                            "value": train_df.shape,
                            "description": "Dimensao do conjunto usado para ajuste dos modelos.",
                        },
                        {
                            "label": "Shape do teste",
                            "value": test_df.shape,
                            "description": "Dimensao do conjunto usado na avaliacao final.",
                        },
                        {
                            "label": "Coluna alvo",
                            "value": TARGET_COLUMN,
                            "description": "Variavel prevista pelos classificadores.",
                        },
                        {
                            "label": "Modelos avaliados",
                            "value": "LogisticRegression e RandomForest",
                            "description": "Comparacao entre baseline linear e modelo nao linear.",
                        },
                        {
                            "label": "Config atual",
                            "value": (
                                f"logreg_c={model_config['logreg_c']}, "
                                f"logreg_max_iter={model_config['logreg_max_iter']}, "
                                f"rf_n_estimators={model_config['rf_n_estimators']}, "
                                f"rf_max_depth={model_config['rf_max_depth']}"
                            ),
                            "description": "Hiperparametros usados neste run para facilitar a comparacao.",
                        },
                    ],
                }
            ],
        },
        {
            "title": "Comparacao de metricas",
            "blocks": [
                {
                    "type": "table",
                    "data": metrics_df.round(4),
                },
                {
                    "type": "figure",
                    "src": f"figures/{comparison_figure_path.name}",
                    "alt": "Comparacao de metricas dos modelos",
                    "caption": "Comparacao visual entre Accuracy, Precision Macro, Recall Macro e F1 Macro.",
                },
            ],
        },
        {
            "title": "Melhor modelo",
            "blocks": [
                {
                    "type": "text",
                    "format": "paragraph",
                    "content": [
                        f"O melhor modelo foi `{best_model_name}`.",
                        (
                            "A escolha foi feita priorizando o `F1 macro`, porque a variavel alvo "
                            "e desbalanceada e essa metrica representa melhor o desempenho medio "
                            "entre todas as classes. Em caso de empate, a `accuracy` foi usada "
                            "como criterio de desempate."
                        ),
                        (
                            f"O modelo selecionado obteve F1 macro de `{best_metrics['f1_macro']:.4f}` "
                            f"e accuracy de `{best_metrics['accuracy']:.4f}`."
                        ),
                    ],
                }
            ],
        },
        {
            "title": "Tracking de experimentos",
            "blocks": [
                {
                    "type": "text",
                    "format": "paragraph",
                    "content": [
                        f"O experimento foi registrado no MLflow com o nome `{experiment_name}`.",
                        (
                            f"O backend local de tracking foi salvo em `{tracking_dir.as_posix()}` e o run "
                            "inclui parametros dos modelos, metricas de teste, o relatorio HTML e o artefato "
                            "do melhor modelo."
                        ),
                    ],
                }
            ],
        },
        {
            "title": "Classification report do melhor modelo",
            "blocks": [
                {
                    "type": "text",
                    "format": "pre",
                    "content": best_metrics["classification_report"],
                }
            ],
        },
    ]

    return build_structured_report(
        title="Relatorio de Treinamento de Modelos",
        subtitle="Comparacao entre modelos de classificacao treinados sobre os dados processados.",
        sections=sections,
    )


def save_model_artifact(
    model_name: str,
    model: object,
    feature_columns: list[str],
    artifacts_dir: Path,
    filename: str = "model.joblib",
) -> Path:
    """Salva o melhor modelo com metadados uteis para a proxima etapa."""
    ensure_dir(artifacts_dir)
    model_path = artifacts_dir / filename
    joblib.dump(
        {
            "model_name": model_name,
            "model": model,
            "feature_columns": feature_columns,
            "target_column": TARGET_COLUMN,
        },
        model_path,
    )
    return model_path


def build_deploy_models(random_state: int) -> dict[str, object]:
    """Monta variantes leves do modelo para deploy no Render."""
    return {
        "deploy_logreg.joblib": LogisticRegression(
            C=1.0,
            max_iter=2000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=random_state,
        ),
        "deploy_rf_compacto.joblib": RandomForestClassifier(
            n_estimators=DEPLOY_RF_N_ESTIMATORS,
            max_depth=DEPLOY_RF_MAX_DEPTH,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        ),
    }


def save_deploy_artifacts(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    artifacts_dir: Path,
    random_state: int,
) -> dict[str, Path]:
    """Treina e salva artefatos menores dedicados ao deploy."""
    deploy_paths: dict[str, Path] = {}
    deploy_models = build_deploy_models(random_state=random_state)

    for filename, model in deploy_models.items():
        model.fit(X_train, y_train)
        model_name = "LogisticRegression" if "logreg" in filename else "RandomForestCompacto"
        deploy_paths[filename] = save_model_artifact(
            model_name=model_name,
            model=model,
            feature_columns=X_train.columns.tolist(),
            artifacts_dir=artifacts_dir,
            filename=filename,
        )

    return deploy_paths


def save_report(report_content: str, reports_dir: Path) -> Path:
    """Salva o relatorio de metricas em HTML."""
    ensure_dir(reports_dir)
    report_path = reports_dir / "model_report.html"
    report_path.write_text(report_content, encoding="utf-8")
    return report_path


def run_training(
    data_dir: Path,
    artifacts_dir: Path,
    reports_dir: Path,
    random_state: int,
    tracking_dir: Path,
    experiment_name: str,
    logreg_c: float = 1.0,
    logreg_max_iter: int = 2000,
    rf_n_estimators: int = 300,
    rf_max_depth: int | None = None,
) -> tuple[Path, Path, str, dict[str, dict[str, object]], Path]:
    """Executa o fluxo completo de treino, avaliacao, persistencia e tracking."""
    train_df, test_df = load_splits(data_dir)
    X_train, y_train = split_xy(train_df)
    X_test, y_test = split_xy(test_df)

    model_config = {
        "logreg_c": logreg_c,
        "logreg_max_iter": logreg_max_iter,
        "rf_n_estimators": rf_n_estimators,
        "rf_max_depth": rf_max_depth,
    }
    models = build_models(
        random_state=random_state,
        logreg_c=logreg_c,
        logreg_max_iter=logreg_max_iter,
        rf_n_estimators=rf_n_estimators,
        rf_max_depth=rf_max_depth,
    )
    results: dict[str, dict[str, object]] = {}
    fitted_models: dict[str, object] = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        fitted_models[model_name] = model
        results[model_name] = evaluate_model(model, X_test, y_test)

    best_model_name = choose_best_model(results)
    best_model = fitted_models[best_model_name]
    metrics_df = build_metrics_dataframe(results)
    figures_dir = ensure_figures_dir(reports_dir)
    comparison_figure_path = plot_model_comparison(metrics_df, figures_dir)

    report_content = build_report(
        train_df=train_df,
        test_df=test_df,
        metrics_df=metrics_df,
        results=results,
        best_model_name=best_model_name,
        comparison_figure_path=comparison_figure_path,
        experiment_name=experiment_name,
        tracking_dir=tracking_dir,
        model_config=model_config,
    )
    report_path = save_report(report_content, reports_dir)
    model_path = save_model_artifact(
        model_name=best_model_name,
        model=best_model,
        feature_columns=X_train.columns.tolist(),
        artifacts_dir=artifacts_dir,
    )
    save_deploy_artifacts(
        X_train=X_train,
        y_train=y_train,
        artifacts_dir=artifacts_dir,
        random_state=random_state,
    )

    tracking_dir = log_training_experiment(
        tracking_dir=tracking_dir,
        experiment_name=experiment_name,
        random_state=random_state,
        target_column=TARGET_COLUMN,
        train_df=train_df,
        test_df=test_df,
        fitted_models=fitted_models,
        results=results,
        best_model_name=best_model_name,
        model_path=model_path,
        report_path=report_path,
        comparison_figure_path=comparison_figure_path,
    )

    return model_path, report_path, best_model_name, results, tracking_dir


def main() -> None:
    """Ponto de entrada do script de treinamento."""
    args = parse_args()
    model_path, report_path, best_model_name, results, tracking_dir = run_training(
        data_dir=Path(args.data_dir),
        artifacts_dir=Path(args.artifacts),
        reports_dir=Path(args.reports),
        random_state=args.random_state,
        tracking_dir=Path(args.tracking_dir),
        experiment_name=args.experiment_name,
        logreg_c=args.logreg_c,
        logreg_max_iter=args.logreg_max_iter,
        rf_n_estimators=args.rf_n_estimators,
        rf_max_depth=args.rf_max_depth,
    )

    print(f"Melhor modelo: {best_model_name}")
    print(f"Modelo salvo em: {model_path}")
    print(f"Relatorio salvo em: {report_path}")
    print(f"Tracking MLflow salvo em: {tracking_dir}")
    for model_name, metrics in results.items():
        print(
            f"{model_name}: "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"precision_macro={metrics['precision_macro']:.4f}, "
            f"recall_macro={metrics['recall_macro']:.4f}, "
            f"f1_macro={metrics['f1_macro']:.4f}"
        )


if __name__ == "__main__":
    main()
