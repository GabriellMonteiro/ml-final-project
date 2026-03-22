from __future__ import annotations
"""Script de treinamento e avaliação de modelos de classificação.

Treina dois modelos, compara métricas no conjunto de teste e salva o melhor
artefato para as próximas etapas do pipeline.
"""

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

from src.utils.report_utils import build_structured_report

TARGET_COLUMN = "Graduacao_Indicada"


def parse_args() -> argparse.Namespace:
    """Lê os argumentos de linha de comando do script de treino."""
    parser = argparse.ArgumentParser(description="Executa o treinamento e a avaliação dos modelos.")
    parser.add_argument(
        "--data_dir",
        default="data/processed",
        help="Diretório com os arquivos train.parquet e test.parquet.",
    )
    parser.add_argument(
        "--artifacts",
        default="artifacts",
        help="Diretório onde o melhor modelo será salvo.",
    )
    parser.add_argument(
        "--reports",
        default="reports",
        help="Diretório onde o relatório de métricas será salvo.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed fixa para reprodutibilidade.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    """Cria o diretório caso ele ainda não exista."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_figures_dir(reports_dir: Path) -> Path:
    """Garante a existência da pasta de figuras do relatório."""
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir


def load_splits(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega os datasets processados de treino e teste."""
    train_path = data_dir / "train.parquet"
    test_path = data_dir / "test.parquet"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Arquivos train.parquet e test.parquet são obrigatórios.")

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    return train_df, test_df


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa as features da coluna alvo."""
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def build_models(random_state: int) -> dict[str, object]:
    """Monta os modelos exigidos pelo escopo da atividade."""
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=random_state,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        ),
    }


def evaluate_model(model: object, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, object]:
    """Calcula as métricas principais de classificação no conjunto de teste."""
    predictions = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision_macro": precision_score(y_test, predictions, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, predictions, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, predictions, average="macro", zero_division=0),
        "classification_report": classification_report(y_test, predictions, zero_division=0),
    }


def choose_best_model(results: dict[str, dict[str, object]]) -> str:
    """Seleciona o melhor modelo priorizando F1 macro e depois accuracy.

    O F1 macro é priorizado porque o problema possui desbalanceamento entre classes.
    """
    ranking = sorted(
        results.items(),
        key=lambda item: (item[1]["f1_macro"], item[1]["accuracy"]),
        reverse=True,
    )
    return ranking[0][0]


def build_metrics_dataframe(results: dict[str, dict[str, object]]) -> pd.DataFrame:
    """Cria uma tabela consolidada com as métricas dos modelos."""
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
    """Gera um gráfico comparando as métricas dos modelos avaliados."""
    plot_df = metrics_df.set_index("Modelo")
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_df.plot(kind="bar", ax=ax, color=["#1d4ed8", "#0f766e", "#d97706", "#7c3aed"])
    ax.set_title("Comparação de métricas dos modelos", fontsize=14, pad=14)
    ax.set_xlabel("Modelo")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Métrica")
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
) -> str:
    """Monta o relatório de treinamento em HTML."""
    best_metrics = results[best_model_name]
    sections = [
        {
            "title": "Visão geral",
            "blocks": [
                {
                    "type": "metrics_grid",
                    "items": [
                        {"label": "Shape do treino", "value": train_df.shape},
                        {"label": "Shape do teste", "value": test_df.shape},
                        {"label": "Coluna alvo", "value": TARGET_COLUMN},
                        {"label": "Modelos avaliados", "value": "LogisticRegression e RandomForest"},
                    ],
                }
            ],
        },
        {
            "title": "Comparação de métricas",
            "blocks": [
                {
                    "type": "table",
                    "data": metrics_df.round(4),
                },
                {
                    "type": "figure",
                    "src": f"figures/{comparison_figure_path.name}",
                    "alt": "Comparação de métricas dos modelos",
                    "caption": "Comparação visual entre Accuracy, Precision Macro, Recall Macro e F1 Macro.",
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
                            "A escolha foi feita priorizando o `F1 macro`, porque a variável alvo "
                            "é desbalanceada e essa métrica representa melhor o desempenho médio "
                            "entre todas as classes. Em caso de empate, a `accuracy` foi usada "
                            "como critério de desempate."
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
        title="Relatório de Treinamento de Modelos",
        subtitle="Comparação entre modelos de classificação treinados sobre os dados processados.",
        sections=sections,
    )


def save_model_artifact(
    model_name: str,
    model: object,
    feature_columns: list[str],
    artifacts_dir: Path,
) -> Path:
    """Salva o melhor modelo com metadados úteis para a próxima etapa."""
    ensure_dir(artifacts_dir)
    model_path = artifacts_dir / "model.joblib"
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


def save_report(report_content: str, reports_dir: Path) -> Path:
    """Salva o relatório de métricas em HTML."""
    ensure_dir(reports_dir)
    report_path = reports_dir / "model_report.html"
    report_path.write_text(report_content, encoding="utf-8")
    return report_path


def run_training(
    data_dir: Path,
    artifacts_dir: Path,
    reports_dir: Path,
    random_state: int,
) -> tuple[Path, Path, str, dict[str, dict[str, object]]]:
    """Executa o fluxo completo de treino, avaliação e persistência."""
    train_df, test_df = load_splits(data_dir)
    X_train, y_train = split_xy(train_df)
    X_test, y_test = split_xy(test_df)

    models = build_models(random_state=random_state)
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
    )
    report_path = save_report(report_content, reports_dir)
    model_path = save_model_artifact(
        model_name=best_model_name,
        model=best_model,
        feature_columns=X_train.columns.tolist(),
        artifacts_dir=artifacts_dir,
    )

    return model_path, report_path, best_model_name, results


def main() -> None:
    """Ponto de entrada do script de treinamento."""
    args = parse_args()
    model_path, report_path, best_model_name, results = run_training(
        data_dir=Path(args.data_dir),
        artifacts_dir=Path(args.artifacts),
        reports_dir=Path(args.reports),
        random_state=args.random_state,
    )

    print(f"Melhor modelo: {best_model_name}")
    print(f"Modelo salvo em: {model_path}")
    print(f"Relatório salvo em: {report_path}")
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
