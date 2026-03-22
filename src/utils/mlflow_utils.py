from __future__ import annotations
"""Helpers para tracking de experimentos com MLflow."""

from pathlib import Path

import mlflow
import pandas as pd


def configure_mlflow_tracking(tracking_dir: Path, experiment_name: str) -> Path:
    """Configura o MLflow para usar tracking local em disco."""
    tracking_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_dir.resolve().as_uri())
    mlflow.set_experiment(experiment_name)
    return tracking_dir


def extract_model_params(model: object) -> dict[str, object]:
    """Extrai os parametros relevantes do estimador para log no MLflow."""
    if not hasattr(model, "get_params"):
        return {}

    params = model.get_params()
    selected_keys = (
        "C",
        "class_weight",
        "max_depth",
        "max_iter",
        "n_estimators",
        "n_jobs",
        "random_state",
        "solver",
    )
    return {key: params[key] for key in selected_keys if key in params}


def log_training_experiment(
    tracking_dir: Path,
    experiment_name: str,
    random_state: int,
    target_column: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fitted_models: dict[str, object],
    results: dict[str, dict[str, object]],
    best_model_name: str,
    model_path: Path,
    report_path: Path,
    comparison_figure_path: Path,
) -> Path:
    """Registra parametros, metricas e artefatos do treino no MLflow."""
    configure_mlflow_tracking(tracking_dir=tracking_dir, experiment_name=experiment_name)
    best_metrics = results[best_model_name]

    with mlflow.start_run(run_name=f"treino-{best_model_name.lower()}"):
        mlflow.log_param("target_column", target_column)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("train_rows", int(train_df.shape[0]))
        mlflow.log_param("train_columns", int(train_df.shape[1]))
        mlflow.log_param("test_rows", int(test_df.shape[0]))
        mlflow.log_param("test_columns", int(test_df.shape[1]))
        mlflow.log_param("candidate_models", ", ".join(fitted_models.keys()))
        mlflow.log_param("best_model_name", best_model_name)

        for model_name, model in fitted_models.items():
            for param_name, param_value in extract_model_params(model).items():
                mlflow.log_param(f"{model_name.lower()}_{param_name}", param_value)

        for model_name, metrics in results.items():
            metric_prefix = model_name.lower()
            mlflow.log_metric(f"{metric_prefix}_accuracy", float(metrics["accuracy"]))
            mlflow.log_metric(f"{metric_prefix}_precision_macro", float(metrics["precision_macro"]))
            mlflow.log_metric(f"{metric_prefix}_recall_macro", float(metrics["recall_macro"]))
            mlflow.log_metric(f"{metric_prefix}_f1_macro", float(metrics["f1_macro"]))

        mlflow.log_metric("best_accuracy", float(best_metrics["accuracy"]))
        mlflow.log_metric("best_precision_macro", float(best_metrics["precision_macro"]))
        mlflow.log_metric("best_recall_macro", float(best_metrics["recall_macro"]))
        mlflow.log_metric("best_f1_macro", float(best_metrics["f1_macro"]))

        mlflow.log_artifact(str(report_path), artifact_path="reports")
        mlflow.log_artifact(str(comparison_figure_path), artifact_path="reports/figures")
        mlflow.log_artifact(str(model_path), artifact_path="model")

    return tracking_dir
