from __future__ import annotations
"""Executa o pipeline completo do projeto em sequência."""

import argparse
from pathlib import Path

from src.eda import resolve_input_path, run_eda
from src.preprocess import run_preprocess
from src.train import run_training


def parse_args() -> argparse.Namespace:
    """Lê os argumentos de linha de comando do pipeline completo."""
    parser = argparse.ArgumentParser(description="Executa EDA, preprocessamento e treinamento.")
    parser.add_argument(
        "--input",
        default="data/dataset_graduacao_indicada.csv",
        help="Arquivo CSV de entrada.",
    )
    parser.add_argument(
        "--reports",
        default="reports",
        help="Diretório onde os relatórios HTML serão salvos.",
    )
    parser.add_argument(
        "--processed",
        default="data/processed",
        help="Diretório onde os datasets processados serão salvos.",
    )
    parser.add_argument(
        "--artifacts",
        default="artifacts",
        help="Diretório onde os artefatos serão salvos.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proporção do conjunto de teste no preprocessamento.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed fixa para reprodutibilidade.",
    )
    return parser.parse_args()


def main() -> None:
    """Executa todas as etapas do pipeline do projeto."""
    args = parse_args()
    input_path = resolve_input_path(args.input)
    reports_dir = Path(args.reports)
    processed_dir = Path(args.processed)
    artifacts_dir = Path(args.artifacts)

    reports_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    eda_report_path, figure_paths = run_eda(input_path=input_path, output_dir=reports_dir)
    (
        preprocess_pipeline_path,
        train_dataset_path,
        test_dataset_path,
        preprocess_report_path,
        train_shape,
        test_shape,
    ) = run_preprocess(
        input_path=input_path,
        output_dir=processed_dir,
        artifacts_dir=artifacts_dir,
        reports_dir=reports_dir,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    model_path, model_report_path, best_model_name, results = run_training(
        data_dir=processed_dir,
        artifacts_dir=artifacts_dir,
        reports_dir=reports_dir,
        random_state=args.random_state,
    )

    print("Pipeline concluído com sucesso.")
    print(f"Relatório EDA: {eda_report_path}")
    print(f"Figuras EDA: {len(figure_paths)} arquivos")
    print(f"Pipeline de preprocessamento: {preprocess_pipeline_path}")
    print(f"Treino processado: {train_dataset_path} | shape={train_shape}")
    print(f"Teste processado: {test_dataset_path} | shape={test_shape}")
    print(f"Relatório de preprocessamento: {preprocess_report_path}")
    print(f"Melhor modelo: {best_model_name}")
    print(f"Artefato do modelo: {model_path}")
    print(f"Relatório de treinamento: {model_report_path}")
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
