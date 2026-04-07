from __future__ import annotations

import argparse
from pathlib import Path

from data_loader import load_medical_dataset
from evaluate import cross_validate_models, save_result_charts
from features import build_features, feature_columns
from train import fit_and_save_final_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Medical supervised ML pipeline")
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    results_dir = project_root / "results"
    model_dir = project_root / "models"

    processed_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_medical_dataset(raw_dir)
    feature_df = build_features(raw_df)
    cols = feature_columns(feature_df)

    feature_df.to_csv(processed_dir / "features.csv", index=False)

    summary, oof_predictions = cross_validate_models(feature_df, cols, n_folds=args.folds)
    summary.to_csv(results_dir / "model_comparison.csv", index=False)
    oof_predictions.to_csv(results_dir / "oof_predictions.csv", index=False)
    save_result_charts(summary, oof_predictions, results_dir)

    fit_and_save_final_models(feature_df, cols, model_dir)

    print("Pipeline complete")
    print(summary.to_string(index=False))
    print(f"Charts saved: {results_dir / 'model_metrics.png'}")
    print(f"Charts saved: {results_dir / 'roc_curves.png'}")


if __name__ == "__main__":
    main()
