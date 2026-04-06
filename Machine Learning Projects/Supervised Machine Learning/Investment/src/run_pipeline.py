# Basic Imports and Libraries
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from data_loader import download_ohlcv
from evaluate import save_result_charts, walk_forward_evaluate
from features import build_features, feature_columns
from train import fit_and_save_final_models

# Main function to run the entire pipeline
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Investment supervised ML pipeline")
    parser.add_argument("--tickers", nargs="+", default=["SPY", "QQQ", "AAPL", "MSFT", "NVDA"])
    parser.add_argument("--start", type=str, default="2016-01-01")
    parser.add_argument("--end", type=str, default="2026-01-01")
    parser.add_argument("--folds", type=int, default=4)
    return parser.parse_args()

# Run the main pipeline
def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    results_dir = project_root / "results"
    model_dir = project_root / "models"
    processed_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    data_map = download_ohlcv(args.tickers, args.start, args.end, raw_dir)
    if not data_map:
        raise RuntimeError("No data downloaded. Check ticker symbols and date range.")
    all_raw = pd.concat(data_map.values(), ignore_index=True)
    feature_df = build_features(all_raw)
    cols = feature_columns(feature_df)
    feature_df.to_csv(processed_dir / "features.csv", index=False)
    summary, predictions = walk_forward_evaluate(feature_df, cols, n_folds=args.folds)
    summary.to_csv(results_dir / "model_comparison.csv", index=False)
    predictions.to_csv(results_dir / "backtest_predictions.csv", index=False)
    save_result_charts(summary, predictions, results_dir)
    fit_and_save_final_models(feature_df, cols, model_dir)
    print("Pipeline complete")
    print(summary.to_string(index=False))
    print(f"Charts saved: {results_dir / 'model_metrics.png'}")
    print(f"Charts saved: {results_dir / 'cumulative_directional_accuracy.png'}")

# Entry point for script execution
if __name__ == "__main__":
    main()