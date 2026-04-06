# Basic Imports and Libraries
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from train import build_models

# Container for evaluation metrics from a single cross-validation fold
@dataclass
class FoldResult:
    model_name: str
    fold: int
    roc_auc: float
    f1: float
    precision: float
    recall: float
    directional_accuracy: float

# Generate time-based train/test splits for walk-forward validation
def _time_folds(unique_dates: np.ndarray, n_folds: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    chunk = len(unique_dates) // (n_folds + 1)
    if chunk < 30:
        raise ValueError("Not enough data for stable walk-forward folds. Increase date range.")
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(1, n_folds + 1):
        train_end_idx = chunk * i
        test_end_idx = chunk * (i + 1)
        train_dates = unique_dates[:train_end_idx]
        test_dates = unique_dates[train_end_idx:test_end_idx]
        folds.append((train_dates, test_dates))
    return folds

# Perform walk-forward evaluations accross multiple models and collect metrics and predictions
def walk_forward_evaluate(
    df_features: pd.DataFrame,
    feature_cols: list[str],
    n_folds: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_dates = np.sort(df_features["Date"].unique())
    folds = _time_folds(unique_dates, n_folds=n_folds)
    fold_rows: list[FoldResult] = []
    pred_rows: list[pd.DataFrame] = []
    for fold_idx, (train_dates, test_dates) in enumerate(folds, start=1):
        train_df = df_features[df_features["Date"].isin(train_dates)]
        test_df = df_features[df_features["Date"].isin(test_dates)]
        X_train = train_df[feature_cols]
        y_train = train_df["target"]
        X_test = test_df[feature_cols]
        y_test = test_df["target"]
        for model_name, model in build_models().items():
            try:
                model.fit(X_train, y_train)
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X_test)[:, 1]
                else:
                    y_prob = model.predict(X_test)
                y_pred = (y_prob >= 0.5).astype(int)
                row = FoldResult(
                    model_name=model_name,
                    fold=fold_idx,
                    roc_auc=float(roc_auc_score(y_test, y_prob)),
                    f1=float(f1_score(y_test, y_pred, zero_division=0)),
                    precision=float(precision_score(y_test, y_pred, zero_division=0)),
                    recall=float(recall_score(y_test, y_pred, zero_division=0)),
                    directional_accuracy=float((y_pred == y_test).mean()),
                )
                fold_rows.append(row)
                pred_rows.append(
                    pd.DataFrame(
                        {
                            "Date": test_df["Date"].values,
                            "Ticker": test_df["Ticker"].values,
                            "model": model_name,
                            "y_true": y_test.values,
                            "y_pred": y_pred,
                            "y_prob": y_prob,
                            "fold": fold_idx,
                        }
                    )
                )
            except Exception as exc:
                print(f"Skipping {model_name} on fold {fold_idx}: {exc}")
                continue
    if not fold_rows:
        raise RuntimeError("No model evaluations completed successfully.")
    fold_metrics = pd.DataFrame([r.__dict__ for r in fold_rows])
    summary = (
        fold_metrics.groupby("model_name", as_index=False)[
            ["roc_auc", "f1", "precision", "recall", "directional_accuracy"]
        ]
        .mean()
        .sort_values("roc_auc", ascending=False)
    )
    predictions = pd.concat(pred_rows, ignore_index=True)
    return summary, predictions

# Save visual summaries so users can inspect outcomes without opening CSV files
def save_result_charts(summary: pd.DataFrame, predictions: pd.DataFrame, results_dir: Path) -> None:
    """Save visual summaries so users can inspect outcomes without opening CSV files."""
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_to_plot = ["roc_auc", "f1", "precision", "recall", "directional_accuracy"]
    chart_data = summary[["model_name", *metrics_to_plot]].set_index("model_name")
    plt.figure(figsize=(10, 6))
    chart_data.plot(kind="bar", ylim=(0.0, 1.0), rot=0)
    plt.title("Model Comparison (Walk-Forward Average Metrics)")
    plt.ylabel("Score")
    plt.xlabel("Model")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "model_metrics.png", dpi=180)
    plt.close()
    pred = predictions.copy()
    pred["Date"] = pd.to_datetime(pred["Date"])
    pred["correct"] = (pred["y_pred"] == pred["y_true"]).astype(int)
    daily_acc = (
        pred.groupby(["Date", "model"], as_index=False)["correct"]
        .mean()
        .sort_values(["model", "Date"])
    )
    daily_acc["cum_directional_accuracy"] = daily_acc.groupby("model")["correct"].transform(
        lambda s: s.expanding().mean()
    )
    plt.figure(figsize=(11, 6))
    for model_name, group_df in daily_acc.groupby("model"):
        plt.plot(group_df["Date"], group_df["cum_directional_accuracy"], label=model_name)
    plt.title("Cumulative Directional Accuracy Over Time")
    plt.ylabel("Cumulative Accuracy")
    plt.xlabel("Date")
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "cumulative_directional_accuracy.png", dpi=180)
    plt.close()