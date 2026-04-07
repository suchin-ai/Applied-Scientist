from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold

from train import build_models


@dataclass
class FoldResult:
    model_name: str
    fold: int
    roc_auc: float
    f1: float
    precision: float
    recall: float
    accuracy: float


def cross_validate_models(
    df_features: pd.DataFrame,
    feature_cols: list[str],
    n_folds: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = df_features[feature_cols]
    y = df_features["target"]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_rows: list[FoldResult] = []
    pred_rows: list[pd.DataFrame] = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        for model_name, model in build_models().items():
            try:
                model.fit(X_train, y_train)
                y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_test)
                y_pred = (y_prob >= 0.5).astype(int)

                fold_rows.append(
                    FoldResult(
                        model_name=model_name,
                        fold=fold_idx,
                        roc_auc=float(roc_auc_score(y_test, y_prob)),
                        f1=float(f1_score(y_test, y_pred, zero_division=0)),
                        precision=float(precision_score(y_test, y_pred, zero_division=0)),
                        recall=float(recall_score(y_test, y_pred, zero_division=0)),
                        accuracy=float(accuracy_score(y_test, y_pred)),
                    )
                )

                pred_rows.append(
                    pd.DataFrame(
                        {
                            "row_id": test_idx,
                            "model": model_name,
                            "fold": fold_idx,
                            "y_true": y_test.values,
                            "y_pred": y_pred,
                            "y_prob": y_prob,
                        }
                    )
                )
            except Exception as exc:
                print(f"Skipping {model_name} on fold {fold_idx}: {exc}")

    if not fold_rows:
        raise RuntimeError("No model evaluations completed successfully.")

    fold_metrics = pd.DataFrame([r.__dict__ for r in fold_rows])
    summary = (
        fold_metrics.groupby("model_name", as_index=False)[["roc_auc", "f1", "precision", "recall", "accuracy"]]
        .mean()
        .sort_values("roc_auc", ascending=False)
    )

    oof_predictions = pd.concat(pred_rows, ignore_index=True)
    return summary, oof_predictions


def save_result_charts(summary: pd.DataFrame, oof_predictions: pd.DataFrame, results_dir: Path) -> None:
    """Save comparison and ROC visualizations."""
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics_to_plot = ["roc_auc", "f1", "precision", "recall", "accuracy"]
    chart_data = summary[["model_name", *metrics_to_plot]].set_index("model_name")

    ax = chart_data.plot(kind="bar", figsize=(11, 6), ylim=(0.0, 1.0), rot=0)
    ax.set_title("Medical Model Comparison (Cross-Validation Average)")
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "model_metrics.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 7))
    for model_name, group in oof_predictions.groupby("model"):
        fpr, tpr, _ = roc_curve(group["y_true"], group["y_prob"])
        auc_val = roc_auc_score(group["y_true"], group["y_prob"])
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc_val:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.title("ROC Curves (Out-of-Fold Predictions)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "roc_curves.png", dpi=180)
    plt.close()
