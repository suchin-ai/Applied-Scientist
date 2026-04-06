# Investment Supervised ML (One-Day Build)

This project predicts next-day price direction (Up=1, Down=0) using daily market data.

## Scope
- Python scripts only (no notebooks)
- Walk-forward time-series validation (no random split)
- Baseline model: Logistic Regression
- Additional supervised models: Random Forest, Extra Trees, Gradient Boosting, SVM (RBF), XGBoost

## Folder Structure
- `data/raw`: downloaded OHLCV data
- `data/processed`: engineered feature data
- `src`: pipeline scripts
- `results`: metrics and predictions
- `models`: trained model artifacts

## Quick Start
1. Open a terminal in this folder.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Run pipeline:
   - `python src/run_pipeline.py --tickers SPY QQQ AAPL MSFT NVDA --start 2016-01-01 --end 2026-01-01`

## Outputs
- `results/model_metrics.png` (bar chart for ROC-AUC, F1, precision, recall, directional accuracy)
- `results/cumulative_directional_accuracy.png` (line chart over time)
- `results/model_comparison.csv` (optional tabular summary)
- `results/backtest_predictions.csv` (optional detailed predictions)
- `models/logistic_regression.joblib`
- `models/random_forest.joblib`
- `models/extra_trees.joblib`
- `models/gradient_boosting.joblib`
- `models/svm_rbf.joblib`
- `models/xgboost.joblib`

## Metrics
- ROC-AUC
- F1
- Precision
- Recall
- Directional Accuracy

## Notes
- Temporal integrity is enforced through walk-forward folds.
- If xgboost is unavailable, the pipeline still runs with Logistic Regression.
