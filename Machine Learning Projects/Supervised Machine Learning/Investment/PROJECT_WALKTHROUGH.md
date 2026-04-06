# Investment ML Project — Detailed Walkthrough

This document explains every part of the project in plain language: what each script does,
why design decisions were made, and how all the pieces connect into a single working pipeline.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Project Folder Structure](#2-project-folder-structure)
3. [Pipeline Overview](#3-pipeline-overview)
4. [Step 1 — Data Loading (`data_loader.py`)](#4-step-1--data-loading-data_loaderpy)
5. [Step 2 — Feature Engineering (`features.py`)](#5-step-2--feature-engineering-featurespy)
6. [Step 3 — Model Definitions (`train.py`)](#6-step-3--model-definitions-trainpy)
7. [Step 4 — Walk-Forward Evaluation (`evaluate.py`)](#7-step-4--walk-forward-evaluation-evaluatepy)
8. [Step 5 — Pipeline Orchestrator (`run_pipeline.py`)](#8-step-5--pipeline-orchestrator-run_pipelinepy)
9. [Outputs Explained](#9-outputs-explained)
10. [Why Walk-Forward and Not Random Split?](#10-why-walk-forward-and-not-random-split)
11. [Metrics Explained](#11-metrics-explained)
12. [Models Explained](#12-models-explained)
13. [Data Leakage Prevention](#13-data-leakage-prevention)
14. [How to Re-run or Extend the Project](#14-how-to-re-run-or-extend-the-project)
15. [Limitations and Next Steps](#15-limitations-and-next-steps)

---

## 1. Problem Statement

> **Can we predict whether a stock's price will go UP or DOWN tomorrow?**

This is a **binary supervised classification** problem:

- **Input (features):** historical price behaviour of a stock up to and including today.
- **Output (target):** `1` if tomorrow's closing price is higher than today's, `0` if it is lower or equal.
- **Scope:** five large-cap US equities and index ETFs: `SPY`, `QQQ`, `AAPL`, `MSFT`, `NVDA`.
- **Date range:** 2016-01-01 to 2026-01-01 (10 years of daily data).

The project does **not** try to predict exact prices (regression). Predicting direction is a more tractable
first step and maps naturally to real decision-making: buy, hold, or sell.

---

## 2. Project Folder Structure

```
Investment/
│
├── data/
│   ├── raw/                  # Raw OHLCV CSVs downloaded from Yahoo Finance
│   │   ├── SPY_daily.csv
│   │   ├── QQQ_daily.csv
│   │   └── ...
│   └── processed/
│       └── features.csv      # Engineered features and target label for all tickers
│
├── models/                   # Trained model files saved as .joblib
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   ├── extra_trees.joblib
│   ├── gradient_boosting.joblib
│   ├── svm_rbf.joblib
│   └── xgboost.joblib
│
├── results/                  # Charts and evaluation outputs
│   ├── model_metrics.png
│   ├── cumulative_directional_accuracy.png
│   ├── model_comparison.csv
│   └── backtest_predictions.csv
│
├── src/                      # All Python source code
│   ├── data_loader.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   └── run_pipeline.py
│
├── requirements.txt
├── README.md
└── PROJECT_WALKTHROUGH.md    ← this file
```

---

## 3. Pipeline Overview

When you run `run_pipeline.py`, the following five steps execute in sequence:

```
[run_pipeline.py]
        │
        ▼
[1] data_loader.py       →  Download OHLCV data from Yahoo Finance
        │
        ▼
[2] features.py          →  Build lag returns, volatility, momentum, volume, calendar features
                            Attach the TARGET column (next-day direction)
        │
        ▼
[3] evaluate.py          →  Split dates into walk-forward folds
                            On each fold: train all models on past, test on future
                            Collect metrics per model per fold
        │
        ▼
[3b] train.py            →  Model definitions accessed by evaluate.py during fold training
                            Also used at end to fit final models on the full dataset
        │
        ▼
[4] evaluate.py          →  Average metrics across folds, produce summary table
                            Generate two result charts (PNG)
        │
        ▼
[5] Outputs              →  results/model_metrics.png
                            results/cumulative_directional_accuracy.png
                            results/model_comparison.csv
                            results/backtest_predictions.csv
                            models/*.joblib
```

---

## 4. Step 1 — Data Loading (`data_loader.py`)

**Purpose:** Download raw daily stock price data and save it to disk.

### What `_flatten_columns()` does

Yahoo Finance's `yfinance` library sometimes returns a **MultiIndex column** structure when
downloading a single ticker (e.g. `("Adj Close", "SPY")` instead of just `"Adj Close"`).
`_flatten_columns()` detects this and converts any such column to a clean flat name, then
renames it to the canonical form (`Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`).

This makes the rest of the pipeline work reliably regardless of the yfinance version.

### What `download_ohlcv()` does

For each ticker in the list:

1. Calls `yf.download()` to fetch daily OHLCV bars between `start` and `end`.
2. Normalises the columns using `_flatten_columns()`.
3. Keeps only the seven essential columns: `Date`, `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`.
4. Sorts by date and drops any rows with missing values.
5. Adds a `Ticker` column so rows from multiple tickers can be stacked into one DataFrame later.
6. Saves a CSV per ticker to `data/raw/`.

**Why `Adj Close` instead of `Close`?**  
`Adj Close` is the closing price adjusted for stock splits and dividend distributions. Using raw `Close`
would introduce artificial price jumps on ex-dividend dates, which would create misleading return signals.

---

## 5. Step 2 — Feature Engineering (`features.py`)

**Purpose:** Transform raw prices into signals that a model can actually learn from.

All features are calculated **per ticker** using `groupby("Ticker")` so that AAPL's lag returns never
bleed into SPY's lag returns.

### Features created

| Feature | Formula | Why it matters |
|---|---|---|
| `return_1d` … `return_10d` | `Adj Close.pct_change(N)` | Captures recent price momentum at different horizons |
| `volatility_5d` … `volatility_20d` | Rolling std of `return_1d` | High-volatility periods behave differently from calm ones |
| `momentum_5d`, `momentum_10d` | `Adj Close.pct_change(5 or 10)` | Short and medium-term trend signals |
| `volume_change_1d` | `Volume.pct_change(1)` | Unusual volume often precedes price moves |
| `hl_range` | `(High - Low) / Close` | Intraday range as a proxy for uncertainty |
| `dow_0` … `dow_4` | One-hot encoded day of week | Monday and Friday often exhibit distinct market behaviour |

### The target label

```python
next_return = group["Adj Close"].pct_change().shift(-1)
target = (next_return > 0).astype(int)
```

- `pct_change()` computes tomorrow's return relative to today.
- `.shift(-1)` looks one row forward in time, which gives us tomorrow's return as of today's row.
- The target is `1` (UP) if that return is positive, `0` (DOWN or flat) otherwise.

**Leakage safety:** the target for row `t` is derived entirely from the price at `t+1`. All features
for row `t` use data up to and including `t`. There is no use of future information in the features.

### Cleanup

After all features are attached, any row containing `NaN` or `Inf` is dropped. Early rows always have
`NaN` because of the rolling windows (e.g. `volatility_20d` needs 20 rows before it has a value).

---

## 6. Step 3 — Model Definitions (`train.py`)

**Purpose:** Define all six supervised classifiers with their hyperparameters.
Also provides `fit_and_save_final_models()` which re-trains every model on the full dataset
and saves a `.joblib` file for each one.

### Models

#### Logistic Regression
The simplest linear classifier. Wrapped in a scikit-learn `Pipeline` with `StandardScaler`
so that feature scales do not distort the coefficients (critical for a linear model and for SVM).

- `class_weight="balanced"` adjusts for any imbalance between UP and DOWN days.
- `max_iter=2000` allows enough iterations to converge on larger feature sets.

#### Random Forest
An **ensemble of decision trees** where each tree is trained on a random subset of rows
(bootstrap sampling) and each split considers a random subset of features.

- 250 trees; `max_depth=8` prevents individual trees from memorising noise.
- Fast because trees are built in parallel (`n_jobs=-1`).

#### Extra Trees (Extremely Randomised Trees)
Similar to Random Forest but **splits are chosen at random** rather than finding the optimal split.
This adds even more randomness, which often reduces variance and can improve generalisation.

- 300 trees; `max_depth=10`; `min_samples_leaf=3`.

#### Gradient Boosting
Builds trees **sequentially**, where each new tree corrects the errors of the previous ones.
Slower than Random Forest but often produces a more accurate model because it directly minimises
the loss at every stage.

- `learning_rate=0.04` (small, to learn slowly and carefully).
- `max_depth=3` (shallow trees are preferred for boosting to avoid overfitting each step).

#### SVM with RBF Kernel
The Support Vector Machine finds a **maximum-margin hyperplane** between classes. The RBF
(radial basis function) kernel maps the features into a higher-dimensional space, allowing
non-linear decision boundaries.

- Wrapped in a `Pipeline` with `StandardScaler` — SVM is sensitive to feature scaling.
- `probability=True` enables `predict_proba()` so we can compute ROC-AUC.

#### XGBoost
An optimised and regularised implementation of gradient boosting. One of the most consistently
top-performing models on tabular data.

- `n_estimators=250`, `learning_rate=0.05`, `max_depth=4`.
- `reg_lambda=1.0` adds L2 regularisation to prevent overfitting.
- Skipped gracefully with a `try/except` if the `xgboost` package is not installed.

---

## 7. Step 4 — Walk-Forward Evaluation (`evaluate.py`)

**Purpose:** Measure how well each model generalises to **unseen future data** using a
time-series-aware validation strategy.

### Why not random train/test split?

In a standard random split, a model can train on data from 2022 and then be tested on data
from 2019.  For finance data this is illegal: you cannot use future data to predict the past.
This is called **data leakage** and produces optimistically wrong results.

Walk-forward validation respects time: the model always trains on the past and tests on the future.

### How `_time_folds()` builds folds

Given `n_folds = 4` and suppose you have 2,500 unique trading dates:

```
chunk = 2500 // (4 + 1) = 500 dates per chunk

Fold 1:  Train = dates[0:500]     Test = dates[500:1000]
Fold 2:  Train = dates[0:1000]    Test = dates[1000:1500]
Fold 3:  Train = dates[0:1500]    Test = dates[1500:2000]
Fold 4:  Train = dates[0:2000]    Test = dates[2000:2500]
```

The training window grows with each fold (expanding window), which mirrors the reality that
more data is always available as time passes.

### The evaluation loop

For every fold:

1. Slice the feature DataFrame into `train_df` and `test_df` by date.
2. For each model:
   - Fit the model on `X_train, y_train`.
   - Predict probabilities on `X_test`.
   - Threshold at 0.5 to get binary predictions.
   - Record ROC-AUC, F1, Precision, Recall, and Directional Accuracy.
   - Store the predictions per row (Date, Ticker, fold).
3. If a model fails on a particular fold it is skipped without stopping the whole run.

### Aggregation

After all folds, metrics are averaged per model across folds. This gives a stable estimate of
real-world performance rather than a single lucky or unlucky test split.

### Chart generation (`save_result_charts()`)

Two charts are produced automatically after evaluation:

**Chart 1: `model_metrics.png`**  
A grouped bar chart showing ROC-AUC, F1, Precision, Recall, and Directional Accuracy
for every model side by side. Makes it easy to compare models at a glance.

**Chart 2: `cumulative_directional_accuracy.png`**  
A line chart showing how each model's cumulative directional accuracy evolves day by day
across the full backtest window. A model that starts lucky but degrades over time is easy to spot here.

---

## 8. Step 5 — Pipeline Orchestrator (`run_pipeline.py`)

**Purpose:** Glue all steps together in a single executable entry point.

```
python src/run_pipeline.py --tickers SPY QQQ AAPL MSFT NVDA \
                           --start 2016-01-01 \
                           --end   2026-01-01 \
                           --folds 4
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--tickers` | `SPY QQQ AAPL MSFT NVDA` | List of ticker symbols to include |
| `--start` | `2016-01-01` | Start date for historical data |
| `--end` | `2026-01-01` | End date for historical data |
| `--folds` | `4` | Number of walk-forward folds |

### Execution order inside `main()`

1. Resolve the project root directory regardless of where the script is called from.
2. Create any missing output directories (`data/processed`, `results`).
3. Call `download_ohlcv()` to fetch and save raw data.
4. Concatenate all tickers into one DataFrame and call `build_features()`.
5. Save the full feature set to `data/processed/features.csv`.
6. Call `walk_forward_evaluate()` to benchmark all models.
7. Save `results/model_comparison.csv` and `results/backtest_predictions.csv`.
8. Call `save_result_charts()` to produce the PNG charts.
9. Call `fit_and_save_final_models()` to train on the full dataset and save `.joblib` files.
10. Print the summary table and chart paths to the terminal.

---

## 9. Outputs Explained

| File | Format | What it contains |
|---|---|---|
| `data/raw/{TICKER}_daily.csv` | CSV | Raw OHLCV daily bars per ticker |
| `data/processed/features.csv` | CSV | All engineered features and target for every ticker-date row |
| `results/model_comparison.csv` | CSV | Average walk-forward metrics per model, sorted by ROC-AUC |
| `results/backtest_predictions.csv` | CSV | Every prediction made during walk-forward (Date, Ticker, fold, y_true, y_pred, y_prob) |
| `results/model_metrics.png` | PNG | Bar chart: all metrics for all models |
| `results/cumulative_directional_accuracy.png` | PNG | Line chart: how accuracy evolves over time |
| `models/*.joblib` | Binary | Trained model objects, loadable with `joblib.load()` |

---

## 10. Why Walk-Forward and Not Random Split?

| | Random Split | Walk-Forward |
|---|---|---|
| Future data in training? | Yes (leakage) | No (safe) |
| Realistic? | No | Yes |
| Suitable for time series? | No | Yes |
| Risk of overstated metrics? | High | Low |

A random split on a 10-year dataset might put 2023 data in the training set and ask you to predict 2019.
Walk-forward always asks the model to predict a period it has never seen, which is the only honest test.

---

## 11. Metrics Explained

### ROC-AUC (Receiver Operating Characteristic — Area Under Curve)
Measures the model's ability to **rank** UP days above DOWN days across all possible thresholds.
- 0.5 = random guessing (no skill).
- 1.0 = perfect discrimination.
- For financial direction prediction, anything consistently above 0.53 is meaningful.

### F1 Score
The harmonic mean of Precision and Recall. Useful when there is imbalance between UP and DOWN days.
$$F_1 = 2 \cdot \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

### Precision
Of all the days the model predicted UP, what fraction actually went UP?
$$\text{Precision} = \frac{TP}{TP + FP}$$
High precision means fewer false "buy" signals.

### Recall
Of all the days that actually went UP, what fraction did the model correctly flag?
$$\text{Recall} = \frac{TP}{TP + FN}$$
High recall means fewer missed UP days.

### Directional Accuracy
The simplest metric: what percentage of predictions matched the actual direction?
$$\text{Directional Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}}$$
Random guessing gives ~50%. Anything above ~53% consistently is considered noteworthy in finance.

---

## 12. Models Explained

| Model | Type | Key strength | When it works best |
|---|---|---|---|
| Logistic Regression | Linear | Interpretable, fast, stable baseline | When features are linearly separable |
| Random Forest | Tree ensemble (bagging) | Handles noisy features, feature importance | Moderate-to-high feature count |
| Extra Trees | Tree ensemble (extreme random) | Low variance, very fast to train | When overfitting is a concern |
| Gradient Boosting | Tree ensemble (boosting) | Corrects errors iteratively, strong accuracy | Structured tabular data |
| SVM RBF | Kernel method | Non-linear boundaries with few features | Clean, scaled, moderate-size data |
| XGBoost | Optimised boosting | Regularisation, best general-purpose | Most tabular ML competitions |

---

## 13. Data Leakage Prevention

This project enforces three leakage rules:

1. **Target construction:** `target[t]` = sign of return at `t+1`, computed via `.shift(-1)`.
   Features at row `t` never include any data from `t+1` or later.

2. **Walk-forward splits:** test folds are always strictly after training folds in time.
   No random shuffling is applied at any stage.

3. **No look-ahead in rolling windows:** `rolling(N).std()` computes using rows `t-N` to `t-1` only.
   Pandas rolling windows are right-aligned by default, so this is safe.

---

## 14. How to Re-run or Extend the Project

### Re-run with different tickers
```
python src/run_pipeline.py --tickers TSLA AMZN GOOGL --start 2018-01-01 --end 2026-01-01
```

### Add more walk-forward folds for a more stable estimate
```
python src/run_pipeline.py --folds 8
```

### Add a new model
Open `src/train.py` and add an entry to `build_models()`:
```python
from sklearn.neighbors import KNeighborsClassifier
models["knn"] = KNeighborsClassifier(n_neighbors=50)
```
No other file needs to change. The model will automatically appear in evaluation and charts.

### Load a saved model for inference
```python
import joblib, pandas as pd
model = joblib.load("models/xgboost.joblib")
features = pd.read_csv("data/processed/features.csv")
latest = features[features["Ticker"] == "SPY"].tail(1)[feature_cols]
prediction = model.predict(latest)  # 1 = UP, 0 = DOWN
```

---

## 15. Limitations and Next Steps

### Current limitations
- **No macro features:** interest rates, VIX, or CPI data are not included. Adding them could improve context.
- **No transaction costs:** the backtest assumes costless trades. Real commissions and slippage reduce returns.
- **Class imbalance handling is basic:** only `class_weight="balanced"` on logistic regression. SMOTE or
  threshold tuning per model could improve recall on the minority class.
- **Single-step forecast only:** the model predicts only the next day. Multi-day horizon forecasting is not implemented.
- **No hyperparameter search:** models use manually set parameters. A grid or Bayesian search could yield better results.
- **Dataset size:** 10 years × 5 tickers × ~252 trading days ≈ 12,600 rows. Deep learning models would need more.

### Suggested next steps
1. Add macro indicators from FRED (Federal Reserve Economic Data) as features.
2. Implement a per-model threshold tuner using the precision-recall curve on the validation fold.
3. Add a ROC curve chart per model for threshold analysis.
4. Extend to 20+ S&P 500 tickers to increase the training sample.
5. Add feature importance charts for tree-based models (Random Forest, XGBoost).
