# Medical ML Project - Detailed Walkthrough

This document explains how the Medical project works in detail, from raw dataset loading to
cross-validation, model comparison, chart generation, and final model export.

---

## 1. Problem Statement

The project solves a binary supervised learning task:

- Input: tabular medical features computed from digitized breast mass images.
- Output: tumor class label.
- Label convention used in this project:
  - `1` = malignant
  - `0` = benign

Primary objective:

- Compare several supervised classifiers under the same validation protocol.
- Select robust high-performing models using ROC-AUC, F1, precision, recall, and accuracy.

---

## 2. Dataset and Label Semantics

Source:

- `sklearn.datasets.load_breast_cancer(as_frame=True)`

Important note on labels:

- In scikit-learn's original dataset, labels are:
  - `0` = malignant
  - `1` = benign
- In this project, labels are inverted with:
  - `y_malignant = 1 - dataset.target`

Why invert:

- It aligns labels with common medical-risk interpretation where positive class (`1`) represents
  disease presence.

---

## 3. Folder Structure and Responsibilities

```
Medical/
|-- data/
|   |-- raw/
|   |   `-- breast_cancer_raw.csv
|   `-- processed/
|       `-- features.csv
|-- models/
|   |-- logistic_regression.joblib
|   |-- random_forest.joblib
|   |-- extra_trees.joblib
|   |-- gradient_boosting.joblib
|   |-- svm_rbf.joblib
|   `-- xgboost.joblib
|-- results/
|   |-- model_metrics.png
|   |-- roc_curves.png
|   |-- model_comparison.csv
|   `-- oof_predictions.csv
`-- src/
    |-- data_loader.py
    |-- features.py
    |-- train.py
    |-- evaluate.py
    `-- run_pipeline.py
```

Design principle:

- Keep data loading, feature engineering, modeling, evaluation, and orchestration in separate
  modules so each part can be changed independently.

---

## 4. End-to-End Pipeline Flow

Pipeline entry point:

- `src/run_pipeline.py`

Execution sequence:

1. Load raw medical data (`data_loader.py`).
2. Build engineered features (`features.py`).
3. Run stratified cross-validation across all models (`evaluate.py`).
4. Save quantitative results (`model_comparison.csv`, `oof_predictions.csv`).
5. Save visual diagnostics (`model_metrics.png`, `roc_curves.png`).
6. Fit each model on the full dataset and save trained artifacts (`train.py`).

---

## 5. Data Loading Logic (`src/data_loader.py`)

Main function:

- `load_medical_dataset(raw_dir: Path) -> pd.DataFrame`

What it does:

1. Creates `data/raw` if missing.
2. Loads dataset as a DataFrame (`X`) with human-readable feature names.
3. Inverts labels so malignant is positive class.
4. Combines features and `target` into one DataFrame.
5. Saves a raw export to `data/raw/breast_cancer_raw.csv`.
6. Returns the DataFrame to the pipeline.

Why save raw CSV even for built-in datasets:

- Reproducibility and auditability.
- Easier debugging and quick spot checks outside Python.

---

## 6. Feature Engineering (`src/features.py`)

Main function:

- `build_features(df: pd.DataFrame) -> pd.DataFrame`

The base dataset is already fairly engineered, but this project adds four ratio features to capture
relative shape and texture relationships:

1. `radius_perimeter_ratio` = `mean radius / mean perimeter`
2. `compactness_smoothness_ratio` = `mean compactness / mean smoothness`
3. `area_radius_ratio` = `mean area / mean radius`
4. `worst_mean_area_ratio` = `worst area / mean area`

Numerical safety:

- Uses `eps = 1e-9` to avoid division-by-zero issues.
- Replaces `+/- inf` with `NaN`, then drops rows with missing values.

Feature selection function:

- `feature_columns(df)` returns every column except `target`.

This keeps training code generic and resilient if new features are added later.

---

## 7. Model Suite (`src/train.py`)

The project benchmarks multiple supervised classifiers under a common setup:

1. Logistic Regression (with scaling and class balancing)
2. Random Forest
3. Extra Trees
4. Gradient Boosting
5. SVM with RBF kernel (with scaling)
6. XGBoost (if available in environment)

Why these models:

- Linear baseline: Logistic Regression
- Bagging-based nonlinear learners: Random Forest, Extra Trees
- Boosting-based nonlinear learners: Gradient Boosting, XGBoost
- Margin-based kernel method: SVM-RBF

This set gives a broad comparison across major supervised modeling families.

Final training/export:

- `fit_and_save_final_models()` trains each model on all processed rows and writes `.joblib`
  files in `models/`.

---

## 8. Validation Strategy (`src/evaluate.py`)

Method:

- Stratified K-Fold cross-validation (`StratifiedKFold`) with `shuffle=True`, `random_state=42`.

Why stratified splits:

- Maintains similar malignant/benign class proportions in each fold.
- Reduces variance and protects against misleading fold-level imbalance.

Per-fold evaluation loop:

1. Split `X, y` into train and validation indices.
2. Fit each model on fold-train data.
3. Get probabilities (`predict_proba`) when available.
4. Convert to class predictions with threshold 0.5.
5. Compute metrics and append fold results.
6. Store out-of-fold predictions for global diagnostics.

Fault tolerance:

- If a model fails on a fold, it is skipped and the run continues.
- If all models fail, a runtime error is raised.

---

## 9. Metrics and What They Mean

The project reports five metrics per model (averaged across folds):

1. ROC-AUC
   - Measures ranking quality across all thresholds.
   - Useful in medical risk settings where threshold may change by policy.

2. F1 Score
   - Harmonic mean of precision and recall.
   - Good summary when false positives and false negatives both matter.

3. Precision
   - Of predicted malignant cases, how many are truly malignant.

4. Recall (Sensitivity)
   - Of actual malignant cases, how many were correctly detected.

5. Accuracy
   - Overall proportion of correct predictions.

Clinical emphasis:

- Recall is especially important when missing malignant cases is costly.
- Precision matters to reduce unnecessary alarms and follow-up procedures.

---

## 10. Out-of-Fold Predictions and ROC Curves

`oof_predictions.csv` stores predictions on validation data only (never trained-on rows).

Why this is important:

- It gives a realistic estimate of performance.
- It supports model-level ROC curves from the same unbiased prediction pool.

`roc_curves.png` is built from these out-of-fold probabilities and shows:

- Trade-off between true positive rate and false positive rate.
- Model discrimination quality beyond a fixed threshold.

---

## 11. Chart Outputs (`save_result_charts`)

Two automatic result visualizations are produced:

1. `results/model_metrics.png`
   - Grouped bar chart across ROC-AUC, F1, precision, recall, and accuracy.
   - Best for quick side-by-side model ranking.

2. `results/roc_curves.png`
   - ROC line per model plus random baseline diagonal.
   - Best for threshold-independent discrimination analysis.

These charts make results presentation-ready without opening CSV files.

---

## 12. Orchestration and CLI (`src/run_pipeline.py`)

Run command:

```
python src/run_pipeline.py --folds 5
```

Arguments:

- `--folds`: number of stratified CV folds (default 5).

What gets created each run:

- `data/raw/breast_cancer_raw.csv`
- `data/processed/features.csv`
- `results/model_comparison.csv`
- `results/oof_predictions.csv`
- `results/model_metrics.png`
- `results/roc_curves.png`
- `models/*.joblib`

Console output includes:

- Sorted summary table by ROC-AUC.
- Absolute chart paths for quick opening.

---

## 13. Reproducibility Controls

Reproducibility measures used in the project:

- Fixed random seeds where supported (`random_state=42`).
- Deterministic dataset source (`load_breast_cancer`).
- Explicit train/eval separation via out-of-fold predictions.
- Saved processed dataset and trained model artifacts.

Potential non-determinism still possible:

- Some libraries and parallel routines can introduce small numeric differences across platforms.

---

## 14. Current Strengths and Limitations

Strengths:

- Clean modular architecture.
- Multiple supervised model families compared consistently.
- Strong validation protocol for this type of tabular dataset.
- Visual + tabular reporting included by default.

Limitations:

- Uses a benchmark dataset, not hospital EHR data.
- No probability calibration step (Platt/Isotonic) yet.
- No fairness slices by subgroup (age bands, etc.) because dataset fields are limited.
- No threshold optimization by clinical objective yet.

---

## 15. Recommended Next Improvements

1. Add calibrated probabilities and calibration plots.
2. Add confusion matrix and precision-recall curve chart outputs.
3. Introduce threshold tuning based on recall target.
4. Add permutation feature importance and SHAP-style explainability.
5. Add external dataset validation to test generalization.

---

## 16. Quick Troubleshooting

If `xgboost` model is missing in outputs:

- Install dependencies with `pip install -r requirements.txt`.
- Re-run `python src/run_pipeline.py --folds 5`.

If charts are not generated:

- Confirm `matplotlib` is installed.
- Confirm run ended with `Pipeline complete`.
- Check `results/` folder for PNG files.

If results look unexpectedly perfect or poor:

- Verify label mapping is still `1 = malignant`.
- Verify metrics are read from `results/model_comparison.csv` after latest run.
