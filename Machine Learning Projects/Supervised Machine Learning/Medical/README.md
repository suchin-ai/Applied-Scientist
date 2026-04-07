# Medical Supervised ML (Python Scripts)

This project predicts whether a tumor is malignant (`1`) or benign (`0`) using supervised machine learning.

## Scope
- Python scripts only (no notebooks)
- Stratified cross-validation for fair evaluation
- Multiple supervised models compared side-by-side
- Auto-generated charts in `results/`

## Folder Structure
- `data/raw`: raw dataset export
- `data/processed`: processed feature table
- `src`: pipeline scripts
- `results`: model metrics and charts
- `models`: trained model files

## Quick Start
1. Open terminal in this folder.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Run pipeline:
   - `python src/run_pipeline.py --folds 5`

## Outputs
- `results/model_metrics.png` (bar chart for ROC-AUC, F1, precision, recall, accuracy)
- `results/roc_curves.png` (ROC curve comparison per model)
- `results/model_comparison.csv`
- `results/oof_predictions.csv`
- `models/*.joblib`

## Notes
- Dataset source: `sklearn.datasets.load_breast_cancer`
- Class target mapping in this project:
  - `1` = malignant (disease present)
  - `0` = benign (disease absent)
