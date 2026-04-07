# Basic Imports and Libraries
from __future__ import annotations
from pathlib import Path
from typing import Dict
import joblib
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Model Building and Training
def build_models() -> Dict[str, object]:
    models: Dict[str, object] = {}
    models["logistic_regression"] = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42)),
        ]
    )
    models["random_forest"] = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    models["extra_trees"] = ExtraTreesClassifier(
        n_estimators=350,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    models["gradient_boosting"] = GradientBoostingClassifier(
        n_estimators=250,
        learning_rate=0.04,
        max_depth=3,
        random_state=42,
    )
    models["svm_rbf"] = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)),
        ]
    )
    try:
        from xgboost import XGBClassifier
        models["xgboost"] = XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            eval_metric="logloss",
            random_state=42,
        )
    except Exception:
        pass
    return models

# Fit each model on the full dataset and save to disk for future inference
def fit_and_save_final_models(
    df_features: pd.DataFrame,
    feature_cols: list[str],
    model_dir: Path,
) -> Dict[str, object]:
    model_dir.mkdir(parents=True, exist_ok=True)
    X = df_features[feature_cols]
    y = df_features["target"]
    fitted: Dict[str, object] = {}
    for name, model in build_models().items():
        model.fit(X, y)
        joblib.dump(model, model_dir / f"{name}.joblib")
        fitted[name] = model
    return fitted