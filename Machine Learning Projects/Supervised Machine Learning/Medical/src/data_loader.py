from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer


def load_medical_dataset(raw_dir: Path) -> pd.DataFrame:
    """Load breast cancer dataset and save an auditable raw CSV export."""
    raw_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data.copy()

    # In sklearn breast-cancer data: 0=malignant, 1=benign. We invert for clarity.
    y_malignant = 1 - dataset.target

    df = X.copy()
    df["target"] = y_malignant

    out_path = raw_dir / "breast_cancer_raw.csv"
    df.to_csv(out_path, index=False)

    return df
