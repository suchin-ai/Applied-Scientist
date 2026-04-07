from __future__ import annotations

import numpy as np
import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create modeling table and a few clinically intuitive ratio features."""
    work = df.copy()

    eps = 1e-9
    work["radius_perimeter_ratio"] = work["mean radius"] / (work["mean perimeter"] + eps)
    work["compactness_smoothness_ratio"] = work["mean compactness"] / (work["mean smoothness"] + eps)
    work["area_radius_ratio"] = work["mean area"] / (work["mean radius"] + eps)
    work["worst_mean_area_ratio"] = work["worst area"] / (work["mean area"] + eps)

    work.replace([np.inf, -np.inf], np.nan, inplace=True)
    work.dropna(inplace=True)

    return work


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != "target"]
