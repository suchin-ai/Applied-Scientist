# Basic Imports and Libraries
from __future__ import annotations
import numpy as np
import pandas as pd

# Create leakage-safe supervised features and next-day direction target
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create leakage-safe supervised features and next-day direction target."""
    work = df.copy()
    work.sort_values(["Ticker", "Date"], inplace=True)
    group = work.groupby("Ticker", group_keys=False)
    work["return_1d"] = group["Adj Close"].pct_change(1)
    work["return_2d"] = group["Adj Close"].pct_change(2)
    work["return_3d"] = group["Adj Close"].pct_change(3)
    work["return_5d"] = group["Adj Close"].pct_change(5)
    work["return_10d"] = group["Adj Close"].pct_change(10)
    work["volatility_5d"] = group["return_1d"].rolling(5).std().reset_index(level=0, drop=True)
    work["volatility_10d"] = group["return_1d"].rolling(10).std().reset_index(level=0, drop=True)
    work["volatility_20d"] = group["return_1d"].rolling(20).std().reset_index(level=0, drop=True)
    work["momentum_5d"] = group["Adj Close"].pct_change(5)
    work["momentum_10d"] = group["Adj Close"].pct_change(10)
    work["volume_change_1d"] = group["Volume"].pct_change(1)
    work["hl_range"] = (work["High"] - work["Low"]) / work["Close"].replace(0, np.nan)
    work["day_of_week"] = pd.to_datetime(work["Date"]).dt.dayofweek
    dow = pd.get_dummies(work["day_of_week"], prefix="dow", dtype=int)
    work = pd.concat([work, dow], axis=1)
    next_return = group["Adj Close"].pct_change().shift(-1)
    work["target"] = (next_return > 0).astype(int)
    work.replace([np.inf, -np.inf], np.nan, inplace=True)
    work.dropna(inplace=True)
    return work

# Identify feature columns by excluding known non-feature fields
def feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {
        "Date",
        "Ticker",
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
        "day_of_week",
        "target",
    }
    return [c for c in df.columns if c not in exclude]