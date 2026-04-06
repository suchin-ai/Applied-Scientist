# Basic Imports and Libraries
from __future__ import annotations
from pathlib import Path
from typing import List
import pandas as pd
import yfinance as yf

# Clean yfinance DataFrame
# 1. Flatten multiIndex columns
# 2. Collapse ticker-specific columns to canonical  OHLCV names
# 3. Ensure 'Adj Close' is present, filling from 'Close' if missing
def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance columns to a flat schema."""
    if isinstance(df.columns, pd.MultiIndex):
        flat = []
        for col in df.columns:
            name = " ".join([str(x) for x in col if str(x) != ""]).strip()
            flat.append(name)
        df.columns = flat
    canonical = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    rename_map: dict[str, str] = {}
    for col in df.columns:
        for base in canonical:
            if col == base or str(col).startswith(f"{base} "):
                rename_map[str(col)] = base
                break
    if rename_map:
        df.rename(columns=rename_map, inplace=True)
    if "Adj Close" not in df.columns and "Close" in df.columns:
        df["Adj Close"] = df["Close"]
    return df

# Downloads, cleans, standardizes, and saves daily OHLCV data for each ticker
def download_ohlcv(
    tickers: List[str],
    start: str,
    end: str,
    raw_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Download daily OHLCV data for each ticker and save raw CSV files."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    data_map: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
        if df.empty:
            continue
        df = _flatten_columns(df)
        df = df.reset_index()
        if "Date" not in df.columns:
            df.rename(columns={df.columns[0]: "Date"}, inplace=True)
        df = df[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]].copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df.sort_values("Date", inplace=True)
        df.dropna(inplace=True)
        df["Ticker"] = ticker
        out_path = raw_dir / f"{ticker}_daily.csv"
        df.to_csv(out_path, index=False)
        data_map[ticker] = df
    return data_map