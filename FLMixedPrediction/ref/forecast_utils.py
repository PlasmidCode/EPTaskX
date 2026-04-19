
# -*- coding: utf-8 -*-

"""
Utility functions for forecasting wind/solar power generation using the
Chinese State Grid Renewable Energy Generation Forecasting Competition dataset.

Assumptions (match the provided notebooks):
- Processed data are in:
    data_processed/wind_farms/*.xlsx
    data_processed/solar_stations/*.xlsx
- Each file has a header row that should be dropped (drop index=0), and then
  columns are renamed to standard names (see notebooks).
- Time granularity is 15 minutes.

This module focuses on reproducibility and strong tabular baselines (LightGBM).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# 修复列名映射问题：确保Power(MW)总是映射到最后一列
WIND_COLS_BASE = ["time", "WS_10", "WD_10", "WS_30", "WD_30", "WS_50", "WD_50",
                 "WS_cen", "WD_cen", "Air_T", "Air_P", "Air_H"]
SOLAR_COLS_BASE = ["time", "TSI", "DNI", "GHI", "Air_T", "Air_P", "Air_H"]

# 完整的列名列表，用于参考
WIND_COLS = WIND_COLS_BASE + ["Power(MW)"]
SOLAR_COLS = SOLAR_COLS_BASE + ["Power(MW)"]


@dataclass
class DatasetInfo:
    kind: str  # "wind" | "solar"
    site_name: str
    path: Path
    nominal_capacity_mw: Optional[float] = None


def list_sites(data_root: Path, kind: str = "wind") -> List[DatasetInfo]:
    """
    List xlsx files under data_root/(wind_farms|solar_stations).
    kind:
        - "wind": data_root/wind_farms/*.xlsx
        - "solar": data_root/solar_stations/*.xlsx
    """
    if kind not in {"wind", "solar"}:
        raise ValueError("kind must be 'wind' or 'solar'")
    folder = "wind_farms" if kind == "wind" else "solar_stations"
    paths = sorted((data_root / folder).glob("*.xlsx"))
    sites: List[DatasetInfo] = []
    for p in paths:
        # Try to parse nominal capacity from file name: "...(Nominal capacity-99MW).xlsx"
        cap = None
        name = p.stem
        import re
        m = re.search(r"Nominal capacity-(\d+(?:\.\d+)?)MW", name)
        if m:
            cap = float(m.group(1))
        sites.append(DatasetInfo(kind=kind, site_name=name, path=p, nominal_capacity_mw=cap))
    return sites


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c == "time" or c not in df.columns:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_site_xlsx(path: Path, kind: str) -> pd.DataFrame:
    """
    Load one site from xlsx, drop the first row (as in the original notebooks),
    rename columns to a consistent schema, set datetime index.

    Returns a DataFrame indexed by datetime with 15-min frequency if possible.
    """
    df = pd.read_excel(path).drop(index=0).copy()
    n_cols = len(df.columns)
    
    if kind == "wind":
        # 确保Power(MW)总是映射到最后一列
        if n_cols > 0:
            if n_cols == 1:
                # 只有时间列
                df.columns = ["time"]
            elif n_cols <= len(WIND_COLS_BASE):
                # 没有Power(MW)列
                df.columns = WIND_COLS_BASE[:n_cols]
            else:
                # 包含Power(MW)列，放在最后
                df.columns = WIND_COLS_BASE[:n_cols-1] + ["Power(MW)"]
        df = _coerce_numeric(df, df.columns.tolist())
    elif kind == "solar":
        # 确保Power(MW)总是映射到最后一列
        if n_cols > 0:
            if n_cols == 1:
                # 只有时间列
                df.columns = ["time"]
            elif n_cols <= len(SOLAR_COLS_BASE):
                # 没有Power(MW)列
                df.columns = SOLAR_COLS_BASE[:n_cols]
            else:
                # 包含Power(MW)列，放在最后
                df.columns = SOLAR_COLS_BASE[:n_cols-1] + ["Power(MW)"]
        df = _coerce_numeric(df, df.columns.tolist())
    else:
        raise ValueError("kind must be 'wind' or 'solar'")

    # time parsing
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time")
    df = df.set_index("time")

    # Optionally enforce 15-min frequency; keep missing timestamps but allow later interpolation.
    # Some stations might have duplicates; remove duplicates by keeping the last record.
    df = df[~df.index.duplicated(keep="last")]

    return df


def clean_and_impute(df: pd.DataFrame, max_gap_steps: int = 10) -> pd.DataFrame:
    """
    Light-touch cleaning:
    - Replace clearly invalid sentinel values (<= -90 for temperature etc.) with NaN.
    - Interpolate short gaps up to max_gap_steps (15min steps).
    - Forward/backward fill remaining NaNs cautiously (last resort).

    NOTE: Your earlier notebooks likely did more sophisticated outlier removal;
          this function is a conservative, generic fallback.
    """
    d = df.copy()

    # Sentinel-like values and impossible zeros (domain-specific heuristics)
    for c in d.columns:
        if c.startswith("WD_"):
            # Wind direction should be [0, 360)
            d.loc[(d[c] < 0) | (d[c] >= 360), c] = np.nan
        if c.startswith("WS_"):
            d.loc[(d[c] < 0) | (d[c] > 60), c] = np.nan  # 60 m/s is extremely high
        if c in {"Air_T"}:
            d.loc[(d[c] < -80) | (d[c] > 60), c] = np.nan
        if c in {"Air_P"}:
            d.loc[(d[c] < 500) | (d[c] > 1200), c] = np.nan  # 修正为合理的气压范围（hPa）
        if c in {"Air_H"}:
            d.loc[(d[c] < 0) | (d[c] > 100), c] = np.nan
        if c in {"TSI", "DNI", "GHI"}:
            d.loc[d[c] < 0, c] = np.nan
        if c == "Power(MW)":
            d.loc[d[c] < 0, c] = np.nan

    # Interpolate short gaps
    # limit = number of consecutive NaNs to fill (<= max_gap_steps)
    d = d.interpolate(method="time", limit=max_gap_steps, limit_direction="both")

    # Fallback fill
    d = d.ffill().bfill()

    return d


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calendar features + cyclical encoding for hour and day-of-year.
    """
    d = df.copy()
    idx = d.index
    d["hour"] = idx.hour
    d["minute"] = idx.minute
    d["dayofweek"] = idx.dayofweek
    d["month"] = idx.month
    d["dayofyear"] = idx.dayofyear

    # cyclic encoding
    d["hour_sin"] = np.sin(2 * np.pi * d["hour"] / 24.0)
    d["hour_cos"] = np.cos(2 * np.pi * d["hour"] / 24.0)
    d["doy_sin"] = np.sin(2 * np.pi * d["dayofyear"] / 365.25)
    d["doy_cos"] = np.cos(2 * np.pi * d["dayofyear"] / 365.25)
    return d


def add_wind_vector_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert wind speed + direction to u/v components to avoid circular discontinuity.
    Use the 'center' (hub) level when available, otherwise use the last available WS/WD pair.
    """
    d = df.copy()
    # Find a preferred pair
    candidates = [("WS_cen", "WD_cen"), ("WS_50", "WD_50"), ("WS_30", "WD_30"), ("WS_10", "WD_10")]
    ws_col, wd_col = None, None
    for ws, wd in candidates:
        if ws in d.columns and wd in d.columns:
            ws_col, wd_col = ws, wd
            break
    if ws_col is None:
        return d

    # Convert to radians; meteorological wind direction is typically degrees from north.
    rad = np.deg2rad(d[wd_col].astype(float))
    ws_val = d[ws_col].astype(float)

    # u/v: here we use a common convention: u = ws * sin(dir), v = ws * cos(dir)
    # (exact convention is less important as long as consistent)
    d[f"{ws_col}_u"] = ws_val * np.sin(rad)
    d[f"{ws_col}_v"] = ws_val * np.cos(rad)

    return d


def make_supervised_frame(
    df: pd.DataFrame,
    target_col: str = "Power(MW)",
    horizon: int = 96,
    lags: Tuple[int, ...] = (1, 2, 3, 4, 8, 12, 24, 48, 96),
    rolling_windows: Tuple[int, ...] = (4, 12, 24, 48, 96),
    exog_lag: Tuple[int, ...] = (1, 4, 12),
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Create a supervised learning table for multi-horizon forecasting.

    X contains:
      - time features (already present in df)
      - meteorological features at time t
      - lagged target and lagged meteorology
      - rolling stats of target

    y contains:
      - y_t+1 ... y_t+horizon (multi-output regression)
    """
    d = df.copy()

    # target lags
    for k in lags:
        d[f"{target_col}_lag_{k}"] = d[target_col].shift(k)

    # rolling stats of target
    for w in rolling_windows:
        d[f"{target_col}_roll_mean_{w}"] = d[target_col].shift(1).rolling(w).mean()
        d[f"{target_col}_roll_std_{w}"] = d[target_col].shift(1).rolling(w).std()

    # exogenous lags for non-target cols
    exog_cols = [c for c in d.columns if c != target_col]
    for c in exog_cols:
        # skip pure categorical/time cols; but safe to lag anyway
        for k in exog_lag:
            d[f"{c}_lag_{k}"] = d[c].shift(k)

    # multi-step targets
    y_cols = []
    for h in range(1, horizon + 1):
        col = f"y_t+{h}"
        d[col] = d[target_col].shift(-h)
        y_cols.append(col)

    # Drop rows with NaNs caused by shifting
    d = d.dropna()

    # Feature columns
    feature_cols = [c for c in d.columns if c not in y_cols]
    return d, feature_cols, y_cols


def time_train_val_test_split(
    supervised_df: pd.DataFrame,
    val_days: int = 30,
    test_days: int = 30,
    freq_minutes: int = 15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by time order (no leakage).
    """
    steps_per_day = int((24 * 60) / freq_minutes)
    val_steps = val_days * steps_per_day
    test_steps = test_days * steps_per_day

    if len(supervised_df) <= (val_steps + test_steps + 100):
        raise ValueError("Not enough data for the requested split; reduce val_days/test_days.")

    train = supervised_df.iloc[: -(val_steps + test_steps)]
    val = supervised_df.iloc[-(val_steps + test_steps) : -test_steps]
    test = supervised_df.iloc[-test_steps:]
    return train, val, test


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, capacity_mw: Optional[float] = None) -> Dict[str, float]:
    """
    y_true/y_pred: shape (n_samples, horizon)
    """
    eps = 1e-9
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps)))) * 100.0
    out = {"MAE": mae, "RMSE": rmse, "MAPE_%": mape}
    if capacity_mw is not None and capacity_mw > 0:
        out["nMAE_%"] = mae / capacity_mw * 100.0
        out["nRMSE_%"] = rmse / capacity_mw * 100.0
    return out
