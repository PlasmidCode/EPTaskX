
# -*- coding: utf-8 -*-

"""
Train a strong multi-horizon forecasting baseline (LightGBM) for a single site.

Why this approach:
- In on-site renewable forecasting, boosted trees with rich lag/rolling features
  are a very strong baseline, often competitive with deep models when you don't
  have NWP forecasts.
- The Scientific Data descriptor for this dataset explicitly points out that
  wind speed / solar irradiance are key factors and suggests seasonal / intensity
  based classification can improve forecasting accuracy. We implement a simple
  "bin feature" as a lightweight alternative to training many separate models.

Usage examples:

1) Wind farm site 2, 24h horizon (96 steps):
    python forecast_lgbm_multihorizon.py --kind wind --site "Wind farm site 2" --horizon 96 \
        --data_root data_processed --out_dir outputs/site2_wind

2) Solar station site 1:
    python forecast_lgbm_multihorizon.py --kind solar --site "Solar station site 1" --horizon 96 \
        --data_root data_processed --out_dir outputs/site1_solar

You can also list sites:
    python forecast_lgbm_multihorizon.py --list --kind wind --data_root data_processed
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error
import joblib

try:
    import lightgbm as lgb
except Exception as e:
    raise SystemExit(
        "LightGBM is required. Install with: pip install lightgbm\n"
        f"Original import error: {e}"
    )

from forecast_utils import (
    add_time_features,
    add_wind_vector_features,
    clean_and_impute,
    compute_metrics,
    list_sites,
    load_site_xlsx,
    make_supervised_frame,
    time_train_val_test_split,
)


def infer_site_path(data_root: Path, kind: str, site_query: str) -> tuple[Path, Optional[float], str]:
    """
    Find a file whose stem contains site_query (case-insensitive).
    Returns (path, nominal_capacity, site_name).
    """
    sites = list_sites(data_root, kind=kind)
    q = site_query.lower().strip()
    for s in sites:
        if q in s.site_name.lower():
            return s.path, s.nominal_capacity_mw, s.site_name
    # fallback: accept exact filename
    p = Path(site_query)
    if p.exists():
        return p, None, p.stem
    raise FileNotFoundError(f"Cannot find site matching '{site_query}' under {data_root}.")


def add_intensity_bins(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """
    A lightweight "classification" trick mentioned in the dataset descriptor:
    add a categorical bin derived from main driver variable:
      - wind: hub wind speed
      - solar: GHI (or TSI)
    Instead of training separate models, we add the bin as a feature.
    """
    d = df.copy()
    if kind == "wind":
        ws = None
        for c in ["WS_cen", "WS_50", "WS_30", "WS_10"]:
            if c in d.columns:
                ws = d[c]
                break
        if ws is not None:
            # 0-3, 3-6, 6-9, 9-12, >12 m/s
            d["ws_bin"] = pd.cut(ws.astype(float), bins=[-0.1, 3, 6, 9, 12, 60], labels=False).astype(int)
    else:
        ghi = d["GHI"] if "GHI" in d.columns else (d["TSI"] if "TSI" in d.columns else None)
        if ghi is not None:
            # low / mid / high irradiance
            d["irr_bin"] = pd.cut(ghi.astype(float), bins=[-1, 100, 400, 800, 2000], labels=False).astype(int)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data_processed", help="Root folder that contains wind_farms/ and solar_stations/")
    ap.add_argument("--kind", type=str, choices=["wind", "solar"], required=True)
    ap.add_argument("--site", type=str, default="", help="Substring of the xlsx filename (stem) to select a site")
    ap.add_argument("--list", action="store_true", help="List available sites and exit")
    ap.add_argument("--horizon", type=int, default=96, help="Forecast horizon in 15-min steps (96 = 24h)")
    ap.add_argument("--val_days", type=int, default=30)
    ap.add_argument("--test_days", type=int, default=30)
    ap.add_argument("--out_dir", type=str, default="outputs")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.list:
        sites = list_sites(data_root, kind=args.kind)
        for s in sites:
            print(f"- {s.site_name} | nominal_capacity={s.nominal_capacity_mw}MW | file={s.path}")
        return

    if not args.site:
        raise SystemExit("--site is required unless --list is used.")

    path, cap_mw, site_name = infer_site_path(data_root, args.kind, args.site)
    print(f"[INFO] Using site: {site_name}")
    print(f"[INFO] File: {path}")
    print(f"[INFO] Nominal capacity: {cap_mw} MW (parsed from filename, may be None)")

    # Load + clean
    df = load_site_xlsx(path, kind=args.kind)
    df = clean_and_impute(df)
    df = add_time_features(df)
    if args.kind == "wind":
        df = add_wind_vector_features(df)
    df = add_intensity_bins(df, kind=args.kind)

    # Supervised table
    supervised, feature_cols, y_cols = make_supervised_frame(
        df,
        target_col="Power(MW)",
        horizon=args.horizon,
        lags=(1, 2, 3, 4, 8, 12, 24, 48, 96),
        rolling_windows=(4, 12, 24, 48, 96),
        exog_lag=(1, 4, 12),
    )

    train_df, val_df, test_df = time_train_val_test_split(
        supervised,
        val_days=args.val_days,
        test_days=args.test_days,
        freq_minutes=15,
    )

    X_train, y_train = train_df[feature_cols], train_df[y_cols]
    X_val, y_val = val_df[feature_cols], val_df[y_cols]
    X_test, y_test = test_df[feature_cols], test_df[y_cols]

    # LightGBM base estimator: tuned defaults that are robust for this kind of data
    base = lgb.LGBMRegressor(
        n_estimators=4000,
        learning_rate=0.02,
        num_leaves=127,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=args.seed,
        n_jobs=-1,
    )

    model = MultiOutputRegressor(base, n_jobs=-1)

    print("[INFO] Training...")
    # MultiOutputRegressor doesn't expose early_stopping. For reproducibility, we train fixed estimators.
    # If you need early stopping, you can train one model per horizon with callbacks.
    model.fit(X_train, y_train)

    # Validate
    print("[INFO] Evaluating...")
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    metrics_val = compute_metrics(y_val.to_numpy(), y_pred_val, capacity_mw=cap_mw)
    metrics_test = compute_metrics(y_test.to_numpy(), y_pred_test, capacity_mw=cap_mw)

    # Save
    joblib.dump(
        {"model": model, "feature_cols": feature_cols, "y_cols": y_cols, "site": site_name, "kind": args.kind, "capacity_mw": cap_mw},
        out_dir / "lgbm_multihorizon.joblib",
    )

    # Report
    report = {
        "site": site_name,
        "kind": args.kind,
        "capacity_mw": cap_mw,
        "horizon": args.horizon,
        "val_days": args.val_days,
        "test_days": args.test_days,
        "metrics_val": metrics_val,
        "metrics_test": metrics_test,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
    }

    (out_dir / "report.json").write_text(pd.Series(report).to_json(force_ascii=False, indent=2), encoding="utf-8")

    print("\n===== Validation metrics =====")
    for k, v in metrics_val.items():
        print(f"{k}: {v:.4f}")

    print("\n===== Test metrics =====")
    for k, v in metrics_test.items():
        print(f"{k}: {v:.4f}")

    print(f"\n[OK] Saved model + report to: {out_dir}")


if __name__ == "__main__":
    main()
