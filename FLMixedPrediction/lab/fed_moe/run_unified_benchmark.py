import argparse
import copy
import itertools
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset


PRIMARY_METHODS = ["Local", "FedAvg", "FedPer", "FL+MoE"]
AUXILIARY_METHODS = ["Persistence", "Centralized"]
METHODS = PRIMARY_METHODS + AUXILIARY_METHODS
HORIZONS = [1, 6, 24]
CORE_METRICS = ["rmse", "mae", "r2", "smape", "wape", "nrmse_range", "corr"]
POINT_METRICS = [
    "mse",
    "rmse",
    "mae",
    "r2",
    "mbe",
    "smape",
    "wape",
    "nrmse_range",
    "nmae_range",
    "corr",
]
INTERVAL_METRICS = [
    "picp_80",
    "pinaw_80",
    "interval_width_mean",
    "coverage_error_80",
    "q10_loss",
    "q90_loss",
]
RESULT_METRICS = POINT_METRICS + INTERVAL_METRICS
AUX_RESULT_FIELDS = ["postprocess_choice", "postprocess_val_rmse", "blend_model_weight", "affine_a", "affine_b"]
LOWER_IS_BETTER = {
    "mse",
    "rmse",
    "mae",
    "mbe_abs",
    "smape",
    "wape",
    "nrmse_range",
    "nmae_range",
    "pinaw_80",
    "coverage_error_80",
    "q10_loss",
    "q90_loss",
}
METHOD_COLORS = {
    "Persistence": "#8a8f98",
    "Local": "#2563eb",
    "Centralized": "#7c3aed",
    "FedAvg": "#f97316",
    "FedPer": "#16a34a",
    "FL+MoE": "#dc2626",
}
METHOD_MARKERS = {
    "Persistence": "X",
    "Local": "o",
    "Centralized": "D",
    "FedAvg": "s",
    "FedPer": "^",
    "FL+MoE": "P",
}
METRIC_LABELS = {
    "rmse": "RMSE",
    "mae": "MAE",
    "r2": "R2",
    "smape": "SMAPE (%)",
    "wape": "WAPE (%)",
    "nrmse_range": "NRMSE",
    "picp_80": "PICP (80% PI)",
    "pinaw_80": "PINAW (80% PI)",
    "corr": "Correlation",
}
INTERVAL_ALPHA = 0.10
DEFAULT_HORIZONS = [1, 6, 24]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True


def parse_csv_ints(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_csv_strings(value: str, choices: List[str]) -> List[str]:
    items = [x.strip() for x in value.split(",") if x.strip()]
    unknown = [x for x in items if x not in choices]
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown choices: {unknown}; valid choices: {choices}")
    return items


def resolve_device(args: argparse.Namespace) -> torch.device:
    if args.device == "cpu":
        return torch.device("cpu")
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def use_amp(args: argparse.Namespace, device: torch.device) -> bool:
    return bool(args.amp and device.type == "cuda")


def move_batch(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    return x.to(device, non_blocking=(device.type == "cuda"))


def loader_kwargs(args: argparse.Namespace) -> Dict[str, object]:
    workers = max(int(getattr(args, "num_workers", 0)), 0)
    kwargs: Dict[str, object] = {
        "num_workers": workers,
        "pin_memory": bool(getattr(args, "pin_memory", False)),
    }
    if workers > 0:
        kwargs["persistent_workers"] = True
    return kwargs


def detect_time_col(df: pd.DataFrame) -> str | None:
    for col in ["Time(year-month-day h:m:s)", "Time", "time", "datetime", "DateTime"]:
        if col in df.columns:
            return col
    return None


def detect_target_col(df: pd.DataFrame) -> str:
    for col in ["Power (MW)", "power", "ActivePower", "Generated Power"]:
        if col in df.columns:
            return col
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric target column found.")
    return numeric_cols[-1]


def site_type(site_name: str) -> str:
    return "wind" if "wind" in site_name else "solar"


def site_index(site_name: str) -> int:
    try:
        return int(site_name.split("_")[-1])
    except ValueError:
        return 0


def load_sites(data_root: Path) -> Dict[str, pd.DataFrame]:
    datasets: Dict[str, pd.DataFrame] = {}
    for sub in ["wind_farms", "solar_stations"]:
        subdir = data_root / sub
        if not subdir.exists():
            continue
        for f in sorted(subdir.glob("*.xlsx")):
            site_name = f.stem.split(" (")[0].replace(" ", "_").lower()
            df = pd.read_excel(f)
            time_col = detect_time_col(df)
            target_col = detect_target_col(df)
            df = df.rename(columns={target_col: "power"}).copy()
            if time_col is not None:
                if df[time_col].dtype == "object":
                    df[time_col] = df[time_col].str.replace(" 24:", " 00:", regex=False)
                df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
                df = df.dropna(subset=[time_col]).set_index(time_col).sort_index()
                df = df.resample("1h").mean()
            else:
                df = df.reset_index(drop=True)
            df["power"] = pd.to_numeric(df["power"], errors="coerce")
            df = df.dropna(subset=["power"]).copy()
            datasets[site_name] = df
    if not datasets:
        raise FileNotFoundError(f"No site files found under {data_root}")
    return datasets


def weather_feature_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for col in df.columns:
        if col == "power":
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def make_common_features(df: pd.DataFrame, site_name: str, feature_preset: str = "v5") -> pd.DataFrame:
    out = df.copy()
    numeric_cols = [col for col in out.columns if pd.api.types.is_numeric_dtype(out[col])]
    for col in numeric_cols:
        low = out[col].quantile(0.01)
        high = out[col].quantile(0.99)
        out[col] = out[col].clip(lower=low, upper=high)

    if feature_preset == "enhanced" and site_type(site_name) == "wind":
        wind_features = {}
        for col in list(numeric_cols):
            lower = col.lower()
            if "wind speed" in lower and "m/s" in lower:
                wind_features[f"{col}_sq"] = out[col] ** 2
                wind_features[f"{col}_cube"] = out[col] ** 3
            if "wind direction" in lower:
                radians = np.deg2rad(out[col])
                wind_features[f"{col}_sin"] = np.sin(radians)
                wind_features[f"{col}_cos"] = np.cos(radians)
        if wind_features:
            out = pd.concat([out, pd.DataFrame(wind_features, index=out.index)], axis=1)

    if isinstance(out.index, pd.DatetimeIndex):
        hours = out.index.hour.values
        dayofyear = out.index.dayofyear.values
        months = out.index.month.values
        out["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
        out["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
        out["month_sin"] = np.sin(2 * np.pi * months / 12.0)
        out["month_cos"] = np.cos(2 * np.pi * months / 12.0)
        out["doy_sin"] = np.sin(2 * np.pi * dayofyear / 365.0)
        out["doy_cos"] = np.cos(2 * np.pi * dayofyear / 365.0)
    else:
        for c in ["hour_sin", "hour_cos", "month_sin", "month_cos", "doy_sin", "doy_cos"]:
            out[c] = 0.0
    power_features = {
        "site_id_norm": pd.Series(site_index(site_name) / 10.0, index=out.index),
        "power_lag_1": out["power"].shift(1),
        "power_lag_2": out["power"].shift(2),
        "power_lag_6": out["power"].shift(6),
        "power_lag_24": out["power"].shift(24),
        "power_roll_mean_6": out["power"].rolling(6).mean(),
        "power_roll_std_6": out["power"].rolling(6).std(),
        "power_roll_mean_24": out["power"].rolling(24).mean(),
        "power_roll_std_24": out["power"].rolling(24).std(),
    }
    if feature_preset == "enhanced":
        power_features.update(
            {
                "power_diff_1": out["power"].diff(1),
                "power_diff_6": out["power"].diff(6),
                "power_roll_min_6": out["power"].rolling(6).min(),
                "power_roll_max_6": out["power"].rolling(6).max(),
                "power_roll_min_24": out["power"].rolling(24).min(),
                "power_roll_max_24": out["power"].rolling(24).max(),
            }
        )
    out = pd.concat([out, pd.DataFrame(power_features, index=out.index)], axis=1)
    weather_lags = {}
    for col in weather_feature_columns(out):
        weather_lags[f"{col}_lag_1"] = out[col].shift(1)
        weather_lags[f"{col}_lag_6"] = out[col].shift(6)
    if weather_lags:
        out = pd.concat([out, pd.DataFrame(weather_lags, index=out.index)], axis=1)
    out = out.dropna().copy()
    return out


def build_sequences(
    df: pd.DataFrame,
    seq_len: int,
    horizon: int,
    feature_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    vals_x = df[feature_cols].values.astype(np.float32)
    vals_y = df["power"].values.astype(np.float32).reshape(-1, 1)
    vals_power = df["power"].values.astype(np.float32).reshape(-1, 1)
    X, y, persistence = [], [], []
    for i in range(seq_len, len(df) - horizon + 1):
        X.append(vals_x[i - seq_len : i])
        y.append(vals_y[i + horizon - 1])
        persistence.append(vals_power[i - 1])
    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.float32),
        np.array(persistence, dtype=np.float32),
    )


def split_and_scale(
    df: pd.DataFrame,
    seq_len: int,
    horizon: int,
    feature_cols: List[str],
) -> Dict[str, np.ndarray | MinMaxScaler]:
    X, y, persistence = build_sequences(df, seq_len, horizon, feature_cols)
    n = len(X)
    s1, s2 = int(n * 0.7), int(n * 0.85)
    X_train, X_val, X_test = X[:s1], X[s1:s2], X[s2:]
    y_train, y_val, y_test = y[:s1], y[s1:s2], y[s2:]
    p_train, p_val, p_test = persistence[:s1], persistence[s1:s2], persistence[s2:]

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    x_scaler.fit(X_train.reshape(-1, X_train.shape[-1]))
    y_scaler.fit(y_train)

    def tx(arr: np.ndarray) -> np.ndarray:
        arr2 = arr.reshape(-1, arr.shape[-1])
        arr2 = x_scaler.transform(arr2)
        return arr2.reshape(arr.shape)

    return {
        "X_train": tx(X_train),
        "X_val": tx(X_val),
        "X_test": tx(X_test),
        "y_train": y_scaler.transform(y_train),
        "y_val": y_scaler.transform(y_val),
        "y_test": y_scaler.transform(y_test),
        "p_train": p_train,
        "p_val": p_val,
        "p_test": p_test,
        "y_scaler": y_scaler,
    }


def align_feature_space(
    datasets: Dict[str, pd.DataFrame],
) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    feature_union = sorted(
        {
            col
            for df in datasets.values()
            for col in df.columns
            if col != "power"
        }
    )
    aligned = {}
    for site, df in datasets.items():
        site_df = df.copy()
        for col in feature_union:
            if col not in site_df.columns:
                site_df[col] = 0.0
        aligned[site] = site_df[feature_union + ["power"]].copy()
    return aligned, feature_union


class LSTMForecast(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])


class FedPerForecast(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.shared_conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.shared_rnn = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.shared_fc = nn.Linear(hidden_dim, hidden_dim)
        self.personal_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.shared_conv(x)
        x = x.permute(0, 2, 1)
        out, _ = self.shared_rnn(x)
        out = self.shared_fc(out[:, -1, :])
        return self.personal_head(out)

    def get_shared_state(self) -> Dict[str, torch.Tensor]:
        return {
            k: v.detach().clone()
            for k, v in self.state_dict().items()
            if k.startswith("shared_")
        }

    def load_shared_state(self, state: Dict[str, torch.Tensor]) -> None:
        own = self.state_dict()
        for k, v in state.items():
            if k in own:
                own[k].copy_(v)


class LegacyMoEForecast(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_experts: int = 4,
    ):
        super().__init__()
        self.encoder = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )
        self.encoder_norm = nn.LayerNorm(hidden_dim)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.GELU(),
                    nn.Linear(hidden_dim // 2, 1),
                )
                for _ in range(num_experts)
            ]
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_experts),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc_out, _ = self.encoder(x)
        shared = self.encoder_norm(enc_out[:, -1, :])
        weights = self.gate(shared)
        outputs = []
        for expert in self.experts:
            outputs.append(expert(shared))
        stacked = torch.stack(outputs, dim=1)
        return torch.sum(stacked * weights.unsqueeze(-1), dim=1)

    def get_shared_state(self) -> Dict[str, torch.Tensor]:
        state = {}
        for key, value in self.state_dict().items():
            if key.startswith("gate."):
                continue
            state[key] = value.detach().clone()
        return state

    def load_shared_state(self, state: Dict[str, torch.Tensor]) -> None:
        own = self.state_dict()
        for key, value in state.items():
            if key in own:
                own[key].copy_(value)


class MoEForecast(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_experts: int = 4,
        top_k: int = 2,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = max(1, min(top_k, num_experts))
        self.encoder = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )
        self.encoder_norm = nn.LayerNorm(hidden_dim)
        self.context_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.base_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.GELU(),
                    nn.Linear(hidden_dim // 2, 1),
                )
                for _ in range(num_experts)
            ]
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_experts),
            nn.Softmax(dim=-1),
        )
        # Site-level affine calibration stays local and is excluded from federation.
        self.calibration = nn.Linear(1, 1)
        with torch.no_grad():
            self.calibration.weight.fill_(1.0)
            self.calibration.bias.zero_()

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        enc_out, _ = self.encoder(x)
        last_hidden = enc_out[:, -1, :]
        mean_hidden = enc_out.mean(dim=1)
        shared = self.context_proj(torch.cat([last_hidden, mean_hidden], dim=-1))
        shared = self.encoder_norm(shared)
        weights = self.gate(shared)
        if self.top_k < self.num_experts:
            top_vals, top_idx = torch.topk(weights, k=self.top_k, dim=-1)
            sparse = torch.zeros_like(weights).scatter(1, top_idx, top_vals)
            weights = sparse / sparse.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        outputs = []
        for expert in self.experts:
            outputs.append(expert(shared))
        stacked = torch.stack(outputs, dim=1)
        base_pred = self.base_head(shared)
        residual_pred = torch.sum(stacked * weights.unsqueeze(-1), dim=1)
        pred = self.calibration(base_pred + residual_pred)
        if return_aux:
            return pred, {
                "gate_weights": weights,
                "base_pred": base_pred,
                "residual_pred": residual_pred,
                "expert_outputs": stacked,
            }
        return pred

    def get_shared_state(self) -> Dict[str, torch.Tensor]:
        state = {}
        for key, value in self.state_dict().items():
            if key.startswith("gate.") or key.startswith("calibration."):
                continue
            state[key] = value.detach().clone()
        return state

    def load_shared_state(self, state: Dict[str, torch.Tensor]) -> None:
        own = self.state_dict()
        for key, value in state.items():
            if key in own:
                own[key].copy_(value)


def make_loaders(
    client: Dict[str, np.ndarray | MinMaxScaler],
    batch_size: int,
    args: argparse.Namespace,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = TensorDataset(
        torch.tensor(client["X_train"], dtype=torch.float32),
        torch.tensor(client["y_train"], dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(client["X_val"], dtype=torch.float32),
        torch.tensor(client["y_val"], dtype=torch.float32),
    )
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs(args)),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs(args)),
    )


def make_adaptation_loader(
    client: Dict[str, np.ndarray | MinMaxScaler],
    batch_size: int,
    args: argparse.Namespace,
) -> DataLoader:
    X = np.concatenate([client["X_train"], client["X_val"]], axis=0)
    y = np.concatenate([client["y_train"], client["y_val"]], axis=0)
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=True, **loader_kwargs(args))


def train_supervised(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    lr: float,
    epochs: int,
    args: argparse.Namespace,
) -> Tuple[Dict[str, torch.Tensor], float]:
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    amp_on = use_amp(args, device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_on)
    best_val = np.inf
    best_state = copy.deepcopy(model.state_dict())
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = move_batch(xb, device), move_batch(yb, device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_on):
                pred = model(xb)
                loss = loss_fn(pred, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = move_batch(xb, device), move_batch(yb, device)
                with torch.amp.autocast(device_type=device.type, enabled=amp_on):
                    pred = model(xb)
                    val_loss = loss_fn(pred, yb)
                total += val_loss.item() * xb.size(0)
                n += xb.size(0)
        val = total / max(n, 1)
        if val < best_val:
            best_val = val
            best_state = copy.deepcopy(model.state_dict())
    return best_state, float(best_val)


def aggregate_weighted(states: List[Dict[str, torch.Tensor]], weights: List[int]) -> Dict[str, torch.Tensor]:
    total = float(sum(weights))
    out = {}
    for key in states[0].keys():
        out[key] = sum(state[key] * (w / total) for state, w in zip(states, weights))
    return out


def create_moe_model(
    input_dim: int,
    args: argparse.Namespace,
) -> nn.Module:
    if args.moe_variant == "legacy":
        return LegacyMoEForecast(input_dim, args.hidden_dim, args.num_experts)
    return MoEForecast(
        input_dim,
        args.hidden_dim,
        args.num_experts,
        args.moe_top_k,
    )


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mbe = float(np.mean(y_pred - y_true))
    smape_denom = np.maximum(np.abs(y_true) + np.abs(y_pred), 1e-8)
    smape = float(np.mean(2.0 * np.abs(y_pred - y_true) / smape_denom) * 100.0)
    wape = float(np.sum(np.abs(y_pred - y_true)) / np.maximum(np.sum(np.abs(y_true)), 1e-8) * 100.0)
    y_range = float(np.max(y_true) - np.min(y_true))
    nrmse_range = float(rmse / max(y_range, 1e-8))
    nmae_range = float(mae / max(y_range, 1e-8))
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else np.nan
    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "mbe": mbe,
        "smape": smape,
        "wape": wape,
        "nrmse_range": nrmse_range,
        "nmae_range": nmae_range,
        "corr": corr,
    }


def pinball_loss(y_true: np.ndarray, y_quantile: np.ndarray, quantile: float) -> float:
    y_true = y_true.reshape(-1)
    y_quantile = y_quantile.reshape(-1)
    diff = y_true - y_quantile
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1.0) * diff)))


def residual_interval(
    y_pred: np.ndarray,
    residuals: np.ndarray,
    lower_q: float = INTERVAL_ALPHA,
    upper_q: float = 1.0 - INTERVAL_ALPHA,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    y_pred = y_pred.reshape(-1)
    residuals = residuals.reshape(-1)
    finite = residuals[np.isfinite(residuals)]
    if finite.size < 2:
        width = np.maximum(np.abs(y_pred) * 0.15, 1e-6)
        lower_residual = -float(np.mean(width))
        upper_residual = float(np.mean(width))
    else:
        lower_residual = float(np.quantile(finite, lower_q))
        upper_residual = float(np.quantile(finite, upper_q))
    y_lower = np.maximum(0.0, y_pred + lower_residual)
    y_upper = np.maximum(y_lower, np.maximum(0.0, y_pred + upper_residual))
    return y_lower, y_upper, lower_residual, upper_residual


def interval_metrics(y_true: np.ndarray, y_lower: np.ndarray, y_upper: np.ndarray) -> Dict[str, float]:
    y_true = y_true.reshape(-1)
    y_lower = y_lower.reshape(-1)
    y_upper = y_upper.reshape(-1)
    inside = (y_true >= y_lower) & (y_true <= y_upper)
    width = np.maximum(y_upper - y_lower, 0.0)
    y_range = float(np.max(y_true) - np.min(y_true)) if y_true.size else 0.0
    picp = float(np.mean(inside)) if y_true.size else np.nan
    pinaw = float(np.mean(width) / max(y_range, 1e-8)) if y_true.size else np.nan
    return {
        "picp_80": picp,
        "pinaw_80": pinaw,
        "interval_width_mean": float(np.mean(width)) if width.size else np.nan,
        "coverage_error_80": float(abs(picp - 0.80)) if np.isfinite(picp) else np.nan,
        "q10_loss": pinball_loss(y_true, y_lower, INTERVAL_ALPHA),
        "q90_loss": pinball_loss(y_true, y_upper, 1.0 - INTERVAL_ALPHA),
    }


def affine_calibrate(
    val_true: np.ndarray,
    val_pred: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    x = val_pred.reshape(-1)
    y = val_true.reshape(-1)
    if len(x) < 2 or float(np.std(x)) < 1e-8:
        return val_pred, y_pred, {"affine_a": 1.0, "affine_b": 0.0}
    design = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(design, y, rcond=None)[0]
    a = float(np.clip(a, 0.25, 1.75))
    y_min = float(min(np.min(y), np.min(y_pred)))
    y_max = float(max(np.max(y), np.max(y_pred)))
    val_out = np.clip(a * val_pred + b, y_min, y_max)
    test_out = np.clip(a * y_pred + b, y_min, y_max)
    return val_out, test_out, {"affine_a": a, "affine_b": float(b)}


def blend_with_persistence(
    val_true: np.ndarray,
    val_pred: np.ndarray,
    y_pred: np.ndarray,
    p_val: np.ndarray,
    p_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    best_alpha = 1.0
    best_rmse = np.inf
    best_val = val_pred
    for alpha in np.linspace(0.0, 1.0, 21):
        cand = alpha * val_pred + (1.0 - alpha) * p_val
        rmse = float(np.sqrt(mean_squared_error(val_true.reshape(-1), cand.reshape(-1))))
        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = float(alpha)
            best_val = cand
    test_out = best_alpha * y_pred + (1.0 - best_alpha) * p_test
    return best_val, test_out, {"blend_model_weight": best_alpha}


def apply_postprocess(
    client: Dict[str, np.ndarray | MinMaxScaler],
    y_pred: np.ndarray,
    val_pred: np.ndarray,
    mode: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float | str]]:
    val_true = client["y_scaler"].inverse_transform(client["y_val"]).reshape(-1, 1).astype(np.float64)
    p_val = client["p_val"].reshape(-1, 1).astype(np.float64)
    p_test = client["p_test"].reshape(-1, 1).astype(np.float64)
    y_pred = y_pred.reshape(-1, 1).astype(np.float64)
    val_pred = val_pred.reshape(-1, 1).astype(np.float64)

    candidates: List[Tuple[str, np.ndarray, np.ndarray, Dict[str, float]]] = [
        ("raw", val_pred, y_pred, {}),
    ]
    if mode in {"affine", "auto"}:
        val_affine, test_affine, meta = affine_calibrate(val_true, val_pred, y_pred)
        candidates.append(("affine", val_affine, test_affine, meta))
    if mode in {"blend", "auto"}:
        val_blend, test_blend, meta = blend_with_persistence(val_true, val_pred, y_pred, p_val, p_test)
        candidates.append(("blend", val_blend, test_blend, meta))
    if mode == "auto":
        val_affine, test_affine, meta_affine = affine_calibrate(val_true, val_pred, y_pred)
        val_blend, test_blend, meta_blend = blend_with_persistence(
            val_true,
            val_affine,
            test_affine,
            p_val,
            p_test,
        )
        candidates.append(("affine_blend", val_blend, test_blend, {**meta_affine, **meta_blend}))

    best = min(
        candidates,
        key=lambda item: float(np.sqrt(mean_squared_error(val_true.reshape(-1), item[1].reshape(-1)))),
    )
    name, best_val, best_test, meta = best
    meta_out: Dict[str, float | str] = {
        "postprocess_choice": name,
        "postprocess_val_rmse": float(np.sqrt(mean_squared_error(val_true.reshape(-1), best_val.reshape(-1)))),
    }
    meta_out.update(meta)
    return best_test, best_val, meta_out


def evaluate_raw_predictions(
    client: Dict[str, np.ndarray | MinMaxScaler],
    y_pred: np.ndarray,
    val_pred: np.ndarray,
    postprocess: str = "none",
) -> Dict[str, np.ndarray | float]:
    y_true = client["y_scaler"].inverse_transform(client["y_test"]).reshape(-1).astype(np.float64)
    val_true = client["y_scaler"].inverse_transform(client["y_val"]).reshape(-1).astype(np.float64)
    y_pred = y_pred.reshape(-1).astype(np.float64)
    val_pred = val_pred.reshape(-1).astype(np.float64)
    post_meta: Dict[str, float | str] = {
        "postprocess_choice": "none",
        "postprocess_val_rmse": float(np.sqrt(mean_squared_error(val_true, val_pred))),
    }
    if postprocess != "none":
        y_pred_pp, val_pred_pp, post_meta = apply_postprocess(client, y_pred, val_pred, postprocess)
        y_pred = y_pred_pp.reshape(-1)
        val_pred = val_pred_pp.reshape(-1)
    metrics = regression_metrics(y_true, y_pred)
    residuals = val_true - val_pred
    y_p10, y_p90, q10_residual, q90_residual = residual_interval(y_pred, residuals)
    metrics.update(interval_metrics(y_true, y_p10, y_p90))
    metrics.update(post_meta)
    metrics["q10_residual"] = q10_residual
    metrics["q90_residual"] = q90_residual
    metrics["y_true"] = y_true
    metrics["y_pred"] = y_pred
    metrics["y_p10"] = y_p10
    metrics["y_p90"] = y_p90
    return metrics


def evaluate_model(
    model: nn.Module,
    client: Dict[str, np.ndarray | MinMaxScaler],
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, np.ndarray | float]:
    model.eval()
    amp_on = use_amp(args, device)
    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, enabled=amp_on):
            val_pred_scaled = model(
                torch.tensor(client["X_val"], dtype=torch.float32, device=device)
            ).cpu().numpy()
            pred_scaled = model(
                torch.tensor(client["X_test"], dtype=torch.float32, device=device)
            ).cpu().numpy()
    val_pred = client["y_scaler"].inverse_transform(val_pred_scaled)
    y_pred = client["y_scaler"].inverse_transform(pred_scaled)
    return evaluate_raw_predictions(client, y_pred, val_pred, args.postprocess)


def result_row(site: str, res: Dict[str, np.ndarray | float]) -> Dict[str, object]:
    return {
        "site": site,
        "type": site_type(site),
        **{k: res[k] for k in RESULT_METRICS},
        **{k: res.get(k, np.nan) for k in AUX_RESULT_FIELDS},
    }


def moe_objective(
    pred: torch.Tensor,
    target: torch.Tensor,
    aux: Dict[str, torch.Tensor],
    model: MoEForecast,
    args: argparse.Namespace,
) -> torch.Tensor:
    mse = F.mse_loss(pred, target, reduction="none")
    huber = F.smooth_l1_loss(pred, target, reduction="none")
    peak_weights = 1.0 + args.moe_peak_weight * target.detach().clamp(0.0, 1.0)
    point_loss = ((0.5 * mse) + (0.5 * huber)) * peak_weights
    point_loss = point_loss.mean()

    under_pred = torch.relu(target - pred)
    under_penalty = (under_pred.pow(2) * peak_weights).mean()

    gate_weights = aux["gate_weights"]
    usage = gate_weights.mean(dim=0)
    uniform = torch.full_like(usage, 1.0 / model.num_experts)
    load_balance = torch.mean((usage - uniform) ** 2)

    return (
        point_loss
        + args.moe_under_weight * under_penalty
        + args.moe_balance_weight * load_balance
    )


def personalize_moe(
    model: nn.Module,
    adapt_loader: DataLoader,
    device: torch.device,
    lr: float,
    epochs: int,
    args: argparse.Namespace,
) -> nn.Module:
    model = copy.deepcopy(model).to(device)
    for name, param in model.named_parameters():
        if args.moe_variant == "legacy":
            param.requires_grad = name.startswith("gate.") or name.startswith("experts.")
        else:
            param.requires_grad = (
                name.startswith("gate.")
                or name.startswith("experts.")
                or name.startswith("calibration.")
                or name.startswith("base_head.")
            )
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4) if args.moe_variant != "legacy" else torch.optim.Adam(params, lr=lr)
    loss_fn = nn.MSELoss()
    amp_on = use_amp(args, device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_on)
    model.train()
    for _ in range(epochs):
        for xb, yb in adapt_loader:
            xb, yb = move_batch(xb, device), move_batch(yb, device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_on):
                if args.moe_variant == "legacy":
                    pred = model(xb)
                    loss = loss_fn(pred, yb)
                else:
                    pred, aux = model(xb, return_aux=True)
                    loss = moe_objective(pred, yb, aux, model, args)
            scaler.scale(loss).backward()
            if args.moe_variant != "legacy":
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            scaler.step(opt)
            scaler.update()
    return model


def run_local(
    clients: Dict[str, Dict[str, np.ndarray | MinMaxScaler]],
    input_dim: int,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, np.ndarray | float]], Dict[str, List[float]]]:
    rows = []
    examples = {}
    history = {}
    epochs = args.rounds * args.local_epochs
    for site, client in clients.items():
        train_loader, val_loader = make_loaders(client, args.batch_size, args)
        model = LSTMForecast(input_dim, args.hidden_dim, args.num_layers)
        state, best_val = train_supervised(model, train_loader, val_loader, device, args.lr, epochs, args)
        model.load_state_dict(state)
        model.to(device)
        res = evaluate_model(model, client, device, args)
        rows.append(result_row(site, res))
        examples[site] = res
        history[site] = [best_val]
    return pd.DataFrame(rows), examples, history


def run_persistence(
    clients: Dict[str, Dict[str, np.ndarray | MinMaxScaler]],
    input_dim: int,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, np.ndarray | float]], Dict[str, List[float]]]:
    rows = []
    examples = {}
    history = {}
    for site, client in clients.items():
        res = evaluate_raw_predictions(client, client["p_test"], client["p_val"], "none")
        rows.append(result_row(site, res))
        examples[site] = res
        history[site] = [0.0]
    return pd.DataFrame(rows), examples, history


def run_centralized(
    clients: Dict[str, Dict[str, np.ndarray | MinMaxScaler]],
    input_dim: int,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, np.ndarray | float]], Dict[str, List[float]]]:
    grouped = {"wind": [], "solar": []}
    for site in clients:
        grouped[site_type(site)].append(site)
    rows = []
    examples = {}
    history = {"wind": [], "solar": []}
    models = {}
    epochs = args.rounds * args.local_epochs
    for typ in ["wind", "solar"]:
        selected = grouped[typ]
        if not selected:
            continue
        X_train = np.concatenate([clients[site]["X_train"] for site in selected], axis=0)
        y_train = np.concatenate([clients[site]["y_train"] for site in selected], axis=0)
        X_val = np.concatenate([clients[site]["X_val"] for site in selected], axis=0)
        y_val = np.concatenate([clients[site]["y_val"] for site in selected], axis=0)
        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        val_ds = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs(args))
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs(args))
        model = LSTMForecast(input_dim, args.hidden_dim, args.num_layers)
        state, best_val = train_supervised(model, train_loader, val_loader, device, args.lr, epochs, args)
        model.load_state_dict(state)
        model.to(device)
        models[typ] = model
        history[typ] = [best_val]

    for site, client in clients.items():
        res = evaluate_model(models[site_type(site)], client, device, args)
        rows.append(result_row(site, res))
        examples[site] = res
    return pd.DataFrame(rows), examples, history


def run_fedavg(
    clients: Dict[str, Dict[str, np.ndarray | MinMaxScaler]],
    input_dim: int,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, np.ndarray | float]], Dict[str, List[float]]]:
    grouped = {"wind": [], "solar": []}
    for site in clients:
        grouped[site_type(site)].append(site)
    rows = []
    examples = {}
    history = {"wind": [], "solar": []}
    models = {}
    for typ in ["wind", "solar"]:
        if not grouped[typ]:
            continue
        global_model = LSTMForecast(input_dim, args.hidden_dim, args.num_layers).to(device)
        for _ in range(args.rounds):
            selected = grouped[typ]
            states, weights, vals = [], [], []
            for site in selected:
                train_loader, val_loader = make_loaders(clients[site], args.batch_size, args)
                local = LSTMForecast(input_dim, args.hidden_dim, args.num_layers)
                local.load_state_dict(copy.deepcopy(global_model.state_dict()))
                state, best_val = train_supervised(local, train_loader, val_loader, device, args.lr, args.local_epochs, args)
                states.append(state)
                weights.append(len(clients[site]["X_train"]))
                vals.append(best_val)
            global_model.load_state_dict(aggregate_weighted(states, weights))
            history[typ].append(float(np.mean(vals)))
        models[typ] = global_model
    for site, client in clients.items():
        res = evaluate_model(models[site_type(site)], client, device, args)
        rows.append(result_row(site, res))
        examples[site] = res
    return pd.DataFrame(rows), examples, history


def run_fedper(
    clients: Dict[str, Dict[str, np.ndarray | MinMaxScaler]],
    input_dim: int,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, np.ndarray | float]], Dict[str, List[float]]]:
    grouped = {"wind": [], "solar": []}
    for site in clients:
        grouped[site_type(site)].append(site)
    rows = []
    examples = {}
    history = {"wind": [], "solar": []}
    global_shared = {}
    personal_models: Dict[str, FedPerForecast] = {}

    for typ in ["wind", "solar"]:
        if not grouped[typ]:
            continue
        template = FedPerForecast(input_dim, args.hidden_dim, args.num_layers).to(device)
        global_shared[typ] = template.get_shared_state()
        for _ in range(args.rounds):
            selected = grouped[typ]
            shared_states, weights, vals = [], [], []
            for site in selected:
                if site not in personal_models:
                    personal_models[site] = FedPerForecast(input_dim, args.hidden_dim, args.num_layers)
                local = personal_models[site]
                local.load_shared_state(global_shared[typ])
                train_loader, val_loader = make_loaders(clients[site], args.batch_size, args)
                state, best_val = train_supervised(local, train_loader, val_loader, device, args.lr, args.local_epochs, args)
                local.load_state_dict(state)
                personal_models[site] = local.cpu()
                shared_states.append(personal_models[site].get_shared_state())
                weights.append(len(clients[site]["X_train"]))
                vals.append(best_val)
            global_shared[typ] = aggregate_weighted(shared_states, weights)
            history[typ].append(float(np.mean(vals)))

    for site, client in clients.items():
        model = personal_models[site]
        model.load_shared_state(global_shared[site_type(site)])
        model.to(device)
        res = evaluate_model(model, client, device, args)
        rows.append(result_row(site, res))
        examples[site] = res
    return pd.DataFrame(rows), examples, history


def run_fl_moe(
    clients: Dict[str, Dict[str, np.ndarray | MinMaxScaler]],
    input_dim: int,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, np.ndarray | float]], Dict[str, List[float]]]:
    grouped = {"wind": [], "solar": []}
    for site in clients:
        grouped[site_type(site)].append(site)
    rows = []
    examples = {}
    history = {"wind": [], "solar": []}
    models = {}
    local_models: Dict[str, nn.Module] = {}
    for typ in ["wind", "solar"]:
        if not grouped[typ]:
            continue
        global_model = create_moe_model(input_dim, args).to(device)
        shared_state = global_model.get_shared_state()
        for _ in range(args.rounds):
            states, weights, vals = [], [], []
            for site in grouped[typ]:
                train_loader, val_loader = make_loaders(clients[site], args.batch_size, args)
                if site not in local_models:
                    local_models[site] = create_moe_model(input_dim, args)
                local = local_models[site]
                local.load_shared_state(shared_state)
                local = local.to(device)
                opt = (
                    torch.optim.Adam(local.parameters(), lr=args.lr)
                    if args.moe_variant == "legacy"
                    else torch.optim.AdamW(local.parameters(), lr=args.lr, weight_decay=1e-4)
                )
                loss_fn = nn.MSELoss()
                amp_on = use_amp(args, device)
                scaler = torch.amp.GradScaler("cuda", enabled=amp_on)
                best_val = np.inf
                best_state = copy.deepcopy(local.state_dict())
                for _epoch in range(args.local_epochs):
                    local.train()
                    for xb, yb in train_loader:
                        xb, yb = move_batch(xb, device), move_batch(yb, device)
                        opt.zero_grad(set_to_none=True)
                        with torch.amp.autocast(device_type=device.type, enabled=amp_on):
                            if args.moe_variant == "legacy":
                                pred = local(xb)
                                loss = loss_fn(pred, yb)
                            else:
                                pred, aux = local(xb, return_aux=True)
                                loss = moe_objective(pred, yb, aux, local, args)
                        scaler.scale(loss).backward()
                        if args.moe_variant != "legacy":
                            scaler.unscale_(opt)
                            torch.nn.utils.clip_grad_norm_(local.parameters(), max_norm=1.0)
                        scaler.step(opt)
                        scaler.update()
                    local.eval()
                    total, n = 0.0, 0
                    with torch.no_grad():
                        for xb, yb in val_loader:
                            xb, yb = move_batch(xb, device), move_batch(yb, device)
                            with torch.amp.autocast(device_type=device.type, enabled=amp_on):
                                if args.moe_variant == "legacy":
                                    pred = local(xb)
                                    batch_loss = loss_fn(pred, yb)
                                else:
                                    pred, aux = local(xb, return_aux=True)
                                    batch_loss = moe_objective(pred, yb, aux, local, args)
                            total += batch_loss.item() * xb.size(0)
                            n += xb.size(0)
                    val_loss = total / max(n, 1)
                    if val_loss < best_val:
                        best_val = val_loss
                        best_state = copy.deepcopy(local.state_dict())
                local.load_state_dict(best_state)
                local_models[site] = copy.deepcopy(local).cpu()
                states.append(local_models[site].get_shared_state())
                weights.append(len(clients[site]["X_train"]))
                vals.append(best_val)
            shared_state = aggregate_weighted(states, weights)
            global_model.load_shared_state(shared_state)
            history[typ].append(float(np.mean(vals)))
        global_model.load_shared_state(shared_state)
        models[typ] = global_model
    for site, client in clients.items():
        model = copy.deepcopy(local_models.get(site, models[site_type(site)]))
        model.load_shared_state(models[site_type(site)].get_shared_state())
        adapt_loader = make_adaptation_loader(client, args.batch_size, args)
        model = personalize_moe(model, adapt_loader, device, args.personal_lr, args.personal_epochs, args)
        res = evaluate_model(model, client, device, args)
        rows.append(result_row(site, res))
        examples[site] = res
    return pd.DataFrame(rows), examples, history


def safe_wilcoxon(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    if len(diff) < 2 or np.allclose(diff, 0):
        return np.nan
    try:
        return float(wilcoxon(a, b, alternative="two-sided").pvalue)
    except ValueError:
        return np.nan


def safe_ttest(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return np.nan
    return float(ttest_rel(a, b, nan_policy="omit").pvalue)


def build_significance_table(results_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon in sorted(results_df["horizon"].unique()):
        sub = results_df[results_df["horizon"] == horizon]
        for metric in CORE_METRICS:
            pivot = sub.pivot(index="site", columns="method", values=metric)
            methods = [m for m in METHODS if m in pivot.columns]
            for a, b in itertools.combinations(methods, 2):
                pair = pivot[[a, b]].dropna()
                if pair.empty:
                    continue
                a_vals = pair[a].values
                b_vals = pair[b].values
                if metric in {"r2", "corr"}:
                    mean_diff = float(np.mean(a_vals - b_vals))
                    rel_improve = np.nan
                else:
                    mean_diff = float(np.mean(a_vals - b_vals))
                    rel_improve = float(np.mean((b_vals - a_vals) / np.maximum(np.abs(b_vals), 1e-8)) * 100.0)
                rows.append(
                    {
                        "horizon": horizon,
                        "metric": metric,
                        "method_a": a,
                        "method_b": b,
                        "n_sites": int(len(pair)),
                        "mean_diff_a_minus_b": mean_diff,
                        "relative_improvement_a_vs_b_pct": rel_improve,
                        "wilcoxon_p": safe_wilcoxon(a_vals, b_vals),
                        "ttest_p": safe_ttest(a_vals, b_vals),
                    }
                )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["significant_0_05"] = out["wilcoxon_p"] < 0.05
    return out


def build_rankings(results_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rank_rows = []
    win_rows = []
    for horizon in sorted(results_df["horizon"].unique()):
        sub = results_df[results_df["horizon"] == horizon]
        for metric in CORE_METRICS:
            pivot = sub.pivot(index="site", columns="method", values=metric)
            methods = [m for m in METHODS if m in pivot.columns]
            ascending = metric not in {"r2", "corr"}
            ranks = pivot.rank(axis=1, ascending=ascending, method="average")
            avg_rank = ranks.mean(axis=0)
            for method, value in avg_rank.items():
                rank_rows.append(
                    {
                        "horizon": horizon,
                        "metric": metric,
                        "method": method,
                        "avg_rank": float(value),
                    }
                )
            winners = pivot.idxmin(axis=1) if ascending else pivot.idxmax(axis=1)
            counts = winners.value_counts()
            for method in methods:
                win_rows.append(
                    {
                        "horizon": horizon,
                        "metric": metric,
                        "method": method,
                        "win_count": int(counts.get(method, 0)),
                    }
                )
    return pd.DataFrame(rank_rows), pd.DataFrame(win_rows)


def setup_plot_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#d1d5db",
            "axes.labelcolor": "#111827",
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "font.size": 10,
            "legend.frameon": False,
            "lines.linewidth": 2.2,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=320)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def method_color(method: str) -> str:
    return METHOD_COLORS.get(method, "#111827")


def method_marker(method: str) -> str:
    return METHOD_MARKERS.get(method, "o")


def ordered_methods(df: pd.DataFrame) -> List[str]:
    present = set(df["method"].dropna().unique())
    return [method for method in METHODS if method in present]


def annotate_matrix(ax: plt.Axes, values: np.ndarray, fmt: str = ".2f") -> None:
    finite = values[np.isfinite(values)]
    threshold = float(np.nanmean(finite)) if finite.size else 0.0
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if not np.isfinite(value):
                label = "-"
                color = "#374151"
            else:
                label = format(value, fmt)
                color = "white" if value > threshold else "#111827"
            ax.text(j, i, label, ha="center", va="center", fontsize=8, color=color)


def plot_method_horizon_curves(summary_df: pd.DataFrame, fig_dir: Path) -> None:
    for metric in ["rmse", "mae", "r2", "smape", "wape", "nrmse_range"]:
        fig, ax = plt.subplots(figsize=(8.8, 5.2))
        for method in METHODS:
            sub = summary_df[summary_df["method"] == method].sort_values("horizon")
            if sub.empty:
                continue
            ax.plot(
                sub["horizon"],
                sub[f"{metric}_mean"],
                marker=method_marker(method),
                color=method_color(method),
                label=method,
            )
            if f"{metric}_sem" in sub.columns:
                lower = sub[f"{metric}_mean"] - sub[f"{metric}_sem"]
                upper = sub[f"{metric}_mean"] + sub[f"{metric}_sem"]
                ax.fill_between(sub["horizon"], lower, upper, color=method_color(method), alpha=0.10, linewidth=0)
        ax.set_xticks(HORIZONS, [f"{h}h" for h in HORIZONS])
        ax.set_xlabel("Forecast horizon")
        ax.set_ylabel(METRIC_LABELS.get(metric, metric.upper()))
        ax.set_title(f"{METRIC_LABELS.get(metric, metric.upper())} across forecast horizons")
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(ncol=3, loc="best")
        save_figure(fig, fig_dir / f"{metric}_across_horizons.png")


def plot_type_bars(type_summary_df: pd.DataFrame, fig_dir: Path) -> None:
    methods = ordered_methods(type_summary_df)
    for horizon in HORIZONS:
        sub = type_summary_df[type_summary_df["horizon"] == horizon]
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), sharey=False)
        for idx, typ in enumerate(["wind", "solar"]):
            ax = axes[idx]
            view = sub[sub["type"] == typ].set_index("method").reindex(methods).reset_index()
            ax.bar(
                np.arange(len(view)),
                view["rmse_mean"],
                yerr=view["rmse_sem"],
                capsize=4,
                color=[method_color(m) for m in view["method"]],
                edgecolor="white",
                linewidth=0.8,
            )
            ax.set_title(f"{typ.capitalize()} sites, {horizon}h horizon")
            ax.set_xticks(np.arange(len(view)), view["method"], rotation=25, ha="right")
            ax.set_ylabel("RMSE")
            ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
            ax.spines[["top", "right"]].set_visible(False)
        save_figure(fig, fig_dir / f"type_rmse_h{horizon}.png")


def plot_metric_boxplots(results_df: pd.DataFrame, fig_dir: Path) -> None:
    methods = ordered_methods(results_df)
    for metric in ["rmse", "mae", "smape", "wape", "nrmse_range"]:
        fig, axes = plt.subplots(1, len(HORIZONS), figsize=(17, 4.8), sharey=False)
        axes = np.atleast_1d(axes)
        for idx, horizon in enumerate(HORIZONS):
            sub = results_df[results_df["horizon"] == horizon]
            horizon_methods = [method for method in methods if not sub[sub["method"] == method].empty]
            series = [sub[sub["method"] == method][metric].values for method in horizon_methods]
            bp = axes[idx].boxplot(series, tick_labels=horizon_methods, showfliers=False, patch_artist=True)
            for patch, method in zip(bp["boxes"], horizon_methods):
                patch.set_facecolor(method_color(method))
                patch.set_alpha(0.24)
                patch.set_edgecolor(method_color(method))
            for median in bp["medians"]:
                median.set_color("#111827")
                median.set_linewidth(1.8)
            axes[idx].set_title(f"{METRIC_LABELS.get(metric, metric.upper())} @ {horizon}h")
            axes[idx].tick_params(axis="x", rotation=25)
            axes[idx].grid(axis="y", color="#e5e7eb", linewidth=0.8)
            axes[idx].spines[["top", "right"]].set_visible(False)
        save_figure(fig, fig_dir / f"{metric}_boxplots.png")


def plot_site_heatmaps(results_df: pd.DataFrame, fig_dir: Path) -> None:
    methods = ordered_methods(results_df)
    for horizon in HORIZONS:
        sub = results_df[results_df["horizon"] == horizon]
        pivot = sub.pivot_table(index="site", columns="method", values="rmse").reindex(columns=methods)
        pivot = pivot.sort_index()
        fig, ax = plt.subplots(figsize=(10.5, max(5.5, len(pivot) * 0.36)))
        im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=25, ha="right")
        ax.set_yticks(range(len(pivot.index)), pivot.index)
        ax.set_title(f"Per-site RMSE heatmap ({horizon}h)")
        annotate_matrix(ax, pivot.values, ".1f")
        fig.colorbar(im, ax=ax, label="RMSE", fraction=0.028, pad=0.02)
        save_figure(fig, fig_dir / f"site_rmse_heatmap_h{horizon}.png")


def plot_significance_heatmap(sig_df: pd.DataFrame, fig_dir: Path) -> None:
    if sig_df.empty or "metric" not in sig_df.columns:
        return
    for metric in ["rmse", "smape", "wape"]:
        sub = sig_df[sig_df["metric"] == metric].copy()
        if sub.empty:
            continue
        sub["pair"] = sub["method_a"] + " vs " + sub["method_b"]
        pivot = sub.pivot(index="pair", columns="horizon", values="wilcoxon_p").sort_index()
        score = -np.log10(pivot.clip(lower=1e-6))
        fig, ax = plt.subplots(figsize=(8.5, max(4.2, len(pivot) * 0.42)))
        im = ax.imshow(score.values, aspect="auto", cmap="magma", vmin=0)
        ax.set_xticks(range(len(pivot.columns)), [f"{c}h" for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)), pivot.index)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                p_val = pivot.values[i, j]
                label = "*" if np.isfinite(p_val) and p_val < 0.05 else ""
                label = f"{label}\n{p_val:.3f}" if np.isfinite(p_val) else "-"
                ax.text(j, i, label, ha="center", va="center", fontsize=8, color="white")
        fig.colorbar(im, ax=ax, label="-log10(Wilcoxon p)", fraction=0.03, pad=0.02)
        ax.set_title(f"{METRIC_LABELS.get(metric, metric.upper())} pairwise significance")
        save_figure(fig, fig_dir / f"{metric}_significance_heatmap.png")


def plot_rank_heatmap(rank_df: pd.DataFrame, fig_dir: Path) -> None:
    for metric in ["rmse", "smape", "wape", "r2"]:
        sub = rank_df[rank_df["metric"] == metric]
        if sub.empty:
            continue
        pivot = sub.pivot(index="method", columns="horizon", values="avg_rank").reindex(ordered_methods(sub))
        fig, ax = plt.subplots(figsize=(6.8, 4.8))
        im = ax.imshow(pivot.values, aspect="auto", cmap="Blues_r")
        ax.set_xticks(range(len(pivot.columns)), [f"{c}h" for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)), pivot.index)
        annotate_matrix(ax, pivot.values, ".2f")
        fig.colorbar(im, ax=ax, label="Average rank", fraction=0.04, pad=0.02)
        ax.set_title(f"{METRIC_LABELS.get(metric, metric.upper())} average rank")
        save_figure(fig, fig_dir / f"{metric}_rank_heatmap.png")


def plot_win_counts(win_df: pd.DataFrame, fig_dir: Path) -> None:
    for metric in ["rmse", "smape", "wape", "r2"]:
        sub = win_df[win_df["metric"] == metric]
        if sub.empty:
            continue
        pivot = sub.pivot(index="method", columns="horizon", values="win_count").reindex(ordered_methods(sub))
        fig, ax = plt.subplots(figsize=(6.8, 4.8))
        im = ax.imshow(pivot.values, aspect="auto", cmap="Greens")
        ax.set_xticks(range(len(pivot.columns)), [f"{c}h" for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)), pivot.index)
        annotate_matrix(ax, pivot.values, ".0f")
        fig.colorbar(im, ax=ax, label="Win count", fraction=0.04, pad=0.02)
        ax.set_title(f"{METRIC_LABELS.get(metric, metric.upper())} win counts")
        save_figure(fig, fig_dir / f"{metric}_win_heatmap.png")


def plot_method_tradeoff(results_df: pd.DataFrame, fig_dir: Path) -> None:
    methods = ordered_methods(results_df)
    for horizon in HORIZONS:
        sub = results_df[results_df["horizon"] == horizon]
        fig, ax = plt.subplots(figsize=(7.6, 5.4))
        for method in methods:
            view = sub[sub["method"] == method]
            ax.scatter(
                view["rmse"],
                view["corr"],
                s=62,
                alpha=0.82,
                label=method,
                color=method_color(method),
                marker=method_marker(method),
                edgecolor="white",
                linewidth=0.5,
            )
        ax.set_xlabel("RMSE")
        ax.set_ylabel("Correlation")
        ax.set_title(f"RMSE-Correlation tradeoff ({horizon}h)")
        ax.grid(color="#e5e7eb", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(ncol=2)
        save_figure(fig, fig_dir / f"rmse_corr_tradeoff_h{horizon}.png")


def plot_interval_curves(summary_df: pd.DataFrame, fig_dir: Path) -> None:
    methods = ordered_methods(summary_df)
    for metric in ["picp_80", "pinaw_80"]:
        fig, ax = plt.subplots(figsize=(8.8, 5.2))
        for method in methods:
            sub = summary_df[summary_df["method"] == method].sort_values("horizon")
            if sub.empty:
                continue
            ax.plot(
                sub["horizon"],
                sub[f"{metric}_mean"],
                marker=method_marker(method),
                color=method_color(method),
                label=method,
            )
            sem_col = f"{metric}_sem"
            if sem_col in sub.columns:
                ax.fill_between(
                    sub["horizon"],
                    sub[f"{metric}_mean"] - sub[sem_col],
                    sub[f"{metric}_mean"] + sub[sem_col],
                    color=method_color(method),
                    alpha=0.10,
                    linewidth=0,
                )
        if metric == "picp_80":
            ax.axhline(0.80, color="#111827", linestyle="--", linewidth=1.2, label="Nominal 80%")
        ax.set_xticks(HORIZONS, [f"{h}h" for h in HORIZONS])
        ax.set_xlabel("Forecast horizon")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_title(f"{METRIC_LABELS[metric]} across forecast horizons")
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(ncol=3, loc="best")
        save_figure(fig, fig_dir / f"{metric}_across_horizons.png")


def plot_paper_summary(summary_df: pd.DataFrame, fig_dir: Path) -> None:
    metrics = ["rmse", "wape", "picp_80", "pinaw_80"]
    methods = ordered_methods(summary_df)
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.4))
    for ax, metric in zip(axes.ravel(), metrics):
        for method in methods:
            sub = summary_df[summary_df["method"] == method].sort_values("horizon")
            if sub.empty:
                continue
            ax.plot(
                sub["horizon"],
                sub[f"{metric}_mean"],
                marker=method_marker(method),
                color=method_color(method),
                label=method,
            )
        if metric == "picp_80":
            ax.axhline(0.80, color="#111827", linestyle="--", linewidth=1.1)
        ax.set_xticks(HORIZONS, [f"{h}h" for h in HORIZONS])
        ax.set_title(METRIC_LABELS.get(metric, metric.upper()))
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.03))
    save_figure(fig, fig_dir / "paper_metric_summary.png")


def plot_prediction_examples(
    example_store: Dict[Tuple[str, int, str], Dict[str, np.ndarray | float]],
    fig_dir: Path,
) -> None:
    methods = [method for method in METHODS if any(key[0] == method for key in example_store)]
    for horizon in HORIZONS:
        n_cols = 3
        n_rows = max(1, int(math.ceil(len(methods) / n_cols)))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15.5, 4.2 * n_rows), sharex=False)
        axes = np.array(axes).reshape(-1)
        for idx, method in enumerate(methods):
            key = (method, horizon, "solar")
            if key not in example_store:
                key = (method, horizon, "wind")
            if key not in example_store:
                continue
            res = example_store[key]
            n = min(240, len(res["y_true"]))
            ax = axes[idx]
            x_axis = np.arange(n)
            if "y_p10" in res and "y_p90" in res:
                ax.fill_between(
                    x_axis,
                    res["y_p10"][:n],
                    res["y_p90"][:n],
                    color=method_color(method),
                    alpha=0.15,
                    linewidth=0,
                    label="P10-P90",
                )
            ax.plot(x_axis, res["y_true"][:n], label="True", linewidth=1.8, color="#111827")
            ax.plot(x_axis, res["y_pred"][:n], label="Pred", linewidth=1.6, alpha=0.92, color=method_color(method))
            ax.set_title(f"{method} example ({horizon}h)")
            ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(loc="upper right")
        for ax in axes[len(methods) :]:
            ax.axis("off")
        save_figure(fig, fig_dir / f"prediction_examples_h{horizon}.png")


def export_scheduler_forecast(
    examples: Dict[str, Dict[str, np.ndarray | float]],
    output_path: Path,
) -> pd.DataFrame:
    rows = []
    for site, res in sorted(examples.items()):
        y_pred = np.maximum(0.0, np.asarray(res["y_pred"]).reshape(-1))
        y_p10 = np.maximum(0.0, np.asarray(res.get("y_p10", y_pred * 0.85)).reshape(-1))
        y_p90 = np.maximum(y_p10, np.maximum(0.0, np.asarray(res.get("y_p90", y_pred * 1.15)).reshape(-1)))
        for slot, (mean, low, high) in enumerate(zip(y_pred, y_p10, y_p90)):
            rows.append(
                {
                    "site_id": site,
                    "slot": int(slot),
                    "r_mean_kwh": float(mean),
                    "r_p10_kwh": float(low),
                    "r_p90_kwh": float(high),
                }
            )
    forecast_df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    forecast_df.to_csv(output_path, index=False, encoding="utf-8")
    return forecast_df


def write_analysis_summary(
    out_root: Path,
    summary_df: pd.DataFrame,
    rank_df: pd.DataFrame,
    win_df: pd.DataFrame,
    forecast_export_rows: int,
) -> None:
    fl_moe = summary_df[summary_df["method"] == "FL+MoE"].sort_values("horizon")
    headline_cols = [
        "horizon",
        "rmse_mean",
        "mae_mean",
        "r2_mean",
        "wape_mean",
        "picp_80_mean",
        "pinaw_80_mean",
    ]
    available_headline_cols = [col for col in headline_cols if col in fl_moe.columns]
    rank_focus = rank_df[rank_df["metric"].isin(["rmse", "wape", "picp_80", "pinaw_80"])] if not rank_df.empty else rank_df
    win_focus = win_df[win_df["metric"].isin(["rmse", "wape", "r2", "corr"])] if not win_df.empty else win_df
    lines = [
        "# Prediction Benchmark Analysis",
        "",
        "## Adopted Improvements",
        "",
        "- Added persistence and centralized-training baselines so FL results are compared against both a simple sanity check and a non-private upper-bound reference.",
        "- Added validation-residual P10/P90 forecast intervals and interval metrics, including PICP, PINAW, interval width, and quantile losses.",
        "- Exported the selected forecast as `forecast_for_scheduler.csv` using the `site_id,slot,r_mean_kwh,r_p10_kwh,r_p90_kwh` schema required by TaskScheduleSimu.",
        "- Regenerated publication-style PNG and PDF figures with consistent colors, annotated heatmaps, uncertainty bands, and a compact paper summary figure.",
        "",
        "## FL+MoE Topline",
        "",
        fl_moe[available_headline_cols].to_markdown(index=False) if not fl_moe.empty else "No FL+MoE rows were produced.",
        "",
        "## Average Rank Focus",
        "",
        rank_focus.to_markdown(index=False) if not rank_focus.empty else "No rank rows were produced.",
        "",
        "## Win Count Focus",
        "",
        win_focus.to_markdown(index=False) if not win_focus.empty else "No win-count rows were produced.",
        "",
        "## Scheduler Forecast Export",
        "",
        f"- Rows exported: {forecast_export_rows}",
        "- The interval is calibrated from validation residuals. This is stronger than a fixed 0.85/1.15 fallback, but it should still be described as empirical calibration rather than a full probabilistic forecasting model.",
    ]
    (out_root / "analysis.md").write_text("\n".join(lines), encoding="utf-8")


def run_one_method(
    method: str,
    clients: Dict[str, Dict[str, np.ndarray | MinMaxScaler]],
    input_dim: int,
    device: torch.device,
    args: argparse.Namespace,
):
    if method == "Persistence":
        return run_persistence(clients, input_dim, device, args)
    if method == "Local":
        return run_local(clients, input_dim, device, args)
    if method == "Centralized":
        return run_centralized(clients, input_dim, device, args)
    if method == "FedAvg":
        return run_fedavg(clients, input_dim, device, args)
    if method == "FedPer":
        return run_fedper(clients, input_dim, device, args)
    if method == "FL+MoE":
        return run_fl_moe(clients, input_dim, device, args)
    raise ValueError(f"Unknown method: {method}")


def run_benchmark(args: argparse.Namespace) -> None:
    global HORIZONS
    if args.horizons:
        HORIZONS = parse_csv_ints(args.horizons)
    active_methods = parse_csv_strings(args.methods, METHODS) if args.methods else METHODS
    primary_methods = [m for m in PRIMARY_METHODS if m in active_methods]
    auxiliary_methods = [m for m in AUXILIARY_METHODS if m in active_methods]
    set_seed(args.seed)
    setup_plot_style()
    device = resolve_device(args)
    args.pin_memory = bool(args.pin_memory and device.type == "cuda")
    root = Path(args.data_root)
    out_root = Path(args.output_dir)
    fig_dir = out_root / "figures"
    out_root.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    raw_sites = load_sites(root)
    prepared_sites = {site: make_common_features(df, site, args.feature_preset) for site, df in raw_sites.items()}
    prepared_sites, feature_cols = align_feature_space(prepared_sites)

    all_rows = []
    all_histories: Dict[str, Dict[int, Dict[str, List[float]]]] = {}
    example_store: Dict[Tuple[str, int, str], Dict[str, np.ndarray | float]] = {}
    scheduler_forecast_examples: Dict[str, Dict[str, np.ndarray | float]] = {}

    def build_horizon_clients(horizon: int) -> Dict[str, Dict[str, np.ndarray | MinMaxScaler]]:
        clients = {}
        for site, df in prepared_sites.items():
            if len(df) < args.seq_len + horizon + 72:
                continue
            clients[site] = split_and_scale(df, args.seq_len, horizon, feature_cols)
        return clients

    def run_method_group(methods: List[str]) -> None:
        nonlocal scheduler_forecast_examples
        input_dim = len(feature_cols)
        for horizon in HORIZONS:
            clients = build_horizon_clients(horizon)
            if not clients:
                continue

            for method in methods:
                if args.reset_seed_per_run:
                    set_seed(args.seed)
                metrics_df, examples, history = run_one_method(method, clients, input_dim, device, args)
                metrics_df["method"] = method
                metrics_df["horizon"] = horizon
                all_rows.append(metrics_df)
                all_histories.setdefault(method, {})[horizon] = history
                if (
                    args.export_scheduler_forecast
                    and method == args.scheduler_forecast_method
                    and horizon == args.scheduler_forecast_horizon
                ):
                    scheduler_forecast_examples = examples

                for typ in ["solar", "wind"]:
                    for site, res in examples.items():
                        if site_type(site) == typ:
                            example_store[(method, horizon, typ)] = res
                            break

    run_method_group(primary_methods)
    run_method_group(auxiliary_methods)

    results_df = pd.concat(all_rows, ignore_index=True)
    results_df = results_df[
        [
            "method",
            "horizon",
            "site",
            "type",
            *RESULT_METRICS,
            *AUX_RESULT_FIELDS,
        ]
    ]
    results_df.to_csv(out_root / "all_results.csv", index=False, encoding="utf-8")
    results_df[
        ["method", "horizon", "site", "type", *AUX_RESULT_FIELDS]
    ].to_csv(out_root / "postprocess_choices.csv", index=False, encoding="utf-8")

    summary_df = (
        results_df.groupby(["method", "horizon"])[RESULT_METRICS]
        .agg(["mean", "std", "sem"])
        .reset_index()
    )
    summary_df.columns = [
        "_".join([str(x) for x in col if x]).rstrip("_")
        for col in summary_df.columns.to_flat_index()
    ]
    summary_df = summary_df.sort_values(["horizon", "method"])
    summary_df.to_csv(out_root / "summary_by_method_horizon.csv", index=False, encoding="utf-8")

    type_summary_df = (
        results_df.groupby(["method", "horizon", "type"])[RESULT_METRICS]
        .agg(["mean", "std", "sem"])
        .reset_index()
    )
    type_summary_df.columns = [
        "_".join([str(x) for x in col if x]).rstrip("_")
        for col in type_summary_df.columns.to_flat_index()
    ]
    type_summary_df = type_summary_df.sort_values(["horizon", "type", "method"])
    type_summary_df.to_csv(out_root / "summary_by_method_horizon_type.csv", index=False, encoding="utf-8")

    sig_df = build_significance_table(results_df)
    sig_df.to_csv(out_root / "significance_tests.csv", index=False, encoding="utf-8")
    rank_df, win_df = build_rankings(results_df)
    rank_df.to_csv(out_root / "average_ranks.csv", index=False, encoding="utf-8")
    win_df.to_csv(out_root / "win_counts.csv", index=False, encoding="utf-8")

    with open(out_root / "training_histories.json", "w", encoding="utf-8") as f:
        json.dump(all_histories, f, ensure_ascii=False, indent=2)

    plot_method_horizon_curves(summary_df, fig_dir)
    plot_type_bars(type_summary_df, fig_dir)
    plot_metric_boxplots(results_df, fig_dir)
    plot_site_heatmaps(results_df, fig_dir)
    plot_significance_heatmap(sig_df, fig_dir)
    plot_rank_heatmap(rank_df, fig_dir)
    plot_win_counts(win_df, fig_dir)
    plot_method_tradeoff(results_df, fig_dir)
    plot_interval_curves(summary_df, fig_dir)
    plot_paper_summary(summary_df, fig_dir)
    plot_prediction_examples(example_store, fig_dir)

    forecast_export_path = out_root / "forecast_for_scheduler.csv"
    forecast_export_rows = 0
    if args.export_scheduler_forecast and scheduler_forecast_examples:
        forecast_df = export_scheduler_forecast(scheduler_forecast_examples, forecast_export_path)
        forecast_export_rows = len(forecast_df)
    write_analysis_summary(out_root, summary_df, rank_df, win_df, forecast_export_rows)

    report_lines = [
        "# Unified Benchmark Report",
        "",
        "## Settings",
        f"- seq_len: {args.seq_len}",
        f"- horizons: {HORIZONS}",
        f"- methods: {active_methods}",
        f"- device: {device}",
        f"- cuda_name: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'n/a'}",
        f"- amp: {use_amp(args, device)}",
        f"- num_workers: {args.num_workers}",
        f"- pin_memory: {args.pin_memory}",
        f"- postprocess: {args.postprocess}",
        f"- feature_preset: {args.feature_preset}",
        f"- reset_seed_per_run: {args.reset_seed_per_run}",
        f"- rounds: {args.rounds}",
        f"- local_epochs: {args.local_epochs}",
        f"- hidden_dim: {args.hidden_dim}",
        f"- num_layers: {args.num_layers}",
        f"- num_experts: {args.num_experts}",
        f"- moe_variant: {args.moe_variant}",
        f"- moe_top_k: {args.moe_top_k}",
        f"- moe_balance_weight: {args.moe_balance_weight}",
        f"- moe_peak_weight: {args.moe_peak_weight}",
        f"- moe_under_weight: {args.moe_under_weight}",
        f"- batch_size: {args.batch_size}",
        f"- lr: {args.lr}",
        f"- interval: validation-residual P10/P90, nominal coverage 80%",
        f"- scheduler_forecast_method: {args.scheduler_forecast_method}",
        f"- scheduler_forecast_horizon: {args.scheduler_forecast_horizon}",
        f"- scheduler_forecast_rows: {forecast_export_rows}",
        "",
        "## Mean metrics by method and horizon",
        summary_df.to_markdown(index=False),
        "",
        "## Mean metrics by method, horizon, and type",
        type_summary_df.to_markdown(index=False),
        "",
        "## Average ranks",
        rank_df.to_markdown(index=False),
        "",
        "## Win counts",
        win_df.to_markdown(index=False),
        "",
        "## Significance tests",
        sig_df.to_markdown(index=False) if not sig_df.empty else "No significance results available.",
    ]
    (out_root / "benchmark_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Unified benchmark finished. Results saved to: {out_root}")
    print(f"- {out_root / 'all_results.csv'}")
    print(f"- {out_root / 'summary_by_method_horizon.csv'}")
    print(f"- {out_root / 'summary_by_method_horizon_type.csv'}")
    print(f"- {out_root / 'significance_tests.csv'}")
    print(f"- {out_root / 'average_ranks.csv'}")
    print(f"- {out_root / 'win_counts.csv'}")
    print(f"- {out_root / 'postprocess_choices.csv'}")
    print(f"- {out_root / 'benchmark_report.md'}")
    print(f"- {out_root / 'analysis.md'}")
    if forecast_export_rows:
        print(f"- {forecast_export_path}")
    print(f"- {fig_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified benchmark for Local/FedAvg/FedPer/FL+MoE.")
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/Renewable-energy-generation-input-feature-variables-analysis-main/data_processed",
    )
    parser.add_argument("--output-dir", type=str, default="results/unified_benchmark")
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--horizons", type=str, default=None, help="Comma-separated horizons, e.g. 1,6,24.")
    parser.add_argument("--methods", type=str, default=None, help="Comma-separated methods to run.")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--personal-epochs", type=int, default=2)
    parser.add_argument("--personal-lr", type=float, default=5e-4)
    parser.add_argument(
        "--moe-variant",
        type=str,
        choices=["legacy", "enhanced"],
        default="legacy",
        help="Use the v4-stable FL+MoE route or the enhanced top-k/residual route.",
    )
    parser.add_argument("--moe-top-k", type=int, default=2)
    parser.add_argument("--moe-balance-weight", type=float, default=0.02)
    parser.add_argument("--moe-peak-weight", type=float, default=1.25)
    parser.add_argument("--moe-under-weight", type=float, default=0.35)
    parser.add_argument(
        "--export-scheduler-forecast",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export the selected method/horizon forecasts as TaskScheduleSimu forecast CSV.",
    )
    parser.add_argument("--scheduler-forecast-method", type=str, choices=METHODS, default="FL+MoE")
    parser.add_argument("--scheduler-forecast-horizon", type=int, choices=HORIZONS, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--cpu", action="store_true", help="Backward-compatible alias for --device cpu.")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--postprocess",
        type=str,
        choices=["none", "affine", "blend", "auto"],
        default="none",
        help="Validation-driven prediction calibration. Use auto for v6 optimized runs.",
    )
    parser.add_argument(
        "--feature-preset",
        type=str,
        choices=["v5", "enhanced"],
        default="v5",
        help="v5 keeps the stable benchmark features; enhanced adds exploratory wind/ramp features.",
    )
    parser.add_argument(
        "--reset-seed-per-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reset the seed before each method/horizon run to make results independent of method order.",
    )
    args = parser.parse_args()
    if args.cpu:
        args.device = "cpu"
    return args


if __name__ == "__main__":
    run_benchmark(parse_args())
