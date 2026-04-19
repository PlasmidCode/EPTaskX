import argparse
import copy
import itertools
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset


METHODS = ["Local", "FedAvg", "FedPer", "FL+MoE"]
HORIZONS = [1, 6, 24]
CORE_METRICS = ["rmse", "mae", "r2", "smape", "wape", "nrmse_range", "corr"]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def make_common_features(df: pd.DataFrame, site_name: str) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = [col for col in out.columns if pd.api.types.is_numeric_dtype(out[col])]
    for col in numeric_cols:
        low = out[col].quantile(0.01)
        high = out[col].quantile(0.99)
        out[col] = out[col].clip(lower=low, upper=high)

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
    out["site_id_norm"] = site_index(site_name) / 10.0
    out["power_lag_1"] = out["power"].shift(1)
    out["power_lag_2"] = out["power"].shift(2)
    out["power_lag_6"] = out["power"].shift(6)
    out["power_lag_24"] = out["power"].shift(24)
    out["power_roll_mean_6"] = out["power"].rolling(6).mean()
    out["power_roll_std_6"] = out["power"].rolling(6).std()
    out["power_roll_mean_24"] = out["power"].rolling(24).mean()
    out["power_roll_std_24"] = out["power"].rolling(24).std()
    for col in weather_feature_columns(out):
        out[f"{col}_lag_1"] = out[col].shift(1)
        out[f"{col}_lag_6"] = out[col].shift(6)
    out = out.dropna().copy()
    return out


def build_sequences(
    df: pd.DataFrame,
    seq_len: int,
    horizon: int,
    feature_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    vals_x = df[feature_cols].values.astype(np.float32)
    vals_y = df["power"].values.astype(np.float32).reshape(-1, 1)
    X, y = [], []
    for i in range(seq_len, len(df) - horizon + 1):
        X.append(vals_x[i - seq_len : i])
        y.append(vals_y[i + horizon - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def split_and_scale(
    df: pd.DataFrame,
    seq_len: int,
    horizon: int,
    feature_cols: List[str],
) -> Dict[str, np.ndarray | MinMaxScaler]:
    X, y = build_sequences(df, seq_len, horizon, feature_cols)
    n = len(X)
    s1, s2 = int(n * 0.7), int(n * 0.85)
    X_train, X_val, X_test = X[:s1], X[s1:s2], X[s2:]
    y_train, y_val, y_test = y[:s1], y[s1:s2], y[s2:]

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


class MoEForecast(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_experts: int = 4):
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


def make_loaders(client: Dict[str, np.ndarray | MinMaxScaler], batch_size: int) -> Tuple[DataLoader, DataLoader]:
    train_ds = TensorDataset(
        torch.tensor(client["X_train"], dtype=torch.float32),
        torch.tensor(client["y_train"], dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(client["X_val"], dtype=torch.float32),
        torch.tensor(client["y_val"], dtype=torch.float32),
    )
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
    )


def make_adaptation_loader(client: Dict[str, np.ndarray | MinMaxScaler], batch_size: int) -> DataLoader:
    X = np.concatenate([client["X_train"], client["X_val"]], axis=0)
    y = np.concatenate([client["y_train"], client["y_val"]], axis=0)
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


def train_supervised(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    lr: float,
    epochs: int,
) -> Tuple[Dict[str, torch.Tensor], float]:
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_val = np.inf
    best_state = copy.deepcopy(model.state_dict())
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                total += loss_fn(pred, yb).item() * xb.size(0)
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


def evaluate_model(
    model: nn.Module,
    client: Dict[str, np.ndarray | MinMaxScaler],
    device: torch.device,
) -> Dict[str, np.ndarray | float]:
    model.eval()
    with torch.no_grad():
        pred_scaled = model(
            torch.tensor(client["X_test"], dtype=torch.float32, device=device)
        ).cpu().numpy()
    y_true = client["y_scaler"].inverse_transform(client["y_test"])
    y_pred = client["y_scaler"].inverse_transform(pred_scaled)
    metrics = regression_metrics(y_true, y_pred)
    metrics["y_true"] = y_true.flatten()
    metrics["y_pred"] = y_pred.flatten()
    return metrics


def personalize_moe(
    model: MoEForecast,
    adapt_loader: DataLoader,
    device: torch.device,
    lr: float,
    epochs: int,
) -> MoEForecast:
    model = copy.deepcopy(model).to(device)
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("gate.") or name.startswith("experts.")
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in adapt_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
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
        train_loader, val_loader = make_loaders(client, args.batch_size)
        model = LSTMForecast(input_dim, args.hidden_dim, args.num_layers)
        state, best_val = train_supervised(model, train_loader, val_loader, device, args.lr, epochs)
        model.load_state_dict(state)
        model.to(device)
        res = evaluate_model(model, client, device)
        rows.append(
            {
                "site": site,
                "type": site_type(site),
                **{k: res[k] for k in ["mse", "rmse", "mae", "r2", "mbe", "smape", "wape", "nrmse_range", "nmae_range", "corr"]},
            }
        )
        examples[site] = res
        history[site] = [best_val]
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
                train_loader, val_loader = make_loaders(clients[site], args.batch_size)
                local = LSTMForecast(input_dim, args.hidden_dim, args.num_layers)
                local.load_state_dict(copy.deepcopy(global_model.state_dict()))
                state, best_val = train_supervised(local, train_loader, val_loader, device, args.lr, args.local_epochs)
                states.append(state)
                weights.append(len(clients[site]["X_train"]))
                vals.append(best_val)
            global_model.load_state_dict(aggregate_weighted(states, weights))
            history[typ].append(float(np.mean(vals)))
        models[typ] = global_model
    for site, client in clients.items():
        res = evaluate_model(models[site_type(site)], client, device)
        rows.append(
            {
                "site": site,
                "type": site_type(site),
                **{k: res[k] for k in ["mse", "rmse", "mae", "r2", "mbe", "smape", "wape", "nrmse_range", "nmae_range", "corr"]},
            }
        )
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
                train_loader, val_loader = make_loaders(clients[site], args.batch_size)
                state, best_val = train_supervised(local, train_loader, val_loader, device, args.lr, args.local_epochs)
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
        res = evaluate_model(model, client, device)
        rows.append(
            {
                "site": site,
                "type": site_type(site),
                **{k: res[k] for k in ["mse", "rmse", "mae", "r2", "mbe", "smape", "wape", "nrmse_range", "nmae_range", "corr"]},
            }
        )
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
    local_models: Dict[str, MoEForecast] = {}
    for typ in ["wind", "solar"]:
        if not grouped[typ]:
            continue
        global_model = MoEForecast(input_dim, args.hidden_dim, args.num_experts).to(device)
        shared_state = global_model.get_shared_state()
        for _ in range(args.rounds):
            states, weights, vals = [], [], []
            for site in grouped[typ]:
                train_loader, val_loader = make_loaders(clients[site], args.batch_size)
                if site not in local_models:
                    local_models[site] = MoEForecast(input_dim, args.hidden_dim, args.num_experts)
                local = local_models[site]
                local.load_shared_state(shared_state)
                state, best_val = train_supervised(local, train_loader, val_loader, device, args.lr, args.local_epochs)
                local.load_state_dict(state)
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
        adapt_loader = make_adaptation_loader(client, args.batch_size)
        model = personalize_moe(model, adapt_loader, device, args.personal_lr, args.personal_epochs)
        res = evaluate_model(model, client, device)
        rows.append(
            {
                "site": site,
                "type": site_type(site),
                **{k: res[k] for k in ["mse", "rmse", "mae", "r2", "mbe", "smape", "wape", "nrmse_range", "nmae_range", "corr"]},
            }
        )
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
            for a, b in itertools.combinations(METHODS, 2):
                pair = pivot[[a, b]].dropna()
                if pair.empty:
                    continue
                a_vals = pair[a].values
                b_vals = pair[b].values
                if metric == "r2":
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
            for method in METHODS:
                win_rows.append(
                    {
                        "horizon": horizon,
                        "metric": metric,
                        "method": method,
                        "win_count": int(counts.get(method, 0)),
                    }
                )
    return pd.DataFrame(rank_rows), pd.DataFrame(win_rows)


def plot_method_horizon_curves(summary_df: pd.DataFrame, fig_dir: Path) -> None:
    for metric in ["rmse", "mae", "r2", "smape", "wape", "nrmse_range"]:
        plt.figure(figsize=(8, 5))
        for method in METHODS:
            sub = summary_df[summary_df["method"] == method].sort_values("horizon")
            plt.plot(sub["horizon"], sub[f"{metric}_mean"], marker="o", linewidth=2, label=method)
            if f"{metric}_sem" in sub.columns:
                lower = sub[f"{metric}_mean"] - sub[f"{metric}_sem"]
                upper = sub[f"{metric}_mean"] + sub[f"{metric}_sem"]
                plt.fill_between(sub["horizon"], lower, upper, alpha=0.12)
        plt.xticks(HORIZONS, [f"{h}h" for h in HORIZONS])
        plt.xlabel("Forecast horizon")
        plt.ylabel(metric.upper())
        plt.title(f"{metric.upper()} across horizons")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / f"{metric}_across_horizons.png", dpi=220)
        plt.close()


def plot_type_bars(type_summary_df: pd.DataFrame, fig_dir: Path) -> None:
    for horizon in HORIZONS:
        sub = type_summary_df[type_summary_df["horizon"] == horizon]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
        for idx, typ in enumerate(["wind", "solar"]):
            view = sub[sub["type"] == typ]
            axes[idx].bar(
                view["method"],
                view["rmse_mean"],
                yerr=view["rmse_sem"],
                capsize=4,
                color=["#4e79a7", "#f28e2b", "#59a14f", "#e15759"],
            )
            axes[idx].set_title(f"{typ.capitalize()} RMSE at {horizon}h")
            axes[idx].tick_params(axis="x", rotation=20)
            axes[idx].grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(fig_dir / f"type_rmse_h{horizon}.png", dpi=220)
        plt.close()


def plot_metric_boxplots(results_df: pd.DataFrame, fig_dir: Path) -> None:
    for metric in ["rmse", "mae", "smape", "wape", "nrmse_range"]:
        fig, axes = plt.subplots(1, len(HORIZONS), figsize=(15, 4.5), sharey=False)
        for idx, horizon in enumerate(HORIZONS):
            sub = results_df[results_df["horizon"] == horizon]
            series = [sub[sub["method"] == method][metric].values for method in METHODS]
            axes[idx].boxplot(series, tick_labels=METHODS, showfliers=False)
            axes[idx].set_title(f"{metric.upper()} @ {horizon}h")
            axes[idx].tick_params(axis="x", rotation=20)
            axes[idx].grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(fig_dir / f"{metric}_boxplots.png", dpi=220)
        plt.close()


def plot_site_heatmaps(results_df: pd.DataFrame, fig_dir: Path) -> None:
    for horizon in HORIZONS:
        sub = results_df[results_df["horizon"] == horizon]
        pivot = sub.pivot_table(index="site", columns="method", values="rmse")
        plt.figure(figsize=(8, max(5, len(pivot) * 0.35)))
        plt.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
        plt.xticks(range(len(pivot.columns)), pivot.columns)
        plt.yticks(range(len(pivot.index)), pivot.index)
        plt.colorbar(label="RMSE")
        plt.title(f"Per-site RMSE heatmap ({horizon}h)")
        plt.tight_layout()
        plt.savefig(fig_dir / f"site_rmse_heatmap_h{horizon}.png", dpi=220)
        plt.close()


def plot_significance_heatmap(sig_df: pd.DataFrame, fig_dir: Path) -> None:
    for metric in ["rmse", "smape", "wape"]:
        sub = sig_df[sig_df["metric"] == metric].copy()
        if sub.empty:
            continue
        sub["pair"] = sub["method_a"] + " vs " + sub["method_b"]
        pivot = sub.pivot(index="pair", columns="horizon", values="wilcoxon_p").sort_index()
        plt.figure(figsize=(8, max(4, len(pivot) * 0.45)))
        finite_vals = pivot.values[np.isfinite(pivot.values)]
        vmax = float(np.max(finite_vals)) if finite_vals.size else 0.05
        plt.imshow(pivot.values, aspect="auto", cmap="viridis_r", vmin=0, vmax=max(0.05, vmax))
        plt.xticks(range(len(pivot.columns)), [f"{c}h" for c in pivot.columns])
        plt.yticks(range(len(pivot.index)), pivot.index)
        plt.colorbar(label="Wilcoxon p-value")
        plt.title(f"{metric.upper()} significance heatmap")
        plt.tight_layout()
        plt.savefig(fig_dir / f"{metric}_significance_heatmap.png", dpi=220)
        plt.close()


def plot_rank_heatmap(rank_df: pd.DataFrame, fig_dir: Path) -> None:
    for metric in ["rmse", "smape", "wape", "r2"]:
        sub = rank_df[rank_df["metric"] == metric]
        if sub.empty:
            continue
        pivot = sub.pivot(index="method", columns="horizon", values="avg_rank").reindex(METHODS)
        plt.figure(figsize=(6, 4))
        plt.imshow(pivot.values, aspect="auto", cmap="Blues_r")
        plt.xticks(range(len(pivot.columns)), [f"{c}h" for c in pivot.columns])
        plt.yticks(range(len(pivot.index)), pivot.index)
        plt.colorbar(label="Average rank")
        plt.title(f"{metric.upper()} average-rank heatmap")
        plt.tight_layout()
        plt.savefig(fig_dir / f"{metric}_rank_heatmap.png", dpi=220)
        plt.close()


def plot_win_counts(win_df: pd.DataFrame, fig_dir: Path) -> None:
    for metric in ["rmse", "smape", "wape", "r2"]:
        sub = win_df[win_df["metric"] == metric]
        if sub.empty:
            continue
        pivot = sub.pivot(index="method", columns="horizon", values="win_count").reindex(METHODS)
        plt.figure(figsize=(6, 4))
        plt.imshow(pivot.values, aspect="auto", cmap="Greens")
        plt.xticks(range(len(pivot.columns)), [f"{c}h" for c in pivot.columns])
        plt.yticks(range(len(pivot.index)), pivot.index)
        plt.colorbar(label="Win count")
        plt.title(f"{metric.upper()} win-count heatmap")
        plt.tight_layout()
        plt.savefig(fig_dir / f"{metric}_win_heatmap.png", dpi=220)
        plt.close()


def plot_method_tradeoff(results_df: pd.DataFrame, fig_dir: Path) -> None:
    for horizon in HORIZONS:
        sub = results_df[results_df["horizon"] == horizon]
        plt.figure(figsize=(7, 5))
        for method in METHODS:
            view = sub[sub["method"] == method]
            plt.scatter(view["rmse"], view["corr"], s=55, alpha=0.75, label=method)
        plt.xlabel("RMSE")
        plt.ylabel("Correlation")
        plt.title(f"RMSE-Correlation tradeoff ({horizon}h)")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / f"rmse_corr_tradeoff_h{horizon}.png", dpi=220)
        plt.close()


def plot_prediction_examples(
    example_store: Dict[Tuple[str, int, str], Dict[str, np.ndarray | float]],
    fig_dir: Path,
) -> None:
    for horizon in HORIZONS:
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=False)
        for idx, method in enumerate(METHODS):
            key = (method, horizon, "solar")
            if key not in example_store:
                key = (method, horizon, "wind")
            if key not in example_store:
                continue
            res = example_store[key]
            n = min(240, len(res["y_true"]))
            ax = axes[idx // 2, idx % 2]
            ax.plot(res["y_true"][:n], label="True", linewidth=1.8)
            ax.plot(res["y_pred"][:n], label="Pred", linewidth=1.5, alpha=0.85)
            ax.set_title(f"{method} example ({horizon}h)")
            ax.grid(alpha=0.3)
            ax.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / f"prediction_examples_h{horizon}.png", dpi=220)
        plt.close()


def run_one_method(
    method: str,
    clients: Dict[str, Dict[str, np.ndarray | MinMaxScaler]],
    input_dim: int,
    device: torch.device,
    args: argparse.Namespace,
):
    if method == "Local":
        return run_local(clients, input_dim, device, args)
    if method == "FedAvg":
        return run_fedavg(clients, input_dim, device, args)
    if method == "FedPer":
        return run_fedper(clients, input_dim, device, args)
    if method == "FL+MoE":
        return run_fl_moe(clients, input_dim, device, args)
    raise ValueError(f"Unknown method: {method}")


def run_benchmark(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    root = Path(args.data_root)
    out_root = Path(args.output_dir)
    fig_dir = out_root / "figures"
    out_root.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    raw_sites = load_sites(root)
    prepared_sites = {site: make_common_features(df, site) for site, df in raw_sites.items()}
    prepared_sites, feature_cols = align_feature_space(prepared_sites)

    all_rows = []
    all_histories: Dict[str, Dict[int, Dict[str, List[float]]]] = {}
    example_store: Dict[Tuple[str, int, str], Dict[str, np.ndarray | float]] = {}

    for horizon in HORIZONS:
        clients = {}
        for site, df in prepared_sites.items():
            if len(df) < args.seq_len + horizon + 72:
                continue
            clients[site] = split_and_scale(df, args.seq_len, horizon, feature_cols)
        if not clients:
            continue

        input_dim = len(feature_cols)
        for method in METHODS:
            metrics_df, examples, history = run_one_method(method, clients, input_dim, device, args)
            metrics_df["method"] = method
            metrics_df["horizon"] = horizon
            all_rows.append(metrics_df)
            all_histories.setdefault(method, {})[horizon] = history

            for typ in ["solar", "wind"]:
                for site, res in examples.items():
                    if site_type(site) == typ:
                        example_store[(method, horizon, typ)] = res
                        break

    results_df = pd.concat(all_rows, ignore_index=True)
    results_df = results_df[
        [
            "method",
            "horizon",
            "site",
            "type",
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
    ]
    results_df.to_csv(out_root / "all_results.csv", index=False, encoding="utf-8")

    summary_df = (
        results_df.groupby(["method", "horizon"])[["mse", "rmse", "mae", "r2", "mbe", "smape", "wape", "nrmse_range", "nmae_range", "corr"]]
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
        results_df.groupby(["method", "horizon", "type"])[["mse", "rmse", "mae", "r2", "mbe", "smape", "wape", "nrmse_range", "nmae_range", "corr"]]
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
    plot_prediction_examples(example_store, fig_dir)

    report_lines = [
        "# Unified Benchmark Report",
        "",
        "## Settings",
        f"- seq_len: {args.seq_len}",
        f"- horizons: {HORIZONS}",
        f"- methods: {METHODS}",
        f"- rounds: {args.rounds}",
        f"- local_epochs: {args.local_epochs}",
        f"- hidden_dim: {args.hidden_dim}",
        f"- num_layers: {args.num_layers}",
        f"- num_experts: {args.num_experts}",
        f"- batch_size: {args.batch_size}",
        f"- lr: {args.lr}",
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
    print(f"- {out_root / 'benchmark_report.md'}")
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
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--personal-epochs", type=int, default=2)
    parser.add_argument("--personal-lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_benchmark(parse_args())
