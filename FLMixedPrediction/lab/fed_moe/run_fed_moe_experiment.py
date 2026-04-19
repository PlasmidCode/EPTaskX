import argparse
import copy
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset


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
        raise ValueError("No numeric columns found for target detection.")
    return numeric_cols[-1]


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
        raise FileNotFoundError(f"No .xlsx site files found under {data_root}")
    return datasets


def site_index(site_name: str) -> int:
    try:
        return int(site_name.split("_")[-1])
    except ValueError:
        return 0


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
        months = out.index.month.values
        dayofyear = out.index.dayofyear.values
        out["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
        out["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
        out["month_sin"] = np.sin(2 * np.pi * months / 12.0)
        out["month_cos"] = np.cos(2 * np.pi * months / 12.0)
        out["doy_sin"] = np.sin(2 * np.pi * dayofyear / 365.0)
        out["doy_cos"] = np.cos(2 * np.pi * dayofyear / 365.0)
    else:
        out["hour_sin"] = 0.0
        out["hour_cos"] = 0.0
        out["month_sin"] = 0.0
        out["month_cos"] = 0.0
        out["doy_sin"] = 0.0
        out["doy_cos"] = 0.0

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
    target_horizon: int,
    feature_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    vals_x = df[feature_cols].values.astype(np.float32)
    vals_y = df["power"].values.astype(np.float32).reshape(-1, 1)
    X, y = [], []
    max_i = len(df) - target_horizon + 1
    for i in range(seq_len, max_i):
        X.append(vals_x[i - seq_len : i])
        y.append(vals_y[i + target_horizon - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


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
        w = self.gate(shared)
        outs = []
        for expert in self.experts:
            outs.append(expert(shared))
        stacked = torch.stack(outs, dim=1)
        pred = torch.sum(stacked * w.unsqueeze(-1), dim=1)
        return pred

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


def train_local(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    lr: float,
    epochs: int,
) -> Tuple[Dict[str, torch.Tensor], float, int]:
    model = model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_val = np.inf
    best_state = copy.deepcopy(model.state_dict())
    for _ in range(epochs):
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
                loss = loss_fn(pred, yb).item()
                total += loss * xb.size(0)
                n += xb.size(0)
        val = total / max(n, 1)
        if val < best_val:
            best_val = val
            best_state = copy.deepcopy(model.state_dict())
        model.train()
    return best_state, float(best_val), len(train_loader.dataset)


def fedavg_weighted(states: List[Dict[str, torch.Tensor]], weights: List[int]) -> Dict[str, torch.Tensor]:
    total = float(sum(weights))
    agg = {}
    for k in states[0].keys():
        agg[k] = sum(s[k] * (w / total) for s, w in zip(states, weights))
    return agg


def site_type(site_name: str) -> str:
    return "wind" if "wind" in site_name else "solar"


def evaluate_site(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler_y: MinMaxScaler,
    device: torch.device,
) -> Dict[str, np.ndarray | float]:
    model.eval()
    with torch.no_grad():
        x = torch.tensor(X_test, dtype=torch.float32, device=device)
        pred_scaled = model(x).cpu().numpy()
    y_true = scaler_y.inverse_transform(y_test)
    y_pred = scaler_y.inverse_transform(pred_scaled)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "y_true": y_true.flatten(),
        "y_pred": y_pred.flatten(),
    }


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
    X_train_2d = X_train.reshape(-1, X_train.shape[-1])
    x_scaler.fit(X_train_2d)
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


def make_adaptation_loader(client: Dict[str, np.ndarray | MinMaxScaler], batch_size: int) -> DataLoader:
    X = np.concatenate([client["X_train"], client["X_val"]], axis=0)
    y = np.concatenate([client["y_train"], client["y_val"]], axis=0)
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


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


def run_experiment(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    root = Path(args.data_root)
    out_root = Path(args.output_dir)
    fig_dir = out_root / "figures"
    out_root.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    datasets_raw = load_sites(root)
    datasets = {k: make_common_features(v, k) for k, v in datasets_raw.items()}
    datasets, feature_cols = align_feature_space(datasets)

    clients: Dict[str, Dict[str, np.ndarray | MinMaxScaler]] = {}
    for sn, df in datasets.items():
        if len(df) < args.seq_len + args.horizon + 48:
            continue
        clients[sn] = split_and_scale(df, args.seq_len, args.horizon, feature_cols)
    if not clients:
        raise RuntimeError("No valid clients after preprocessing.")

    grouped = {"wind": [], "solar": []}
    for sn in clients.keys():
        grouped[site_type(sn)].append(sn)

    globals_by_type: Dict[str, MoEForecast] = {}
    local_models: Dict[str, MoEForecast] = {}
    history: Dict[str, List[float]] = {"wind": [], "solar": []}

    for t in ["wind", "solar"]:
        if not grouped[t]:
            continue
        global_model = MoEForecast(len(feature_cols), args.hidden_dim, args.num_experts)
        global_model.to(device)
        shared_state = global_model.get_shared_state()
        for _round in range(args.rounds):
            n_sel = max(1, int(np.ceil(len(grouped[t]) * args.client_fraction)))
            selected = list(np.random.choice(grouped[t], n_sel, replace=False))
            states, weights, vals = [], [], []
            for sn in selected:
                c = clients[sn]
                train_ds = TensorDataset(
                    torch.tensor(c["X_train"], dtype=torch.float32),
                    torch.tensor(c["y_train"], dtype=torch.float32),
                )
                val_ds = TensorDataset(
                    torch.tensor(c["X_val"], dtype=torch.float32),
                    torch.tensor(c["y_val"], dtype=torch.float32),
                )
                train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
                val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
                if sn not in local_models:
                    local_models[sn] = MoEForecast(len(feature_cols), args.hidden_dim, args.num_experts)
                local_model = local_models[sn]
                local_model.load_shared_state(shared_state)
                state, best_val, n_train = train_local(
                    local_model,
                    train_loader,
                    val_loader,
                    device,
                    args.lr,
                    args.local_epochs,
                )
                local_model.load_state_dict(state)
                local_models[sn] = copy.deepcopy(local_model).cpu()
                states.append(local_models[sn].get_shared_state())
                weights.append(n_train)
                vals.append(best_val)
            shared_state = fedavg_weighted(states, weights)
            global_model.load_shared_state(shared_state)
            history[t].append(float(np.mean(vals)))
        global_model.load_shared_state(shared_state)
        globals_by_type[t] = global_model

    rows = []
    pred_examples = {}
    for sn, c in clients.items():
        t = site_type(sn)
        model = copy.deepcopy(local_models.get(sn, globals_by_type[t]))
        model.load_shared_state(globals_by_type[t].get_shared_state())
        adapt_loader = make_adaptation_loader(c, args.batch_size)
        model = personalize_moe(model, adapt_loader, device, args.personal_lr, args.personal_epochs)
        res = evaluate_site(model.to(device), c["X_test"], c["y_test"], c["y_scaler"], device)
        rows.append(
            {
                "site": sn,
                "type": t,
                "mse": res["mse"],
                "rmse": res["rmse"],
                "mae": res["mae"],
                "r2": res["r2"],
            }
        )
        if t not in pred_examples:
            pred_examples[t] = res

    metrics_df = pd.DataFrame(rows).sort_values(["type", "site"]).reset_index(drop=True)
    metrics_df.to_csv(out_root / "site_metrics.csv", index=False, encoding="utf-8")

    summary_df = metrics_df.groupby("type")[["mse", "rmse", "mae", "r2"]].mean().reset_index()
    summary_df.to_csv(out_root / "type_summary.csv", index=False, encoding="utf-8")

    with open(out_root / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    plt.figure(figsize=(12, 5))
    for t in ["wind", "solar"]:
        if history[t]:
            plt.plot(range(1, len(history[t]) + 1), history[t], marker="o", label=t)
    plt.xlabel("Federated round")
    plt.ylabel("Mean best val MSE (scaled)")
    plt.title("FL+MoE validation trend")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "fl_moe_round_curve.png", dpi=220)
    plt.close()

    plt.figure(figsize=(14, 6))
    colors = ["#1f77b4" if t == "wind" else "#ff7f0e" for t in metrics_df["type"]]
    plt.bar(metrics_df["site"], metrics_df["rmse"], color=colors)
    plt.xticks(rotation=60, ha="right")
    plt.ylabel("RMSE")
    plt.title("Per-site RMSE (FL+MoE)")
    plt.tight_layout()
    plt.savefig(fig_dir / "fl_moe_site_rmse.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    x = np.arange(len(summary_df))
    w = 0.35
    plt.bar(x - w / 2, summary_df["rmse"], width=w, label="RMSE")
    plt.bar(x + w / 2, summary_df["mae"], width=w, label="MAE")
    plt.xticks(x, summary_df["type"])
    plt.ylabel("Error")
    plt.title("Type-level error summary (FL+MoE)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "fl_moe_type_summary.png", dpi=220)
    plt.close()

    if "wind" in pred_examples or "solar" in pred_examples:
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False)
        idx = 0
        for t in ["wind", "solar"]:
            if t not in pred_examples:
                continue
            res = pred_examples[t]
            n = min(240, len(res["y_true"]))
            axes[idx].plot(res["y_true"][:n], label="True", linewidth=1.8)
            axes[idx].plot(res["y_pred"][:n], label="Pred", linewidth=1.6, alpha=0.85)
            axes[idx].set_title(f"{t.capitalize()} example prediction")
            axes[idx].legend()
            axes[idx].grid(alpha=0.3)
            idx += 1
        plt.tight_layout()
        plt.savefig(fig_dir / "fl_moe_examples.png", dpi=220)
        plt.close()

    report_lines = [
        "# FL+MoE Cross-domain Renewable Forecasting Results",
        "",
        "## Settings",
        f"- seq_len: {args.seq_len}",
        f"- horizon: {args.horizon}",
        f"- rounds: {args.rounds}",
        f"- local_epochs: {args.local_epochs}",
        f"- num_experts: {args.num_experts}",
        f"- hidden_dim: {args.hidden_dim}",
        f"- batch_size: {args.batch_size}",
        f"- lr: {args.lr}",
        f"- client_fraction: {args.client_fraction}",
        "",
        "## Type-level Mean Metrics",
        summary_df.to_markdown(index=False),
        "",
        "## Per-site Metrics",
        metrics_df.to_markdown(index=False),
    ]
    (out_root / "results_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Done. Results saved to: {out_root}")
    print("Key files:")
    print(f"- {out_root / 'site_metrics.csv'}")
    print(f"- {out_root / 'type_summary.csv'}")
    print(f"- {out_root / 'results_report.md'}")
    print(f"- {fig_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FL+MoE experiment for wind+solar sites.")
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/Renewable-energy-generation-input-feature-variables-analysis-main/data_processed",
    )
    parser.add_argument("--output-dir", type=str, default="results/fed_moe_experiment")
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--client-fraction", type=float, default=1.0)
    parser.add_argument("--personal-epochs", type=int, default=2)
    parser.add_argument("--personal-lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
