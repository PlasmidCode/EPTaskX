from pathlib import Path
from typing import List, Tuple
import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset, DataLoader

def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    tcol = 'timestamp' if 'timestamp' in df.columns else df.columns[0]
    df[tcol] = pd.to_datetime(df[tcol])
    df = df.sort_values(tcol).set_index(tcol)
    assert 'power' in df.columns, f"{path} must include a 'power' column."
    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    return df

def make_windows(df: pd.DataFrame, seq: int, horizon: int, feature_cols: List[str]):
    vals = df[feature_cols + ['power']].values.astype('float32')
    X, y = [], []
    F = len(feature_cols)
    for i in range(seq, len(vals) - horizon + 1):
        X.append(vals[i - seq:i, :F])
        y.append(vals[i + horizon - 1, -1])
    return np.asarray(X), np.asarray(y)

class SeqDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X); self.y = torch.from_numpy(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

def split_train_val(X, y, val_ratio=0.2, seed=42):
    n = len(y); idx = np.arange(n); rng = np.random.default_rng(seed); rng.shuffle(idx)
    nv = int(n * val_ratio); vidx, tidx = idx[:nv], idx[nv:]
    return (X[tidx], y[tidx]), (X[vidx], y[vidx])

def build_loaders(df: pd.DataFrame, seq: int, horizon: int, batch: int):
    feature_cols = [c for c in df.columns if c != 'power']
    if not feature_cols:
        df = df.copy(); df['power_lag1'] = df['power'].shift(1).bfill(); feature_cols = ['power_lag1']
    X, y = make_windows(df, seq, horizon, feature_cols)
    if len(y) < 100: raise ValueError('Too few samples after windowing; reduce --seq or use more data.')
    (Xtr, ytr), (Xva, yva) = split_train_val(X, y, 0.2)
    return DataLoader(SeqDS(Xtr, ytr), batch_size=batch, shuffle=True), DataLoader(SeqDS(Xva, yva), batch_size=batch, shuffle=False), len(feature_cols)
