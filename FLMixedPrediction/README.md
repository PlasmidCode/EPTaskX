# FLMixedPrediction: Two-Site Federated Forecasting (PV + Wind)

## Overview
This project implements federated learning for forecasting solar PV and wind power generation using two different models:
1. `TinyLSTM`: A simple LSTM-based model
2. `MOE`: A Mixture of Experts model with multiple LSTM experts and a gating network

## Installation and Setup

```bash
conda create -n fl_pred python=3.10 -y
conda activate fl_pred
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Data Generation

Generate synthetic PV and wind data:

```bash
python scripts/make_data.py
```

## Running the Federated Learning Simulation

### Using TinyLSTM model (default)

```bash
python -m src.run_demo --rounds 3 --seq 24 --horizon 1
```

### Using MOE (Mixture of Experts) model

```bash
python -m src.run_demo --rounds 3 --seq 24 --horizon 1 --model moe --num_experts 3
```

## Command Line Arguments

- `--rounds`: Number of federated learning rounds (default: 3)
- `--seq`: Sequence length for input windows (default: 24)
- `--horizon`: Prediction horizon (default: 1)
- `--local_epochs`: Number of local epochs per round (default: 1)
- `--batch`: Batch size (default: 64)
- `--model`: Model type to use (`tiny_lstm` or `moe`, default: `tiny_lstm`)
- `--num_experts`: Number of experts for MOE model (default: 3)

## Project Structure

```
FLMixedPrediction/
├── README.md              # Project documentation
├── data/                 # Generated data files
│   ├── pv_site.csv       # Solar PV data
│   └── wind_site.csv     # Wind power data
├── requirements.txt      # Python dependencies
├── scripts/              # Utility scripts
│   └── make_data.py      # Data generation script
└── src/                  # Source code
    ├── __pycache__/      # Compiled Python files
    ├── model.py          # Model definitions (TinyLSTM and MOE)
    ├── run_demo.py       # Federated learning simulation
    └── utils.py          # Utility functions
```

## Understanding the Output

When running the federated learning simulation, you'll see output like this:

```
History (loss, distributed, evaluate):
         [(1, 0.0703411385343879),
          (2, 0.06812340837751217),
          (3, 0.06592635041173739)]
History (metrics, distributed, evaluate):
         {'rmse': [(1, 0.2554991804322815),
                  (2, 0.27754303186545276),
                  (3, 0.2761212573169815)]}
```

- `loss`: Mean squared error (MSE) across all clients for each round
- `rmse`: Root mean squared error (RMSE) across all clients for each round

## Troubleshooting

### CUDA GPU Not Available
If you see an error like `RuntimeError: No CUDA GPUs are available`, don't worry. The code is designed to automatically fall back to CPU if no GPU is available.

### Insufficient Data
If you see an error like `ValueError: Too few samples after windowing`, try:
- Reducing the sequence length (`--seq` parameter)
- Increasing the amount of data by modifying `scripts/make_data.py`

### Dependency Issues
If you encounter dependency conflicts, try:
- Using a fresh conda environment
- Upgrading pip before installing dependencies
- Installing specific versions of conflicting packages
