# FL+MoE Cross-domain Renewable Forecasting Results

## Settings
- seq_len: 24
- horizon: 1
- rounds: 4
- local_epochs: 3
- num_experts: 4
- hidden_dim: 64
- batch_size: 128
- lr: 0.001
- client_fraction: 1.0

## Type-level Mean Metrics
| type   |      mse |    rmse |     mae |       r2 |
|:-------|---------:|--------:|--------:|---------:|
| solar  |  17.0026 | 3.55742 | 1.75898 | 0.937779 |
| wind   | 129.569  | 9.43419 | 6.49952 | 0.823968 |

## Per-site Metrics
| site                 | type   |       mse |     rmse |       mae |       r2 |
|:---------------------|:-------|----------:|---------:|----------:|---------:|
| solar_station_site_1 | solar  |   9.83318 |  3.13579 |  1.41995  | 0.946287 |
| solar_station_site_2 | solar  |  50.5567  |  7.11033 |  3.64716  | 0.928797 |
| solar_station_site_3 | solar  |   6.47814 |  2.54522 |  1.58142  | 0.89121  |
| solar_station_site_4 | solar  |  39.1199  |  6.25459 |  2.97783  | 0.930289 |
| solar_station_site_5 | solar  |  22.2017  |  4.71187 |  2.1173   | 0.947173 |
| solar_station_site_6 | solar  |   2.82401 |  1.68048 |  0.782833 | 0.962978 |
| solar_station_site_7 | solar  |   3.92671 |  1.98159 |  1.01604  | 0.925367 |
| solar_station_site_8 | solar  |   1.08055 |  1.03949 |  0.529261 | 0.970135 |
| wind_farm_site_1     | wind   |  99.4459  |  9.97226 |  6.29167  | 0.840003 |
| wind_farm_site_2     | wind   | 520.313   | 22.8104  | 15.516    | 0.83706  |
| wind_farm_site_3     | wind   |  63.214   |  7.95073 |  5.64586  | 0.837439 |
| wind_farm_site_4     | wind   |  21.9254  |  4.68246 |  3.03353  | 0.94957  |
| wind_farm_site_5     | wind   |  11.3429  |  3.36793 |  2.19764  | 0.900487 |
| wind_farm_site_6     | wind   |  61.1741  |  7.82139 |  6.31239  | 0.57925  |