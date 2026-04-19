# FL+MoE Cross-domain Renewable Forecasting Results

## Settings
- seq_len: 24
- horizon: 1
- rounds: 3
- local_epochs: 2
- num_experts: 4
- hidden_dim: 64
- batch_size: 64
- lr: 0.001
- client_fraction: 1.0

## Type-level Mean Metrics
| type   |      mse |    rmse |     mae |       r2 |
|:-------|---------:|--------:|--------:|---------:|
| solar  |  33.2029 | 4.80704 | 2.96136 | 0.897182 |
| wind   | 148.92   | 9.96102 | 6.74834 | 0.822601 |

## Per-site Metrics
| site                 | type   |       mse |     rmse |      mae |       r2 |
|:---------------------|:-------|----------:|---------:|---------:|---------:|
| solar_station_site_1 | solar  |  13.2774  |  3.64382 |  2.12367 | 0.927659 |
| solar_station_site_2 | solar  |  93.6425  |  9.6769  |  6.15669 | 0.868806 |
| solar_station_site_3 | solar  |   5.06653 |  2.2509  |  1.3383  | 0.914916 |
| solar_station_site_4 | solar  |  77.064   |  8.77861 |  5.40557 | 0.863153 |
| solar_station_site_5 | solar  |  64.1802  |  8.01126 |  4.99471 | 0.847572 |
| solar_station_site_6 | solar  |   4.1063  |  2.0264  |  1.14246 | 0.94616  |
| solar_station_site_7 | solar  |   4.42915 |  2.10456 |  1.18866 | 0.91579  |
| solar_station_site_8 | solar  |   3.85669 |  1.96385 |  1.34079 | 0.893403 |
| wind_farm_site_1     | wind   | 130.455   | 11.4217  |  7.31977 | 0.796584 |
| wind_farm_site_2     | wind   | 610.037   | 24.6989  | 16.6433  | 0.80896  |
| wind_farm_site_3     | wind   |  61.8428  |  7.86402 |  5.39848 | 0.841832 |
| wind_farm_site_4     | wind   |  30.251   |  5.50009 |  3.80531 | 0.930565 |
| wind_farm_site_5     | wind   |   9.79918 |  3.13036 |  2.04928 | 0.91384  |
| wind_farm_site_6     | wind   |  51.1367  |  7.15099 |  5.27385 | 0.643823 |