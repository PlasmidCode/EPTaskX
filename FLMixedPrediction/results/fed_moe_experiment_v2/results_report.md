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
| type   |      mse |     rmse |     mae |       r2 |
|:-------|---------:|---------:|--------:|---------:|
| solar  |  23.4456 |  4.19147 | 2.2249  | 0.909925 |
| wind   | 162.688  | 10.4283  | 7.15006 | 0.789167 |

## Per-site Metrics
| site                 | type   |       mse |     rmse |       mae |       r2 |
|:---------------------|:-------|----------:|---------:|----------:|---------:|
| solar_station_site_1 | solar  |  12.7883  |  3.57608 |  1.68959  | 0.930145 |
| solar_station_site_2 | solar  |  70.7056  |  8.40866 |  4.75661  | 0.900419 |
| solar_station_site_3 | solar  |  12.1055  |  3.4793  |  2.16488  | 0.796707 |
| solar_station_site_4 | solar  |  49.8247  |  7.05866 |  3.63175  | 0.911214 |
| solar_station_site_5 | solar  |  32.5543  |  5.70564 |  2.8973   | 0.922539 |
| solar_station_site_6 | solar  |   3.13827 |  1.77152 |  0.866762 | 0.958858 |
| solar_station_site_7 | solar  |   4.37187 |  2.0909  |  1.05098  | 0.916906 |
| solar_station_site_8 | solar  |   2.07648 |  1.441   |  0.741285 | 0.942609 |
| wind_farm_site_1     | wind   | 103.693   | 10.183   |  6.19249  | 0.833169 |
| wind_farm_site_2     | wind   | 687.38    | 26.2179  | 18.0084   | 0.784742 |
| wind_farm_site_3     | wind   |  64.7991  |  8.04979 |  5.37741  | 0.833363 |
| wind_farm_site_4     | wind   |  30.8244  |  5.55197 |  3.70993  | 0.929101 |
| wind_farm_site_5     | wind   |  15.9692  |  3.99615 |  2.93229  | 0.859901 |
| wind_farm_site_6     | wind   |  73.463   |  8.57106 |  6.67978  | 0.494728 |