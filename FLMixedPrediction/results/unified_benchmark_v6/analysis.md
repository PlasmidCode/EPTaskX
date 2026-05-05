# Prediction Benchmark Analysis

## Adopted Improvements

- Added persistence and centralized-training baselines so FL results are compared against both a simple sanity check and a non-private upper-bound reference.
- Added validation-residual P10/P90 forecast intervals and interval metrics, including PICP, PINAW, interval width, and quantile losses.
- Exported the selected forecast as `forecast_for_scheduler.csv` using the `site_id,slot,r_mean_kwh,r_p10_kwh,r_p90_kwh` schema required by TaskScheduleSimu.
- Regenerated publication-style PNG and PDF figures with consistent colors, annotated heatmaps, uncertainty bands, and a compact paper summary figure.

## FL+MoE Topline

|   horizon |   rmse_mean |   mae_mean |   r2_mean |   wape_mean |   picp_80_mean |   pinaw_80_mean |
|----------:|------------:|-----------:|----------:|------------:|---------------:|----------------:|
|         1 |     5.26252 |    2.98798 |  0.922885 |     19.173  |       0.705921 |        0.134387 |
|         6 |    11.4964  |    7.93834 |  0.63265  |     44.5026 |       0.67972  |        0.298869 |
|        24 |    14.2318  |   10.7723  |  0.413829 |     56.7655 |       0.646952 |        0.392642 |

## Average Rank Focus

|   horizon | metric   | method      |   avg_rank |
|----------:|:---------|:------------|-----------:|
|         1 | rmse     | Centralized |    1       |
|         1 | rmse     | FL+MoE      |    3.28571 |
|         1 | rmse     | FedAvg      |    4.14286 |
|         1 | rmse     | FedPer      |    4.17857 |
|         1 | rmse     | Local       |    3.14286 |
|         1 | rmse     | Persistence |    5.25    |
|         1 | wape     | Centralized |    2.28571 |
|         1 | wape     | FL+MoE      |    2.64286 |
|         1 | wape     | FedAvg      |    3.92857 |
|         1 | wape     | FedPer      |    4.10714 |
|         1 | wape     | Local       |    4.14286 |
|         1 | wape     | Persistence |    3.89286 |
|         6 | rmse     | Centralized |    1.28571 |
|         6 | rmse     | FL+MoE      |    2.07143 |
|         6 | rmse     | FedAvg      |    3.92857 |
|         6 | rmse     | FedPer      |    4.14286 |
|         6 | rmse     | Local       |    3.71429 |
|         6 | rmse     | Persistence |    5.85714 |
|         6 | wape     | Centralized |    1.92857 |
|         6 | wape     | FL+MoE      |    2.21429 |
|         6 | wape     | FedAvg      |    4.42857 |
|         6 | wape     | FedPer      |    4.21429 |
|         6 | wape     | Local       |    3.78571 |
|         6 | wape     | Persistence |    4.42857 |
|        24 | rmse     | Centralized |    2.28571 |
|        24 | rmse     | FL+MoE      |    2.5     |
|        24 | rmse     | FedAvg      |    4       |
|        24 | rmse     | FedPer      |    3.42857 |
|        24 | rmse     | Local       |    3.21429 |
|        24 | rmse     | Persistence |    5.57143 |
|        24 | wape     | Centralized |    3.42857 |
|        24 | wape     | FL+MoE      |    2.21429 |
|        24 | wape     | FedAvg      |    4.5     |
|        24 | wape     | FedPer      |    4.21429 |
|        24 | wape     | Local       |    4.07143 |
|        24 | wape     | Persistence |    2.57143 |

## Win Count Focus

|   horizon | metric   | method      |   win_count |
|----------:|:---------|:------------|------------:|
|         1 | rmse     | Local       |           0 |
|         1 | rmse     | FedAvg      |           0 |
|         1 | rmse     | FedPer      |           0 |
|         1 | rmse     | FL+MoE      |           0 |
|         1 | rmse     | Persistence |           0 |
|         1 | rmse     | Centralized |          14 |
|         1 | r2       | Local       |           0 |
|         1 | r2       | FedAvg      |           0 |
|         1 | r2       | FedPer      |           0 |
|         1 | r2       | FL+MoE      |           0 |
|         1 | r2       | Persistence |           0 |
|         1 | r2       | Centralized |          14 |
|         1 | wape     | Local       |           0 |
|         1 | wape     | FedAvg      |           0 |
|         1 | wape     | FedPer      |           1 |
|         1 | wape     | FL+MoE      |           2 |
|         1 | wape     | Persistence |           5 |
|         1 | wape     | Centralized |           6 |
|         1 | corr     | Local       |           0 |
|         1 | corr     | FedAvg      |           0 |
|         1 | corr     | FedPer      |           0 |
|         1 | corr     | FL+MoE      |           0 |
|         1 | corr     | Persistence |           0 |
|         1 | corr     | Centralized |          14 |
|         6 | rmse     | Local       |           0 |
|         6 | rmse     | FedAvg      |           0 |
|         6 | rmse     | FedPer      |           0 |
|         6 | rmse     | FL+MoE      |           3 |
|         6 | rmse     | Persistence |           0 |
|         6 | rmse     | Centralized |          11 |
|         6 | r2       | Local       |           0 |
|         6 | r2       | FedAvg      |           0 |
|         6 | r2       | FedPer      |           0 |
|         6 | r2       | FL+MoE      |           3 |
|         6 | r2       | Persistence |           0 |
|         6 | r2       | Centralized |          11 |
|         6 | wape     | Local       |           1 |
|         6 | wape     | FedAvg      |           0 |
|         6 | wape     | FedPer      |           0 |
|         6 | wape     | FL+MoE      |           2 |
|         6 | wape     | Persistence |           4 |
|         6 | wape     | Centralized |           7 |
|         6 | corr     | Local       |           0 |
|         6 | corr     | FedAvg      |           0 |
|         6 | corr     | FedPer      |           0 |
|         6 | corr     | FL+MoE      |           2 |
|         6 | corr     | Persistence |           0 |
|         6 | corr     | Centralized |          12 |
|        24 | rmse     | Local       |           1 |
|        24 | rmse     | FedAvg      |           1 |
|        24 | rmse     | FedPer      |           2 |
|        24 | rmse     | FL+MoE      |           6 |
|        24 | rmse     | Persistence |           0 |
|        24 | rmse     | Centralized |           4 |
|        24 | r2       | Local       |           1 |
|        24 | r2       | FedAvg      |           1 |
|        24 | r2       | FedPer      |           2 |
|        24 | r2       | FL+MoE      |           6 |
|        24 | r2       | Persistence |           0 |
|        24 | r2       | Centralized |           4 |
|        24 | wape     | Local       |           0 |
|        24 | wape     | FedAvg      |           1 |
|        24 | wape     | FedPer      |           1 |
|        24 | wape     | FL+MoE      |           4 |
|        24 | wape     | Persistence |           7 |
|        24 | wape     | Centralized |           1 |
|        24 | corr     | Local       |           4 |
|        24 | corr     | FedAvg      |           2 |
|        24 | corr     | FedPer      |           2 |
|        24 | corr     | FL+MoE      |           3 |
|        24 | corr     | Persistence |           1 |
|        24 | corr     | Centralized |           2 |

## Scheduler Forecast Export

- Rows exported: 32258
- The interval is calibrated from validation residuals. This is stronger than a fixed 0.85/1.15 fallback, but it should still be described as empirical calibration rather than a full probabilistic forecasting model.

## V5-to-V6 Optimization Check

V6 uses CUDA/AMP training, validation-driven automatic postprocessing, the stable v5 feature preset, and per-run seed reset so method results are independent of execution order. The exploratory enhanced wind/ramp feature preset was kept available but was not used for the final v6 because it overfit the 24h wind horizon.

### FL+MoE Wind

|   horizon |   rmse_v5 |   rmse_v6 |   rmse_improve_pct |   mae_v5 |   mae_v6 |   mae_improve_pct |   wape_v5 |   wape_v6 |   wape_improve_pct |      r2_v5 |      r2_v6 |   r2_delta |
|----------:|----------:|----------:|-------------------:|---------:|---------:|------------------:|----------:|----------:|-------------------:|-----------:|-----------:|-----------:|
|         1 |   10.1407 |   7.27109 |           28.2979  |  7.04079 |  4.41385 |         37.3103   |   25.6883 |   16.4211 |          36.0758   |  0.774754  |  0.910974  |  0.13622   |
|         6 |   17.8698 |  17.6619  |            1.1634  | 13.8235  | 13.706   |          0.849983 |   51.8895 |   49.8474 |           3.93536  |  0.431466  |  0.453344  |  0.021878  |
|        24 |   24.2543 |  23.9379  |            1.30433 | 20.552   | 20.3255  |          1.10215  |   79.456  |   78.9544 |           0.631388 | -0.0888223 | -0.0562376 |  0.0325847 |

### FL+MoE Solar

|   horizon |   rmse_v5 |   rmse_v6 |   rmse_improve_pct |   mae_v5 |   mae_v6 |   mae_improve_pct |   wape_v5 |   wape_v6 |   wape_improve_pct |
|----------:|----------:|----------:|-------------------:|---------:|---------:|------------------:|----------:|----------:|-------------------:|
|         1 |   3.8632  |   3.75609 |            2.77254 |  1.95921 |  1.91857 |          2.07405  |   21.9334 |   21.2369 |            3.17557 |
|         6 |   6.95413 |   6.87219 |            1.17827 |  3.63822 |  3.61258 |          0.704764 |   40.9561 |   40.4939 |            1.1284  |
|        24 |   7.14128 |   6.95229 |            2.64652 |  3.7107  |  3.60749 |          2.78144  |   41.2493 |   40.1239 |            2.72836 |

Additional comparison files: `v5_v6_type_comparison.csv`, `v5_v6_site_comparison.csv`, `wind_v5_v6_comparison.csv`, and `solar_v5_v6_comparison.csv`.
