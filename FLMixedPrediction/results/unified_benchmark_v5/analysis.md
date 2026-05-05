# Prediction Benchmark Analysis

## Adopted Improvements

- Added persistence and centralized-training baselines so FL results are compared against both a simple sanity check and a non-private upper-bound reference.
- Added validation-residual P10/P90 forecast intervals and interval metrics, including PICP, PINAW, interval width, and quantile losses.
- Exported the selected forecast as `forecast_for_scheduler.csv` using the `site_id,slot,r_mean_kwh,r_p10_kwh,r_p90_kwh` schema required by TaskScheduleSimu.
- Regenerated publication-style PNG and PDF figures with consistent colors, annotated heatmaps, uncertainty bands, and a compact paper summary figure.

## FL+MoE Topline

|   horizon |   rmse_mean |   mae_mean |   r2_mean |   wape_mean |   picp_80_mean |   pinaw_80_mean |
|----------:|------------:|-----------:|----------:|------------:|---------------:|----------------:|
|         1 |     6.55355 |    4.13703 |  0.861297 |     23.5427 |       0.686622 |        0.166519 |
|         6 |    11.6323  |    8.00334 |  0.620112 |     45.6418 |       0.68333  |        0.306399 |
|        24 |    14.4754  |   10.9284  |  0.389605 |     57.6236 |       0.64594  |        0.39027  |

## Average Rank Focus

|   horizon | metric   | method      |   avg_rank |
|----------:|:---------|:------------|-----------:|
|         1 | rmse     | Centralized |    1.14286 |
|         1 | rmse     | FL+MoE      |    3.42857 |
|         1 | rmse     | FedAvg      |    3.78571 |
|         1 | rmse     | FedPer      |    5.85714 |
|         1 | rmse     | Local       |    3.21429 |
|         1 | rmse     | Persistence |    3.57143 |
|         1 | wape     | Centralized |    1.42857 |
|         1 | wape     | FL+MoE      |    2.85714 |
|         1 | wape     | FedAvg      |    4       |
|         1 | wape     | FedPer      |    5.92857 |
|         1 | wape     | Local       |    3.5     |
|         1 | wape     | Persistence |    3.28571 |
|         6 | rmse     | Centralized |    1.85714 |
|         6 | rmse     | FL+MoE      |    1.78571 |
|         6 | rmse     | FedAvg      |    3.71429 |
|         6 | rmse     | FedPer      |    5.28571 |
|         6 | rmse     | Local       |    3.28571 |
|         6 | rmse     | Persistence |    5.07143 |
|         6 | wape     | Centralized |    2.71429 |
|         6 | wape     | FL+MoE      |    2       |
|         6 | wape     | FedAvg      |    4.14286 |
|         6 | wape     | FedPer      |    5.28571 |
|         6 | wape     | Local       |    2.85714 |
|         6 | wape     | Persistence |    4       |
|        24 | rmse     | Centralized |    2.14286 |
|        24 | rmse     | FL+MoE      |    2.07143 |
|        24 | rmse     | FedAvg      |    3.57143 |
|        24 | rmse     | FedPer      |    5.21429 |
|        24 | rmse     | Local       |    3.07143 |
|        24 | rmse     | Persistence |    4.92857 |
|        24 | wape     | Centralized |    2.42857 |
|        24 | wape     | FL+MoE      |    2.71429 |
|        24 | wape     | FedAvg      |    4.78571 |
|        24 | wape     | FedPer      |    5       |
|        24 | wape     | Local       |    3.78571 |
|        24 | wape     | Persistence |    2.28571 |

## Win Count Focus

|   horizon | metric   | method      |   win_count |
|----------:|:---------|:------------|------------:|
|         1 | rmse     | Local       |           0 |
|         1 | rmse     | FedAvg      |           0 |
|         1 | rmse     | FedPer      |           0 |
|         1 | rmse     | FL+MoE      |           0 |
|         1 | rmse     | Persistence |           2 |
|         1 | rmse     | Centralized |          12 |
|         1 | r2       | Local       |           0 |
|         1 | r2       | FedAvg      |           0 |
|         1 | r2       | FedPer      |           0 |
|         1 | r2       | FL+MoE      |           0 |
|         1 | r2       | Persistence |           2 |
|         1 | r2       | Centralized |          12 |
|         1 | wape     | Local       |           0 |
|         1 | wape     | FedAvg      |           0 |
|         1 | wape     | FedPer      |           0 |
|         1 | wape     | FL+MoE      |           0 |
|         1 | wape     | Persistence |           6 |
|         1 | wape     | Centralized |           8 |
|         1 | corr     | Local       |           0 |
|         1 | corr     | FedAvg      |           0 |
|         1 | corr     | FedPer      |           0 |
|         1 | corr     | FL+MoE      |           0 |
|         1 | corr     | Persistence |           1 |
|         1 | corr     | Centralized |          13 |
|         6 | rmse     | Local       |           1 |
|         6 | rmse     | FedAvg      |           1 |
|         6 | rmse     | FedPer      |           0 |
|         6 | rmse     | FL+MoE      |           5 |
|         6 | rmse     | Persistence |           0 |
|         6 | rmse     | Centralized |           7 |
|         6 | r2       | Local       |           1 |
|         6 | r2       | FedAvg      |           1 |
|         6 | r2       | FedPer      |           0 |
|         6 | r2       | FL+MoE      |           5 |
|         6 | r2       | Persistence |           0 |
|         6 | r2       | Centralized |           7 |
|         6 | wape     | Local       |           1 |
|         6 | wape     | FedAvg      |           1 |
|         6 | wape     | FedPer      |           0 |
|         6 | wape     | FL+MoE      |           4 |
|         6 | wape     | Persistence |           4 |
|         6 | wape     | Centralized |           4 |
|         6 | corr     | Local       |           1 |
|         6 | corr     | FedAvg      |           2 |
|         6 | corr     | FedPer      |           0 |
|         6 | corr     | FL+MoE      |           4 |
|         6 | corr     | Persistence |           1 |
|         6 | corr     | Centralized |           6 |
|        24 | rmse     | Local       |           2 |
|        24 | rmse     | FedAvg      |           0 |
|        24 | rmse     | FedPer      |           1 |
|        24 | rmse     | FL+MoE      |           5 |
|        24 | rmse     | Persistence |           0 |
|        24 | rmse     | Centralized |           6 |
|        24 | r2       | Local       |           2 |
|        24 | r2       | FedAvg      |           0 |
|        24 | r2       | FedPer      |           1 |
|        24 | r2       | FL+MoE      |           5 |
|        24 | r2       | Persistence |           0 |
|        24 | r2       | Centralized |           6 |
|        24 | wape     | Local       |           0 |
|        24 | wape     | FedAvg      |           0 |
|        24 | wape     | FedPer      |           2 |
|        24 | wape     | FL+MoE      |           1 |
|        24 | wape     | Persistence |           8 |
|        24 | wape     | Centralized |           3 |
|        24 | corr     | Local       |           1 |
|        24 | corr     | FedAvg      |           1 |
|        24 | corr     | FedPer      |           0 |
|        24 | corr     | FL+MoE      |           6 |
|        24 | corr     | Persistence |           2 |
|        24 | corr     | Centralized |           4 |

## Scheduler Forecast Export

- Rows exported: 32258
- The interval is calibrated from validation residuals. This is stronger than a fixed 0.85/1.15 fallback, but it should still be described as empirical calibration rather than a full probabilistic forecasting model.
## V4 Regression Check

The rerun uses the v4-stable FL+MoE route (`moe_variant=legacy`) while keeping the enhanced benchmark outputs. Common methods between v4 and v5 were compared on RMSE, MAE, SMAPE, WAPE, NRMSE, R2, and correlation across 1h/6h/24h horizons.

- Regression count: 0
- Maximum absolute delta on common summary metrics: 0.0
- FL+MoE matches v4 exactly on all checked core metrics.
- Detailed comparison: `v4_v5_regression_check.csv`
