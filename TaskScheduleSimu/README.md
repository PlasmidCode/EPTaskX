# 多域清洁能源预测驱动的跨域任务迁移与调度

`TaskScheduleSimu` 用于验证“站点清洁能源预测结果驱动计算任务迁移与调度”的实验流程。当前版本支持合成任务负载和 Alibaba Cluster Trace 真实负载，并提供鲁棒调度、启发式基线、统一指标报告和可复现 CSV 输出。

## 核心能力

- 读取预测 CSV：`site_id, slot, r_mean_kwh, r_p10_kwh, r_p90_kwh`
- 使用 `r_p10_kwh` 作为鲁棒清洁能源下界，缺失时回退到 `0.85 * r_mean_kwh`
- 支持 Alibaba v2023 GPU pod trace 和 v2018 batch trace
- 支持 `peak/random/earliest/latest` 多时间窗 trace 采样，适合做多场景统计对比
- 建模储能 SOC、充放电功率、弃电、棕电补能、链路级迁移带宽和迁移冷却
- 输出 Markdown 报告、图表、场景级 `metrics.csv` 和综合 `comprehensive_metrics.csv`

## 安装依赖

```bash
cd TaskScheduleSimu
pip install -r requirements.txt
```

默认鲁棒调度依赖 `numpy`、`scipy`、`pandas`、`matplotlib`。`cvxpy` 是可选依赖，未安装时 `optimization` 基线会自动回退。

## 调度算法

默认先进算法：

- `robust_mpc_alns`：鲁棒 MPC 线性松弛 + ALNS 修复，使用 P10 清洁能源预测下界、储能和链路带宽约束。

对比基线：

- `min_grid`：棕电最小化启发式，兼顾 P10 风险、迁移和截止期。
- `edf`：Earliest Deadline First，按截止期优先调度。
- `source_affinity`：源站点优先/不迁移基线，用于衡量迁移收益。
- `least_loaded`：最小容量占用率优先，用于衡量负载均衡策略。
- `greedy`：清洁能源贪心基线，保留旧实验接口。
- `random`：随机可行基线，用于 sanity check。
- `optimization`：可选 `cvxpy` 连续优化基线。

## 快速运行

合成负载小实验：

```bash
python src/main.py --domains 2 --slots 4 --tasks 3 --alns-iterations 100
```

多算法对比：

```bash
python src/main.py --domains 2 --slots 4 --tasks 3 --algorithms robust_mpc_alns,min_grid,edf,source_affinity,least_loaded,greedy
```

运行合成多场景：

```bash
python src/run_experiments.py --algorithms robust_mpc_alns,min_grid,edf,source_affinity,least_loaded,greedy
```

导出或使用预测 CSV：

```bash
python src/main.py --export-forecast-csv results/demo_forecast.csv
python src/main.py --forecast-csv results/demo_forecast.csv --tasks 5
```

## Alibaba Trace 实验

仓库内提供极小 smoke-test 样例，用于确认环境和入口可运行：

```bash
python src/main.py --workload alibaba-v2023 --pod-csv examples/alibaba_v2023_pods_sample.csv --node-csv examples/alibaba_v2023_nodes_sample.csv --domains 2 --slots 4 --trace-task-limit 6 --alns-iterations 20
python src/main.py --workload alibaba-v2018 --batch-task-csv examples/alibaba_v2018_batch_task_sample.csv --machine-meta-csv examples/alibaba_v2018_machine_meta_sample.csv --domains 2 --slots 4 --trace-task-limit 6 --alns-iterations 20
```

更有说服力的实验建议使用 Alibaba 官方完整 CSV，并从真实时间线上采样多个峰值窗口：

```bash
python src/run_experiments.py ^
  --workload alibaba-v2023 ^
  --pod-csv path\to\openb_pod_list_default.csv ^
  --node-csv path\to\openb_node_list_default.csv ^
  --domains 4 ^
  --slots 24 ^
  --trace-task-limit 300 ^
  --trace-sample-policy peak ^
  --trace-windows 5 ^
  --algorithms robust_mpc_alns,min_grid,edf,source_affinity,least_loaded,greedy ^
  --alns-iterations 300
```

v2018 batch trace：

```bash
python src/run_experiments.py ^
  --workload alibaba-v2018 ^
  --batch-task-csv path\to\batch_task.csv ^
  --machine-meta-csv path\to\machine_meta.csv ^
  --domains 4 ^
  --slots 24 ^
  --trace-task-limit 300 ^
  --trace-sample-policy peak ^
  --trace-windows 5
```

如果已有预测模块输出的站点清洁能源预测，可直接和 trace 负载组合：

```bash
python src/run_experiments.py ^
  --workload alibaba-v2023 ^
  --pod-csv path\to\openb_pod_list_default.csv ^
  --forecast-csv path\to\forecast.csv ^
  --trace-sample-policy peak ^
  --trace-windows 5
```

## 指标

报告会输出以下核心指标：

- Common objective：统一目标函数，便于跨算法比较。
- Expected grid / P10 grid：均值预测和 P10 鲁棒场景下的棕电消耗。
- CO2e：按默认电网排放因子估算的碳排。
- Migration / migration events：迁移数据量和迁移次数。
- Completion / deadline miss rate：任务完成率和截止期未完成比例。
- Clean utilization / CFE：清洁能源消纳率和清洁能源供能占比。
- Curtailment：弃电量。
- Peak capacity / peak link utilization：峰值容量和链路利用率。
- Forecast risk exposure：预测区间宽度加权后的风险暴露。

## 代码结构

```text
src/
  algorithms/
    robust_mpc_alns_scheduler.py   # 鲁棒 MPC + ALNS
    heuristic_schedulers.py        # EDF、min-grid、source-affinity 等基线
    greedy_scheduler.py            # 兼容旧接口的清洁能源贪心基线
    optimization_scheduler.py      # cvxpy 可选优化基线
  model/
    problem.py                     # 问题模型
  utils/
    alibaba_trace_loader.py        # Alibaba trace 适配
    forecast_loader.py             # 预测 CSV 读写
    schedule_metrics.py            # 统一指标计算
    test_data_generator.py         # 合成场景
  visualization/
    visualizer.py                  # 图表和 Markdown 报告
```

实验结果默认写入 `results/`，该目录已加入 `.gitignore`。
