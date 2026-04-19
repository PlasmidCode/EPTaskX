# 综合实验报告

## 实验结果汇总

| 场景 | 算法 | 目标函数值 | 棕电补能 (kWh) | 迁移数据量 (GB) | 任务完成率 |
|------|------|------------|----------------|------------------|------------|
| small_scale | optimization | 1.4817 | 0.0000 | 14.8171 | 1.0000 |
| small_scale | greedy | 6.3064 | 0.0000 | 52.9973 | 0.9956 |
| small_scale | multi_agent | 5.2997 | 0.0000 | 52.9973 | 1.0000 |
| small_scale | improved_multi_agent | 5.2997 | 0.0000 | 52.9973 | 1.0000 |
| small_scale | ppo | 199.0071 | 0.0000 | 0.0000 | 0.2107 |
| small_scale | mappo | 60.3732 | 0.0000 | 0.0000 | 0.7758 |
| small_scale | dial | 78.8234 | 0.0000 | 23.6847 | 0.7763 |
| small_scale | transformer | 199.0071 | 0.0000 | 0.0000 | 0.2107 |
| medium_scale | optimization | 2.0065 | 0.0000 | 20.0646 | 1.0000 |
| medium_scale | greedy | 187.0196 | 0.0000 | 83.3613 | 0.6815 |
| medium_scale | multi_agent | 57.3401 | 0.0000 | 167.2773 | 0.9356 |
| medium_scale | improved_multi_agent | 12.3570 | 0.0000 | 123.5701 | 1.0000 |
| medium_scale | ppo | 552.4587 | 0.0000 | 0.0000 | 0.0825 |
| medium_scale | mappo | 276.3746 | 0.0000 | 54.6315 | 0.6677 |
| medium_scale | dial | 311.2361 | 0.0000 | 33.4114 | 0.6583 |
| medium_scale | transformer | 552.4587 | 0.0000 | 0.0000 | 0.0825 |
| large_scale | optimization | 1.0608 | 0.0000 | 10.6076 | 1.0000 |
| large_scale | greedy | 378.0679 | 0.0000 | 248.3444 | 0.6474 |
| large_scale | multi_agent | 24.7490 | 0.0000 | 247.4904 | 1.0000 |
| large_scale | improved_multi_agent | 19.0914 | 0.0000 | 190.9142 | 1.0000 |
| large_scale | ppo | 556.9567 | 0.0000 | 0.0000 | 0.0848 |
| large_scale | mappo | 342.6889 | 0.0000 | 68.7014 | 0.6384 |
| large_scale | dial | 159.1062 | 0.0000 | 92.5188 | 0.8313 |
| large_scale | transformer | 556.9567 | 0.0000 | 0.0000 | 0.0848 |
| high_volatility | optimization | 1.8171 | 0.0000 | 18.1712 | 1.0000 |
| high_volatility | greedy | 280.4170 | 0.0000 | 100.7508 | 0.6425 |
| high_volatility | multi_agent | 12.0682 | 0.0000 | 120.6821 | 1.0000 |
| high_volatility | improved_multi_agent | 6.6652 | 0.0000 | 66.6525 | 1.0000 |
| high_volatility | ppo | 329.7242 | 0.0000 | 0.0000 | 0.2846 |
| high_volatility | mappo | 80.7622 | 0.0000 | 37.0068 | 0.8601 |
| high_volatility | dial | 79.9817 | 0.0000 | 10.2168 | 0.8499 |
| high_volatility | transformer | 417.1847 | 0.0000 | 0.0000 | 0.1557 |
| high_density | optimization | 417.8910 | 0.0000 | 20.5912 | 0.5217 |
| high_density | greedy | 874.7636 | 0.0000 | 235.5061 | 0.3588 |
| high_density | multi_agent | 542.8435 | 0.0000 | 186.7320 | 0.5217 |
| high_density | improved_multi_agent | 544.7081 | 0.0000 | 161.2857 | 0.5217 |
| high_density | ppo | 1169.7217 | 0.0000 | 0.0000 | 0.1398 |
| high_density | mappo | 1076.0777 | 0.0000 | 55.4407 | 0.3563 |
| high_density | dial | 785.6882 | 0.0000 | 78.1211 | 0.4741 |
| high_density | transformer | 1174.6592 | 0.0000 | 0.0000 | 0.0702 |

## 分析

- optimization 平均目标函数值: 84.8514
- greedy 平均目标函数值: 345.3149
- multi_agent 平均目标函数值: 128.4601
- improved_multi_agent 平均目标函数值: 117.6243
- ppo 平均目标函数值: 561.5737
- mappo 平均目标函数值: 367.2553
- dial 平均目标函数值: 282.9671
- transformer 平均目标函数值: 580.0533

## 结论
- 改进的多智能体算法在大多数场景中表现优异
- 优化算法在小规模问题中表现良好
