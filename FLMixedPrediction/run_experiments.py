#!/usr/bin/env python3
"""
Run basic experiments for FLMixedPrediction project
"""

import os
import sys
import subprocess

# 创建outputs目录
os.makedirs('outputs', exist_ok=True)

# 实验配置
experiments = [
    {
        'name': 'tinylstm_1h',
        'command': ['python', '-m', 'src.run_demo', '--rounds', '5', '--seq', '24', '--horizon', '1'],
        'output_file': 'outputs/tinylstm_1h.txt'
    },
    {
        'name': 'tinylstm_6h',
        'command': ['python', '-m', 'src.run_demo', '--rounds', '5', '--seq', '24', '--horizon', '6'],
        'output_file': 'outputs/tinylstm_6h.txt'
    },
    {
        'name': 'tinylstm_24h',
        'command': ['python', '-m', 'src.run_demo', '--rounds', '5', '--seq', '24', '--horizon', '24'],
        'output_file': 'outputs/tinylstm_24h.txt'
    },
    {
        'name': 'moe_1h',
        'command': ['python', '-m', 'src.run_demo', '--rounds', '5', '--seq', '24', '--horizon', '1', '--model', 'moe', '--num_experts', '3'],
        'output_file': 'outputs/moe_1h.txt'
    },
    {
        'name': 'moe_6h',
        'command': ['python', '-m', 'src.run_demo', '--rounds', '5', '--seq', '24', '--horizon', '6', '--model', 'moe', '--num_experts', '3'],
        'output_file': 'outputs/moe_6h.txt'
    },
    {
        'name': 'moe_24h',
        'command': ['python', '-m', 'src.run_demo', '--rounds', '5', '--seq', '24', '--horizon', '24', '--model', 'moe', '--num_experts', '3'],
        'output_file': 'outputs/moe_24h.txt'
    }
]

for exp in experiments:
    print(f"Running experiment: {exp['name']}")
    print(f"Command: {' '.join(exp['command'])}")
    
    # 运行实验并保存结果
    with open(exp['output_file'], 'w') as f:
        result = subprocess.run(exp['command'], capture_output=True, text=True, cwd='/home/plasmid/Project/FLMixedPrediction'
        f.write(f"STDOUT:\n{result.stdout}\n")
        f.write(f"\nSTDERR:\n{result.stderr}\n")
        f.write(f"\nReturn Code: {result.returncode}\n")
    
    print(f"Experiment {exp['name']} completed. Results saved to {exp['output_file']}")
    print("=" * 50)

print("All experiments completed!")