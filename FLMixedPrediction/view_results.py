import os
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
import json
from pathlib import Path

# 获取最新的日志目录
def get_latest_log_dir(logs_dir):
    log_dirs = [d for d in os.listdir(logs_dir) if os.path.isdir(os.path.join(logs_dir, d))]
    log_dirs.sort(reverse=True)
    return os.path.join(logs_dir, log_dirs[0]) if log_dirs else None

# 读取TensorBoard事件文件
def read_tensorboard_events(log_dir):
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    return ea

# 生成HTML页面
def generate_html_report(data, output_file):
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Federated Learning Results</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            .section {{ margin-bottom: 30px; }}
            .metric {{ margin-bottom: 20px; }}
            .chart-container {{ width: 800px; height: 400px; margin: 20px 0; }}
            canvas {{ max-width: 100%; }}
        </style>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    </head>
    <body>
        <h1>Federated Learning Simulation Results</h1>
    """
    
    # 为每个指标创建图表
    for metric_name, metric_data in data.items():
        html += f"""
        <div class="section">
            <h2>{metric_name}</h2>
            <div class="chart-container">
                <canvas id="chart-{metric_name}"></canvas>
            </div>
        </div>
        """
    
    # JavaScript代码来绘制图表
    html += "<script>"
    for metric_name, metric_data in data.items():
        labels = json.dumps(list(metric_data.keys()))
        values = json.dumps(list(metric_data.values()))
        html += f"""
        var ctx = document.getElementById('chart-{metric_name}').getContext('2d');
        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: {labels},
                datasets: [{{
                    label: '{metric_name}',
                    data: {values},
                    borderColor: 'blue',
                    backgroundColor: 'rgba(0, 0, 255, 0.1)',
                    tension: 0.3,
                    fill: true
                }}]
            }},
            options: {{
                responsive: true,
                scales: {{
                    x: {{ title: {{ display: true, text: 'Round Number' }} }},
                    y: {{ title: {{ display: true, text: 'Metric Value' }} }}
                }}
            }}
        }});
        """
    
    html += "</script></body></html>"
    
    with open(output_file, 'w') as f:
        f.write(html)
    
    print(f"Report generated: {output_file}")

# 主函数
def main():
    logs_dir = "logs"
    latest_log_dir = get_latest_log_dir(logs_dir)
    
    if not latest_log_dir:
        print("No log directories found.")
        return
    
    print(f"Reading logs from: {latest_log_dir}")
    
    # 读取事件文件
    ea = read_tensorboard_events(latest_log_dir)
    
    # 获取所有标签
    tags = ea.Tags()['scalars']
    print(f"Available metrics: {tags}")
    
    # 提取数据
    data = {}
    for tag in tags:
        events = ea.Scalars(tag)
        rounds = [event.step for event in events]
        values = [event.value for event in events]
        data[tag] = dict(zip(rounds, values))
    
    # 生成HTML报告
    generate_html_report(data, "results_report.html")

if __name__ == "__main__":
    main()
