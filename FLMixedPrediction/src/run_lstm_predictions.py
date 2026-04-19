import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.model import TinyLSTM
from src.renewable_data_integrator import get_all_sites_dataframes
from src.utils import build_loaders

# 创建必要的目录
figure_dir = Path('./figure')
figure_dir.mkdir(exist_ok=True)

models_dir = Path('./models')
models_dir.mkdir(exist_ok=True)

logs_dir = Path('./logs')
logs_dir.mkdir(exist_ok=True)


def train_model(model, tr_loader, va_loader, device, epochs, lr, log_interval, writer, model_name):
    """
    训练模型
    
    Args:
        model: 模型实例
        tr_loader: 训练数据加载器
        va_loader: 验证数据加载器
        device: 设备
        epochs: 训练轮数
        lr: 学习率
        log_interval: 日志间隔
        writer: TensorBoard写入器
        model_name: 模型名称
        
    Returns:
        训练损失和验证损失列表
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        
        # 计算平均训练损失
        train_loss /= len(tr_loader.dataset)
        train_losses.append(train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_ys = []
        
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)
                loss = criterion(outputs, yb)
                val_loss += loss.item() * xb.size(0)
                all_preds.extend(outputs.cpu().numpy())
                all_ys.extend(yb.cpu().numpy())
        
        # 计算平均验证损失
        val_loss /= len(va_loader.dataset)
        val_losses.append(val_loss)
        
        # 计算验证集指标
        mse = mean_squared_error(all_ys, all_preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(all_ys, all_preds)
        r2 = r2_score(all_ys, all_preds)
        
        # 记录到TensorBoard
        writer.add_scalar(f'{model_name}/Loss/Train', train_loss, epoch)
        writer.add_scalar(f'{model_name}/Loss/Val', val_loss, epoch)
        writer.add_scalar(f'{model_name}/Metrics/MSE', mse, epoch)
        writer.add_scalar(f'{model_name}/Metrics/RMSE', rmse, epoch)
        writer.add_scalar(f'{model_name}/Metrics/MAE', mae, epoch)
        writer.add_scalar(f'{model_name}/Metrics/R2', r2, epoch)
        
        # 打印日志
        if (epoch + 1) % log_interval == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val RMSE: {rmse:.4f}')
    
    return train_losses, val_losses, all_ys, all_preds


def plot_loss(train_losses, val_losses, site_type, site_name, writer, epoch):
    """
    绘制损失曲线
    
    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        site_type: 站点类型
        site_name: 站点名称
        writer: TensorBoard写入器
        epoch: 当前轮数
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'{site_type.upper()} - {site_name} - Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    
    # 保存图片
    filename = f'{site_type}_{site_name}_loss.png'
    plt.savefig(figure_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'Loss plot saved: {filename}')


def plot_predictions(y_true, y_pred, site_type, site_name, writer, epoch):
    """
    绘制预测结果
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        site_type: 站点类型
        site_name: 站点名称
        writer: TensorBoard写入器
        epoch: 当前轮数
    """
    # 绘制前500个样本的预测结果
    plt.figure(figsize=(15, 8))
    plt.plot(y_true[:500], label='True Values', alpha=0.7)
    plt.plot(y_pred[:500], label='Predictions', alpha=0.7)
    plt.title(f'{site_type.upper()} - {site_name} - True vs Predicted Values (First 500 Samples)')
    plt.xlabel('Sample Index')
    plt.ylabel('Power (MW)')
    plt.legend()
    plt.grid(True)
    
    # 保存图片
    filename = f'{site_type}_{site_name}_predictions.png'
    plt.savefig(figure_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'Predictions plot saved: {filename}')
    
    # 绘制散点图
    plt.figure(figsize=(10, 8))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label='Perfect Prediction')
    plt.title(f'{site_type.upper()} - {site_name} - True vs Predicted Values')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.grid(True)
    
    # 保存图片
    filename = f'{site_type}_{site_name}_scatter.png'
    plt.savefig(figure_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'Scatter plot saved: {filename}')


def calculate_metrics(y_true, y_pred):
    """
    计算评估指标
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        评估指标字典
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }


def print_metrics(metrics, site_type, site_name):
    """
    打印评估指标
    
    Args:
        metrics: 评估指标字典
        site_type: 站点类型
        site_name: 站点名称
    """
    print(f'\n--- {site_type.upper()} - {site_name} Evaluation Metrics ---')
    for metric_name, value in metrics.items():
        print(f'{metric_name}: {value:.4f}')


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq', type=int, default=24, help='序列长度')
    parser.add_argument('--horizon', type=int, default=1, help='预测步长')
    parser.add_argument('--batch', type=int, default=32, help='批次大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--log_interval', type=int, default=5, help='日志间隔')
    parser.add_argument('--hidden_size', type=int, default=64, help='LSTM隐藏层大小')
    parser.add_argument('--num_layers', type=int, default=2, help='LSTM层数')
    parser.add_argument('--use_gpu', action='store_true', help='使用GPU')
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 初始化TensorBoard写入器
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    tb_log_dir = logs_dir / f'rnn_predictions_{current_time}'
    writer = SummaryWriter(str(tb_log_dir))
    
    print(f'TensorBoard logs will be saved to: {tb_log_dir}')
    
    # 加载数据
    data_dir = Path('./data/Renewable-energy-generation-input-feature-variables-analysis-main/data_processed')
    print(f'Loading data from: {data_dir}')
    
    # 获取所有站点的数据帧
    all_dfs = get_all_sites_dataframes(str(data_dir), 'all')
    print(f'Loaded {len(all_dfs)} sites data')
    
    # 分离风场和光伏电站数据
    wind_dfs = []
    solar_dfs = []
    
    for i, df in enumerate(all_dfs):
        # 简单判断：风场数据通常有风速、风向等特征
        # 光伏数据通常有辐照度等特征
        if any(col.lower().startswith('wind') for col in df.columns):
            wind_dfs.append(df)
        else:
            solar_dfs.append(df)
    
    print(f'Found {len(wind_dfs)} wind farms and {len(solar_dfs)} solar stations')
    
    # 选择一个风场和一个光伏电站
    selected_sites = []
    
    if wind_dfs:
        selected_sites.append(('wind', 'wind_site_0', wind_dfs[0]))
    
    if solar_dfs:
        selected_sites.append(('solar', 'solar_site_0', solar_dfs[0]))
    
    # 训练每个站点的模型
    for site_type, site_name, df in selected_sites:
        print(f'\n\n===== Training {site_type} site: {site_name} =====')
        
        # 构建数据加载器
        tr_loader, va_loader, in_dim = build_loaders(df, args.seq, args.horizon, args.batch)
        
        
        print(f'Input dimension: {in_dim}')
        
        # 创建模型
        model = TinyLSTM(in_dim, args.hidden_size, args.num_layers).to(device)
        print(f'Model architecture: {model}')
        
        # 训练模型
        train_losses, val_losses, y_true, y_pred = train_model(
            model, tr_loader, va_loader, device, args.epochs, args.lr, 
            args.log_interval, writer, f'{site_type}_{site_name}'
        )
        
        # 保存模型
        model_path = models_dir / f'{site_type}_{site_name}_model.pth'
        torch.save(model.state_dict(), model_path)
        print(f'Model saved: {model_path}')
        
        # 绘制损失曲线
        plot_loss(train_losses, val_losses, site_type, site_name, writer, args.epochs)
        
        # 绘制预测结果
        plot_predictions(y_true, y_pred, site_type, site_name, writer, args.epochs)
        
        # 计算并打印评估指标
        metrics = calculate_metrics(y_true, y_pred)
        print_metrics(metrics, site_type, site_name)
        
        # 记录指标到TensorBoard
        for metric_name, value in metrics.items():
            writer.add_scalar(f'{site_type}_{site_name}/Final/{metric_name}', value, 0)
    
    # 关闭TensorBoard写入器
    writer.close()
    
    print(f'\n\nAll training completed!')
    print(f'TensorBoard logs saved to: {tb_log_dir}')
    print(f'To view results, run:')
    print(f'  tensorboard --logdir={logs_dir}')
    print(f'\nResults saved in: {figure_dir}')
    print(f'Models saved in: {models_dir}')


if __name__ == '__main__':
    main()
