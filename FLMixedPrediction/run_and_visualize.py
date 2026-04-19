import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# 确保使用绝对路径
project_root = Path('/home/plasmid/Project/FLMixedPrediction')

# 定义输出目录
output_dir = project_root / 'results'
output_dir.mkdir(exist_ok=True)

figure_dir = output_dir / 'figures'
figure_dir.mkdir(exist_ok=True)

models_dir = output_dir / 'models'
models_dir.mkdir(exist_ok=True)

logs_dir = output_dir / 'logs'
logs_dir.mkdir(exist_ok=True)

print(f'Project root: {project_root}')
print(f'Output directory: {output_dir}')
print(f'Figure directory: {figure_dir}')
print(f'Models directory: {models_dir}')
print(f'Logs directory: {logs_dir}')

# 设置TensorBoard
current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
tb_writer = SummaryWriter(logs_dir / f'rnn_predictions_{current_time}')
print(f'TensorBoard logs will be saved to: {tb_writer.log_dir}')

# 加载数据
def load_data(site_type, site_index=0):
    """
    加载风能或光伏站点数据
    """
    data_dir = project_root / 'data' / 'Renewable-energy-generation-input-feature-variables-analysis-main' / 'data_processed'
    
    if site_type == 'wind':
        file_path = data_dir / 'wind_farms' / f'Wind farm site {site_index+1} (Nominal capacity-99MW).xlsx'
    else:
        file_path = data_dir / 'solar_stations' / f'Solar station site {site_index+1} (Nominal capacity-50MW).xlsx'
    
    print(f'Loading {site_type} data from: {file_path}')
    df = pd.read_excel(file_path)
    print(f'Data shape: {df.shape}')
    print(f'Data columns: {list(df.columns)}')
    
    return df

# 数据预处理
def preprocess_data(df):
    """
    预处理数据
    """
    # 处理时间列
    time_columns = ['Time(year-month-day h:m:s)', 'Time', 'time']
    for col in time_columns:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.replace(' 24:', ' 00:')
                df[col] = pd.to_datetime(df[col])
            break
    
    # 处理缺失值
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # 确定目标列
    target_col = None
    possible_target_cols = ['Power (MW)', 'power', 'ActivePower', 'Generated Power']
    for col in possible_target_cols:
        if col in df.columns:
            target_col = col
            break
    if target_col is None:
        target_col = df.columns[-1]
    
    # 重命名目标列为'power'
    df = df.rename(columns={target_col: 'power'})
    
    # 移除时间列
    for col in time_columns:
        if col in df.columns:
            df = df.drop(columns=[col])
            break
    
    return df

# 构建特征和目标数据
def build_features(df, seq_len=24, horizon=1):
    """
    构建特征和目标数据
    """
    # 提取特征列
    feature_cols = [col for col in df.columns if col != 'power']
    if not feature_cols:
        df = df.copy()
        df['power_lag1'] = df['power'].shift(1).bfill()
        feature_cols = ['power_lag1']
    
    # 构建序列数据
    X, y = [], []
    vals = df[feature_cols + ['power']].values.astype('float32')
    
    for i in range(seq_len, len(vals) - horizon + 1):
        X.append(vals[i - seq_len:i, :len(feature_cols)])
        y.append(vals[i + horizon - 1, -1])
    
    return np.array(X), np.array(y), feature_cols

# 划分训练集和验证集
def split_train_val(X, y, val_ratio=0.2, seed=42):
    """
    划分训练集和验证集
    """
    np.random.seed(seed)
    idx = np.arange(len(y))
    np.random.shuffle(idx)
    nv = int(len(y) * val_ratio)
    vidx, tidx = idx[:nv], idx[nv:]
    return (X[tidx], y[tidx]), (X[vidx], y[vidx])

# 定义LSTM模型
class LSTMModel(nn.Module):
    """
    LSTM模型
    """
    def __init__(self, input_dim, hidden_size=64, num_layers=2):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)

# 训练模型
def train_model(model, X_train, y_train, X_val, y_val, device, epochs=50, batch_size=32, lr=0.001, site_type='wind', site_name='site_1'):
    """
    训练模型
    """
    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_ys = []
        
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)
                loss = criterion(outputs, yb)
                val_loss += loss.item() * xb.size(0)
                all_preds.extend(outputs.cpu().numpy())
                all_ys.extend(yb.cpu().numpy())
        
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # 记录到TensorBoard
        tb_writer.add_scalar(f'{site_type}/{site_name}/Loss/train', train_loss, epoch)
        tb_writer.add_scalar(f'{site_type}/{site_name}/Loss/val', val_loss, epoch)
        
        # 计算并记录评估指标
        if (epoch + 1) % 10 == 0:
            mse = mean_squared_error(all_ys, all_preds)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(all_ys, all_preds)
            r2 = r2_score(all_ys, all_preds)
            
            tb_writer.add_scalar(f'{site_type}/{site_name}/Metrics/MSE', mse, epoch)
            tb_writer.add_scalar(f'{site_type}/{site_name}/Metrics/RMSE', rmse, epoch)
            tb_writer.add_scalar(f'{site_type}/{site_name}/Metrics/MAE', mae, epoch)
            tb_writer.add_scalar(f'{site_type}/{site_name}/Metrics/R2', r2, epoch)
            
            print(f'{site_type.upper()} {site_name} - Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    return train_losses, val_losses, all_ys, all_preds

# 绘制损失曲线
def plot_loss_curve(train_losses, val_losses, site_type, site_name):
    """
    绘制损失曲线
    """
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.title(f'{site_type.upper()} - {site_name} - Training and Validation Loss', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图片
    filename = f'{site_type}_{site_name}_loss.png'
    filepath = figure_dir / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'Loss curve saved to: {filepath}')
    
    # 记录到TensorBoard
    fig = plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.title(f'{site_type.upper()} - {site_name} - Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    tb_writer.add_figure(f'{site_type}/{site_name}/Loss_Curve', fig)
    plt.close()

# 绘制预测结果
def plot_predictions(y_true, y_pred, site_type, site_name):
    """
    绘制预测结果
    """
    # 绘制前500个样本
    plt.figure(figsize=(15, 8))
    plt.plot(y_true[:500], label='True Values', alpha=0.7, linewidth=2)
    plt.plot(y_pred[:500], label='Predictions', alpha=0.7, linewidth=2)
    plt.title(f'{site_type.upper()} - {site_name} - True vs Predicted Values (First 500 Samples)', fontsize=14)
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Power (MW)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图片
    filename = f'{site_type}_{site_name}_predictions.png'
    filepath = figure_dir / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'Predictions plot saved to: {filepath}')
    
    # 绘制散点图
    plt.figure(figsize=(10, 8))
    plt.scatter(y_true, y_pred, alpha=0.5, s=50, c='blue')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label='Perfect Prediction', linewidth=2)
    plt.title(f'{site_type.upper()} - {site_name} - True vs Predicted Values', fontsize=14)
    plt.xlabel('True Values (MW)', fontsize=12)
    plt.ylabel('Predicted Values (MW)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图片
    filename = f'{site_type}_{site_name}_scatter.png'
    filepath = figure_dir / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'Scatter plot saved to: {filepath}')

# 计算并打印评估指标
def calculate_and_print_metrics(y_true, y_pred, site_type, site_name):
    """
    计算并打印评估指标
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }
    
    print(f'\n--- {site_type.upper()} - {site_name} Evaluation Metrics ---')
    for metric_name, value in metrics.items():
        print(f'{metric_name}: {value:.4f}')
        tb_writer.add_scalar(f'{site_type}/{site_name}/Final/{metric_name}', value, 0)
    
    return metrics

# 主函数
def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 配置参数
    config = {
        'seq_len': 24,
        'horizon': 1,
        'batch_size': 32,
        'epochs': 30,
        'lr': 0.001,
        'hidden_size': 64,
        'num_layers': 2
    }
    
    print(f'\nExperiment Configuration:')
    for key, value in config.items():
        print(f'  {key}: {value}')
    
    # 处理风能站点
    print('\n' + '='*60)
    print('PROCESSING WIND FARM DATA')
    print('='*60)
    
    wind_df = load_data('wind', site_index=0)
    wind_df_processed = preprocess_data(wind_df)
    
    X_wind, y_wind, feature_cols_wind = build_features(wind_df_processed, seq_len=config['seq_len'], horizon=config['horizon'])
    (X_wind_train, y_wind_train), (X_wind_val, y_wind_val) = split_train_val(X_wind, y_wind)
    
    print(f'\nWind Farm Data:')
    print(f'  Total samples: {len(X_wind)}')
    print(f'  Train samples: {len(X_wind_train)}')
    print(f'  Validation samples: {len(X_wind_val)}')
    print(f'  Feature columns: {feature_cols_wind}')
    print(f'  Input dimension: {len(feature_cols_wind)}')
    
    # 创建并训练风能模型
    wind_model = LSTMModel(input_dim=len(feature_cols_wind), hidden_size=config['hidden_size'], num_layers=config['num_layers']).to(device)
    print(f'\nWind Farm Model: {wind_model}')
    
    print(f'\nTraining wind farm model...')
    wind_train_losses, wind_val_losses, wind_y_true, wind_y_pred = train_model(
        wind_model, X_wind_train, y_wind_train, X_wind_val, y_wind_val, device,
        epochs=config['epochs'], batch_size=config['batch_size'], lr=config['lr'],
        site_type='wind', site_name='site_1'
    )
    
    # 保存风能模型
    wind_model_path = models_dir / 'wind_site_1_model.pth'
    torch.save(wind_model.state_dict(), wind_model_path)
    print(f'\nWind model saved to: {wind_model_path}')
    
    # 绘制风能模型结果
    plot_loss_curve(wind_train_losses, wind_val_losses, 'wind', 'site_1')
    plot_predictions(wind_y_true, wind_y_pred, 'wind', 'site_1')
    
    # 计算并打印风能模型指标
    wind_metrics = calculate_and_print_metrics(wind_y_true, wind_y_pred, 'wind', 'site_1')
    
    # 处理光伏站点
    print('\n' + '='*60)
    print('PROCESSING SOLAR STATION DATA')
    print('='*60)
    
    solar_df = load_data('solar', site_index=0)
    solar_df_processed = preprocess_data(solar_df)
    
    X_solar, y_solar, feature_cols_solar = build_features(solar_df_processed, seq_len=config['seq_len'], horizon=config['horizon'])
    (X_solar_train, y_solar_train), (X_solar_val, y_solar_val) = split_train_val(X_solar, y_solar)
    
    print(f'\nSolar Station Data:')
    print(f'  Total samples: {len(X_solar)}')
    print(f'  Train samples: {len(X_solar_train)}')
    print(f'  Validation samples: {len(X_solar_val)}')
    print(f'  Feature columns: {feature_cols_solar}')
    print(f'  Input dimension: {len(feature_cols_solar)}')
    
    # 创建并训练光伏模型
    solar_model = LSTMModel(input_dim=len(feature_cols_solar), hidden_size=config['hidden_size'], num_layers=config['num_layers']).to(device)
    print(f'\nSolar Station Model: {solar_model}')
    
    print(f'\nTraining solar station model...')
    solar_train_losses, solar_val_losses, solar_y_true, solar_y_pred = train_model(
        solar_model, X_solar_train, y_solar_train, X_solar_val, y_solar_val, device,
        epochs=config['epochs'], batch_size=config['batch_size'], lr=config['lr'],
        site_type='solar', site_name='site_1'
    )
    
    # 保存光伏模型
    solar_model_path = models_dir / 'solar_site_1_model.pth'
    torch.save(solar_model.state_dict(), solar_model_path)
    print(f'\nSolar model saved to: {solar_model_path}')
    
    # 绘制光伏模型结果
    plot_loss_curve(solar_train_losses, solar_val_losses, 'solar', 'site_1')
    plot_predictions(solar_y_true, solar_y_pred, 'solar', 'site_1')
    
    # 计算并打印光伏模型指标
    solar_metrics = calculate_and_print_metrics(solar_y_true, solar_y_pred, 'solar', 'site_1')
    
    # 汇总结果
    print('\n' + '='*60)
    print('EXPERIMENT SUMMARY')
    print('='*60)
    
    print('\nWind Farm Model Metrics:')
    for metric, value in wind_metrics.items():
        print(f'  {metric}: {value:.4f}')
    
    print('\nSolar Station Model Metrics:')
    for metric, value in solar_metrics.items():
        print(f'  {metric}: {value:.4f}')
    
    # 关闭TensorBoard
    tb_writer.close()
    
    print('\n' + '='*60)
    print('EXPERIMENT COMPLETED')
    print('='*60)
    print(f'All results saved to: {output_dir}')
    print(f'TensorBoard logs: {tb_writer.log_dir}')
    print(f'Figures saved to: {figure_dir}')
    print(f'Models saved to: {models_dir}')
    print('\nTo view TensorBoard results, run:')
    print(f'  tensorboard --logdir={logs_dir}')
    print('Then open http://localhost:6006 in your browser.')

# 运行主函数
if __name__ == '__main__':
    main()
