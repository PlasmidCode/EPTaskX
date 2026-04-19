import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 加载风能站点数据
data_dir = '/home/plasmid/Project/FLMixedPrediction/data/Renewable-energy-generation-input-feature-variables-analysis-main/data_processed'
print(f'Loading data from: {data_dir}')

# 读取风场数据
wind_file = f'{data_dir}/wind_farms/Wind farm site 1 (Nominal capacity-99MW).xlsx'
print(f'Reading wind farm data: {wind_file}')
df_wind = pd.read_excel(wind_file)
print(f'Wind farm data shape: {df_wind.shape}')
print(f'Wind farm columns: {list(df_wind.columns)}')

# 读取光伏电站数据
solar_file = f'{data_dir}/solar_stations/Solar station site 1 (Nominal capacity-50MW).xlsx'
print(f'Reading solar station data: {solar_file}')
df_solar = pd.read_excel(solar_file)
print(f'Solar station data shape: {df_solar.shape}')
print(f'Solar station columns: {list(df_solar.columns)}')

# 数据预处理函数
def preprocess_data(df):
    # 处理时间列
    time_columns = ['Time(year-month-day h:m:s)', 'Time', 'time']
    for col in time_columns:
        if col in df.columns:
            # 处理24:00:00的情况
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
    np.random.seed(seed)
    idx = np.arange(len(y))
    np.random.shuffle(idx)
    nv = int(len(y) * val_ratio)
    vidx, tidx = idx[:nv], idx[nv:]
    return (X[tidx], y[tidx]), (X[vidx], y[vidx])

# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_size=64, num_layers=2):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)

# 训练模型
def train_model(model, X_train, y_train, X_val, y_val, device, epochs=50, batch_size=32, lr=0.001):
    # 转换为Tensor
    train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
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
        
        # 计算平均训练损失
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
        
        # 计算平均验证损失
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # 打印日志
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    return train_losses, val_losses, all_ys, all_preds

# 计算评估指标
def calculate_metrics(y_true, y_pred):
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

# 打印评估指标
def print_metrics(metrics, site_type, site_name):
    print(f'\n--- {site_type.upper()} - {site_name} Evaluation Metrics ---')
    for metric_name, value in metrics.items():
        print(f'{metric_name}: {value:.4f}')

# 主函数
def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 处理风能数据
    print('\n\n===== Processing Wind Farm Data =====')
    df_wind_processed = preprocess_data(df_wind)
    print(f'Processed wind data shape: {df_wind_processed.shape}')
    print(f'Processed wind columns: {list(df_wind_processed.columns)}')
    
    # 构建风能特征和目标
    X_wind, y_wind, feature_cols_wind = build_features(df_wind_processed, seq_len=24, horizon=1)
    print(f'Wind data - X shape: {X_wind.shape}, y shape: {y_wind.shape}')
    print(f'Wind feature columns: {feature_cols_wind}')
    
    # 划分风能训练集和验证集
    (X_wind_train, y_wind_train), (X_wind_val, y_wind_val) = split_train_val(X_wind, y_wind)
    print(f'Wind train data - X shape: {X_wind_train.shape}, y shape: {y_wind_train.shape}')
    print(f'Wind val data - X shape: {X_wind_val.shape}, y shape: {y_wind_val.shape}')
    
    # 创建风能模型
    wind_model = LSTMModel(input_dim=len(feature_cols_wind), hidden_size=64, num_layers=2).to(device)
    print(f'Wind model: {wind_model}')
    
    # 训练风能模型
    print('\nTraining wind farm model...')
    wind_train_losses, wind_val_losses, wind_y_true, wind_y_pred = train_model(
        wind_model, X_wind_train, y_wind_train, X_wind_val, y_wind_val, device, epochs=20
    )
    
    # 计算并打印风能模型指标
    wind_metrics = calculate_metrics(wind_y_true, wind_y_pred)
    print_metrics(wind_metrics, 'wind', 'site_1')
    
    # 处理光伏数据
    print('\n\n===== Processing Solar Station Data =====')
    df_solar_processed = preprocess_data(df_solar)
    print(f'Processed solar data shape: {df_solar_processed.shape}')
    print(f'Processed solar columns: {list(df_solar_processed.columns)}')
    
    # 构建光伏特征和目标
    X_solar, y_solar, feature_cols_solar = build_features(df_solar_processed, seq_len=24, horizon=1)
    print(f'Solar data - X shape: {X_solar.shape}, y shape: {y_solar.shape}')
    print(f'Solar feature columns: {feature_cols_solar}')
    
    # 划分光伏训练集和验证集
    (X_solar_train, y_solar_train), (X_solar_val, y_solar_val) = split_train_val(X_solar, y_solar)
    print(f'Solar train data - X shape: {X_solar_train.shape}, y shape: {y_solar_train.shape}')
    print(f'Solar val data - X shape: {X_solar_val.shape}, y shape: {y_solar_val.shape}')
    
    # 创建光伏模型
    solar_model = LSTMModel(input_dim=len(feature_cols_solar), hidden_size=64, num_layers=2).to(device)
    print(f'Solar model: {solar_model}')
    
    # 训练光伏模型
    print('\nTraining solar station model...')
    solar_train_losses, solar_val_losses, solar_y_true, solar_y_pred = train_model(
        solar_model, X_solar_train, y_solar_train, X_solar_val, y_solar_val, device, epochs=20
    )
    
    # 计算并打印光伏模型指标
    solar_metrics = calculate_metrics(solar_y_true, solar_y_pred)
    print_metrics(solar_metrics, 'solar', 'site_1')
    
    # 汇总结果
    print('\n\n===== Summary =====')
    print('Wind Farm Model Metrics:')
    for metric, value in wind_metrics.items():
        print(f'  {metric}: {value:.4f}')
    
    print('\nSolar Station Model Metrics:')
    for metric, value in solar_metrics.items():
        print(f'  {metric}: {value:.4f}')


if __name__ == '__main__':
    main()
