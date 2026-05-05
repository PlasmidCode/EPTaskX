import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_size=64, num_layers=2):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.head(out[:, -1, :])
        return out

class CentralizedTrainer:
    def __init__(self, data_path="/home/plasmid/Project/FLMixedPrediction/data/Renewable-energy-generation-input-feature-variables-analysis-main/data_processed/", window_size=24, batch_size=32, 
                 learning_rate=0.001, epochs=50, hidden_size=64, num_layers=2):
        self.data_path = data_path
        self.window_size = window_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.scalers = {} 
        self.models = {} 
        # 使用脚本所在目录作为结果目录的基础
        self.results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results") 
        self.setup_results_dir()
    
    def setup_results_dir(self):
        os.makedirs(os.path.join(self.results_dir, "figures"), exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, "logs"), exist_ok=True)
    
    def generate_sample_data(self, site_name, n_samples=150):
        """生成样本数据用于测试"""
        print(f"Generating sample data for site: {site_name}")
        
        # 创建时间序列
        start_date = pd.to_datetime("2023-01-01 00:00:00")
        datetime_index = pd.date_range(start=start_date, periods=n_samples, freq="H")
        
        # 生成特征数据
        wind_speed = np.random.normal(5.0, 1.5, n_samples)
        wind_direction = np.random.uniform(0, 360, n_samples)
        temperature = np.random.normal(10.0, 2.0, n_samples)
        humidity = np.random.normal(75.0, 5.0, n_samples)
        
        # 生成功率数据（基于风速的简单函数）
        power = 20 * wind_speed + np.random.normal(0, 5.0, n_samples)
        power = np.maximum(power, 0)  # 确保功率不为负
        
        # 创建DataFrame
        df = pd.DataFrame({
            "datetime": datetime_index,
            "wind_speed": wind_speed,
            "wind_direction": wind_direction,
            "temperature": temperature,
            "humidity": humidity,
            "power": power
        })
        
        df.set_index("datetime", inplace=True)
        return df
    
    def preprocess_data(self, df):
        """
        预处理数据，沿用旧版单站点实验的特征处理逻辑
        """
        # 处理时间列
        time_columns = ['Time(year-month-day h:m:s)', 'Time', 'time']
        time_col = None
        for col in time_columns:
            if col in df.columns:
                time_col = col
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
        
        return df, time_col
    
    def load_data(self):
        self.datasets = {}
        # 解析绝对路径以确保正确性
        self.data_path = os.path.abspath(self.data_path)
        print(f"Looking for data in: {self.data_path}")
        
        # 确保数据目录存在
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data directory not found: {self.data_path}")
        
        # 获取所有Excel文件
        site_files = []
        
        # 检查是否包含子目录（wind_farms和solar_stations）
        subdirs = [d for d in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, d))]
        
        if subdirs:
            # 遍历子目录
            for subdir in subdirs:
                subdir_path = os.path.join(self.data_path, subdir)
                subdir_files = [os.path.join(subdir_path, f) for f in os.listdir(subdir_path) if f.endswith(".xlsx")]
                site_files.extend(subdir_files)
        else:
            # 直接在当前目录查找Excel文件
            site_files = [os.path.join(self.data_path, f) for f in os.listdir(self.data_path) if f.endswith(".xlsx")]
        
        print(f"Found Excel files: {site_files}")
        
        if not site_files:
            raise ValueError(f"No Excel files found in data directory: {self.data_path}")
        
        # 加载所有数据文件
        for file_path in site_files:
            print(f"Loading file: {file_path}")
            # 提取站点名称
            file_name = os.path.basename(file_path)
            site_name = file_name.split(" (")[0].replace(" ", "_").lower()
            
            # 读取Excel文件
            df = pd.read_excel(file_path)
            
            # 预处理数据
            df_processed, time_col = self.preprocess_data(df)
            
            # 设置时间索引
            if time_col is not None:
                df_processed.set_index(time_col, inplace=True)
            
            # 可选：将15分钟数据降采样为小时数据
            df_processed = df_processed.resample('H').mean()
            
            self.datasets[site_name] = df_processed
        
        print(f"Loaded datasets: {list(self.datasets.keys())}")
        print(f"Total datasets loaded: {len(self.datasets)}")
        # 显示每个数据集的形状
        for site_name, df in self.datasets.items():
            print(f"{site_name}: {df.shape}")
    
    def prepare_features(self, df, site_name):
        """
        构建特征和目标数据
        确保所有站点的特征数量一致
        """
        # 处理缺失值
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # 定义标准特征集
        if 'wind' in site_name:
            # 风能站点标准特征集
            standard_features = [
                'Wind speed at height of 10 meters (m/s)',
                'Wind direction at height of 10 meters (˚)',
                'Wind speed at height of 30 meters (m/s)',
                'Wind direction at height of 30 meters (˚)',
                'Wind speed at height of 50 meters (m/s)',
                'Wind direction at height of 50 meters (˚)',
                'Wind speed - at the height of wheel hub (m/s)',
                'Wind speed - at the height of wheel hub (˚)',
                'Air temperature  (°C) '
            ]
        else:
            # 光伏站点标准特征集
            standard_features = [
                'Total solar irradiance (W/m2)',
                'Direct normal irradiance (W/m2)',
                'Global horizontal irradiance (W/m2)',
                'Air temperature  (°C) ',
                'Atmosphere (hpa)',
                'Relative humidity (%)'
            ]
        
        # 提取并标准化特征
        df_features = df.copy()
        
        # 确保所有标准特征存在，缺失的用0填充
        for feature in standard_features:
            if feature not in df_features.columns:
                print(f"Warning: Feature '{feature}' not found in {site_name}, filling with 0")
                df_features[feature] = 0.0
        
        # 只保留标准特征
        feature_cols = standard_features
        
        # 处理特征中的异常值
        df_features[feature_cols] = df_features[feature_cols].apply(lambda x: x.clip(lower=x.quantile(0.01), upper=x.quantile(0.99)), axis=0)
        
        # 处理目标变量中的异常值
        df_features['power'] = np.clip(df_features['power'], df_features['power'].quantile(0.01), df_features['power'].quantile(0.99))
        
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()
        
        features = df_features[feature_cols].values
        target = df_features['power'].values.reshape(-1, 1)
        
        features_scaled = scaler_x.fit_transform(features)
        target_scaled = scaler_y.fit_transform(target)
        
        # 处理可能出现的NaN值（替换为0）
        features_scaled = np.nan_to_num(features_scaled, nan=0.0)
        target_scaled = np.nan_to_num(target_scaled, nan=0.0)
        
        self.scalers[site_name] = (scaler_x, scaler_y)
        
        # 构建序列数据
        X, y = [], []
        vals = np.hstack((features_scaled, target_scaled))
        
        for i in range(self.window_size, len(vals)):
            X.append(vals[i - self.window_size:i, :len(feature_cols)])
            y.append(vals[i, -1])
        
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)  # 保持与原代码兼容的形状
        
        return X, y
    
    def merge_all_data(self):
        """
        合并所有站点的数据，按站点类型分别合并
        """
        # 按站点类型分组
        wind_sites = {site: df for site, df in self.datasets.items() if 'wind_farm' in site}
        solar_sites = {site: df for site, df in self.datasets.items() if 'solar_station' in site}
        
        print(f"Found {len(wind_sites)} wind farm sites")
        print(f"Found {len(solar_sites)} solar station sites")
        
        # 分别处理风能和光伏站点
        self.wind_data = self._merge_data_by_type(wind_sites, "wind")
        self.solar_data = self._merge_data_by_type(solar_sites, "solar")
    
    def _merge_data_by_type(self, sites, site_type):
        """
        按站点类型合并数据
        """
        all_X_train = []
        all_y_train = []
        all_X_val = []
        all_y_val = []
        
        # 定义标准特征集，与prepare_features方法保持一致
        if site_type == "wind":
            # 风能站点标准特征集
            standard_features = [
                'Wind speed at height of 10 meters (m/s)',
                'Wind direction at height of 10 meters (˚)',
                'Wind speed at height of 30 meters (m/s)',
                'Wind direction at height of 30 meters (˚)',
                'Wind speed at height of 50 meters (m/s)',
                'Wind direction at height of 50 meters (˚)',
                'Wind speed - at the height of wheel hub (m/s)',
                'Wind speed - at the height of wheel hub (˚)',
                'Air temperature  (°C) '
            ]
            standard_feature_count = len(standard_features)
        else:
            # 光伏站点标准特征集
            standard_features = [
                'Total solar irradiance (W/m2)',
                'Direct normal irradiance (W/m2)',
                'Global horizontal irradiance (W/m2)',
                'Air temperature  (°C) ',
                'Atmosphere (hpa)',
                'Relative humidity (%)'
            ]
            standard_feature_count = len(standard_features)
        
        print(f"{site_type} sites standard feature count: {standard_feature_count}")
        
        # 处理所有站点
        selected_sites = sites
        print(f"Selected {len(selected_sites)} {site_type} sites with {standard_feature_count} features each")
        print(f"Selected {site_type} sites: {list(selected_sites.keys())}")
        
        for site_name, df in selected_sites.items():
            print(f"Processing {site_type} site {site_name} data with shape: {df.shape}")
            X, y = self.prepare_features(df, site_name)
            print(f"{site_type} site {site_name} features shape after prepare_features: {X.shape}, {y.shape}")
            
            if len(X) == 0:
                print(f"Warning: No features generated for {site_type} site {site_name}. Skipping.")
                continue
            
            # 确保有足够的数据进行分割
            split1 = max(1, int(0.7 * len(X)))
            split2 = max(split1 + 1, int(0.85 * len(X)))
            
            if split1 >= len(X) - 1:
                print(f"Warning: Not enough data for {site_type} site {site_name} to split. Using all data for training.")
                X_train, X_val = X[:len(X)//2], X[len(X)//2:]
                y_train, y_val = y[:len(y)//2], y[len(y)//2:]
            else:
                X_train, X_val, _ = X[:split1], X[split1:split2], X[split2:]
                y_train, y_val, _ = y[:split1], y[split1:split2], y[split2:]
            
            print(f"{site_type} site {site_name} split data shapes - Train: {X_train.shape}, Val: {X_val.shape}")
            
            if len(X_train) > 0:
                all_X_train.append(X_train)
                all_y_train.append(y_train)
            if len(X_val) > 0:
                all_X_val.append(X_val)
                all_y_val.append(y_val)
        
        # 确保至少有一个数据集被合并
        if all_X_train:
            X_train = np.concatenate(all_X_train, axis=0)
            y_train = np.concatenate(all_y_train, axis=0)
        else:
            # 如果没有训练数据，生成一些示例数据
            print(f"Warning: No {site_type} training data available. Generating sample training data.")
            X_train = np.random.randn(10, self.window_size, standard_feature_count)
            y_train = np.random.randn(10, 1)
        
        if all_X_val:
            X_val = np.concatenate(all_X_val, axis=0)
            y_val = np.concatenate(all_y_val, axis=0)
        else:
            # 如果没有验证数据，生成一些示例数据
            print(f"Warning: No {site_type} validation data available. Generating sample validation data.")
            X_val = np.random.randn(5, self.window_size, standard_feature_count)
            y_val = np.random.randn(5, 1)
        
        print(f"Merged {site_type} data: Train samples={len(X_train)}, Val samples={len(X_val)}")
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'feature_count': standard_feature_count,
            'sites': list(selected_sites.keys())
        }
    
    def train_global_model(self, data, site_type):
        """
        训练全局模型
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        input_dim = data['X_train'].shape[2]
        model = LSTMModel(input_dim, self.hidden_size, self.num_layers).to(device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        
        train_dataset = TensorDataset(torch.Tensor(data['X_train']), torch.Tensor(data['y_train']))
        val_dataset = TensorDataset(torch.Tensor(data['X_val']), torch.Tensor(data['y_val']))
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        writer = SummaryWriter(log_dir=os.path.join(self.results_dir, "logs", f"centralized_{site_type}"))
        
        train_losses = []
        val_losses = []
        
        for epoch in range(self.epochs):
            model.train()
            train_loss = 0.0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
            
            train_loss /= len(train_loader.dataset)
            train_losses.append(train_loss)
            
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)
            
            val_loss /= len(val_loader.dataset)
            val_losses.append(val_loss)
            
            writer.add_scalars(f"Loss/{site_type}", {
                "train": train_loss,
                "val": val_loss
            }, epoch)
            
            if (epoch + 1) % 10 == 0:
                print(f"{site_type.upper()} Global Model - Epoch: {epoch+1}/{self.epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        writer.close()
        
        return model, train_losses, val_losses
    
    def evaluate_model(self, model, site_name, X_test, y_test):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        with torch.no_grad():
            inputs = torch.Tensor(X_test).to(device)
            outputs = model(inputs).cpu().numpy()
        
        scaler_y = self.scalers[site_name][1]
        y_test_original = scaler_y.inverse_transform(y_test)
        outputs_original = scaler_y.inverse_transform(outputs)
        
        mse = mean_squared_error(y_test_original, outputs_original)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_original, outputs_original)
        r2 = r2_score(y_test_original, outputs_original)
        
        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "y_true": y_test_original.flatten(),
            "y_pred": outputs_original.flatten()
        }
    
    def save_model(self, model, site_type):
        model_path = os.path.join(self.results_dir, "models", f"centralized_{site_type}_global_model.pth")
        torch.save(model.state_dict(), model_path)
    
    def plot_loss_curve(self, train_losses, val_losses, site_type):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, self.epochs+1), train_losses, label="Train Loss")
        plt.plot(range(1, self.epochs+1), val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("MSE Loss")
        plt.title(f"Loss Curve - Centralized {site_type.upper()} Model")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.results_dir, "figures", f"centralized_{site_type}_global_loss.png"))
        plt.close()
    
    def plot_predictions(self, site_name, y_true, y_pred):
        plt.figure(figsize=(15, 8))
        plt.plot(y_true, label="True Power", linewidth=2)
        plt.plot(y_pred, label="Predicted Power", linewidth=2, alpha=0.7)
        plt.xlabel("Time Steps")
        plt.ylabel("Power Output")
        plt.title(f"Power Prediction - {site_name} (Centralized)")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.results_dir, "figures", f"{site_name}_centralized_predictions.png"))
        plt.close()
    
    def plot_scatter(self, site_name, y_true, y_pred):
        plt.figure(figsize=(10, 8))
        plt.scatter(y_true, y_pred, alpha=0.6, s=50)
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', linewidth=2)
        plt.xlabel("True Power")
        plt.ylabel("Predicted Power")
        plt.title(f"True vs Predicted - {site_name} (Centralized)")
        plt.grid(True)
        plt.savefig(os.path.join(self.results_dir, "figures", f"{site_name}_centralized_scatter.png"))
        plt.close()
    
    def run(self):
        print("Starting run method...")
        self.load_data()
        print(f"Datasets loaded: {len(self.datasets)}")
        
        if not self.datasets:
            print("Error: No datasets loaded!")
            return
        
        print("Merging all data...")
        self.merge_all_data()
        print(f"Merged wind data samples: Train={len(self.wind_data['X_train'])}, Val={len(self.wind_data['X_val'])}")
        print(f"Merged solar data samples: Train={len(self.solar_data['X_train'])}, Val={len(self.solar_data['X_val'])}")
        
        # 训练风能全局模型
        print("\n" + "="*60)
        print("TRAINING WIND FARMS GLOBAL MODEL")
        print("="*60)
        wind_model, wind_train_losses, wind_val_losses = self.train_global_model(self.wind_data, "wind")
        print(f"Wind model training completed. Final train loss: {wind_train_losses[-1]:.6f}, Final val loss: {wind_val_losses[-1]:.6f}")
        self.plot_loss_curve(wind_train_losses, wind_val_losses, "wind")
        self.save_model(wind_model, "wind")
        
        # 训练光伏全局模型
        print("\n" + "="*60)
        print("TRAINING SOLAR STATIONS GLOBAL MODEL")
        print("="*60)
        solar_model, solar_train_losses, solar_val_losses = self.train_global_model(self.solar_data, "solar")
        print(f"Solar model training completed. Final train loss: {solar_train_losses[-1]:.6f}, Final val loss: {solar_val_losses[-1]:.6f}")
        self.plot_loss_curve(solar_train_losses, solar_val_losses, "solar")
        self.save_model(solar_model, "solar")
        
        # 评估所有站点
        print("\n" + "="*60)
        print("EVALUATING ALL SITES")
        print("="*60)
        
        for site_name, df in self.datasets.items():
            print(f"\nEvaluating site: {site_name}")
            
            X, y = self.prepare_features(df, site_name)
            print(f"Site {site_name} feature shape: {X.shape}")
            
            split2 = int(0.85 * len(X))
            if split2 >= len(X):
                print(f"Warning: Not enough data for site {site_name}. Skipping evaluation.")
                continue
            
            X_test, y_test = X[split2:], y[split2:]
            print(f"Site {site_name} test data shape: {X_test.shape}")
            
            if len(X_test) == 0:
                print(f"Warning: No test data for site {site_name}. Skipping evaluation.")
                continue
            
            # 根据站点类型选择合适的模型
            if 'wind_farm' in site_name:
                # 检查特征数量是否匹配
                if X_test.shape[2] != self.wind_data['feature_count']:
                    print(f"Warning: Incompatible feature count for wind site {site_name}. Expected {self.wind_data['feature_count']}, got {X_test.shape[2]}. Skipping evaluation.")
                    continue
                eval_results = self.evaluate_model(wind_model, site_name, X_test, y_test)
            else:
                # 检查特征数量是否匹配
                if X_test.shape[2] != self.solar_data['feature_count']:
                    print(f"Warning: Incompatible feature count for solar site {site_name}. Expected {self.solar_data['feature_count']}, got {X_test.shape[2]}. Skipping evaluation.")
                    continue
                eval_results = self.evaluate_model(solar_model, site_name, X_test, y_test)
            
            print(f"Site {site_name} evaluation completed.")
            
            # 生成并保存图片结果
            self.plot_predictions(site_name, eval_results["y_true"], eval_results["y_pred"])
            self.plot_scatter(site_name, eval_results["y_true"], eval_results["y_pred"])
            
            print(f"Site: {site_name} Evaluation Results:")
            print(f"MSE: {eval_results['mse']:.4f}")
            print(f"RMSE: {eval_results['rmse']:.4f}")
            print(f"MAE: {eval_results['mae']:.4f}")
            print(f"R²: {eval_results['r2']:.4f}")
        
        print("\n=== Centralized Training Completed Successfully! ===")

if __name__ == "__main__":
    # 使用用户指定的数据路径，指向data_processed子目录
    data_path = "/home/plasmid/Project/FLMixedPrediction/data/Renewable-energy-generation-input-feature-variables-analysis-main/data_processed/"
    trainer = CentralizedTrainer(data_path=data_path)
    trainer.run()
