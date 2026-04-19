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

class FedAvgTrainer:
    def __init__(self, data_path="/home/plasmid/Project/FLMixedPrediction/data/Renewable-energy-generation-input-feature-variables-analysis-main/data_processed/", 
                 window_size=24, batch_size=32, learning_rate=0.001, epochs=50, 
                 hidden_size=64, num_layers=2, num_rounds=10, num_clients_per_round=5):
        self.data_path = data_path
        self.window_size = window_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_rounds = num_rounds  # 联邦学习轮次
        self.num_clients_per_round = num_clients_per_round  # 每轮参与训练的客户端数量
        self.scalers = {} 
        self.global_models = {}  # 全局模型（风能和光伏）
        # 使用脚本所在目录作为结果目录的基础
        self.results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results") 
        self.setup_results_dir()
    
    def setup_results_dir(self):
        os.makedirs(os.path.join(self.results_dir, "figures"), exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, "logs"), exist_ok=True)
    
    def preprocess_data(self, df):
        """
        预处理数据，参考run_and_visualize.py中的实现
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
        构建特征和目标数据，确保所有站点的特征数量一致
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
        y = np.array(y).reshape(-1, 1)
        
        return X, y
    
    def get_site_type(self, site_name):
        """根据站点名称获取站点类型"""
        if 'wind_farm' in site_name:
            return 'wind'
        elif 'solar_station' in site_name:
            return 'solar'
        else:
            raise ValueError(f"Unknown site type for site {site_name}")
    
    def get_feature_count(self, site_type):
        """根据站点类型获取特征数量"""
        if site_type == 'wind':
            return 9
        elif site_type == 'solar':
            return 6
        else:
            raise ValueError(f"Unknown site type {site_type}")
    
    def initialize_global_models(self):
        """初始化全局模型"""
        # 初始化风能全局模型
        wind_feature_count = self.get_feature_count('wind')
        self.global_models['wind'] = LSTMModel(wind_feature_count, self.hidden_size, self.num_layers)
        
        # 初始化光伏全局模型
        solar_feature_count = self.get_feature_count('solar')
        self.global_models['solar'] = LSTMModel(solar_feature_count, self.hidden_size, self.num_layers)
        
        print("Initialized global models:")
        print(f"  Wind model: input_dim={wind_feature_count}, hidden_size={self.hidden_size}, num_layers={self.num_layers}")
        print(f"  Solar model: input_dim={solar_feature_count}, hidden_size={self.hidden_size}, num_layers={self.num_layers}")
    
    def train_local_model(self, site_name, local_model, X_train, y_train, X_val, y_val):
        """在单个客户端上训练本地模型"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_model = local_model.to(device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(local_model.parameters(), lr=self.learning_rate)
        
        train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
        val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val))
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        for epoch in range(self.epochs):
            local_model.train()
            train_loss = 0.0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = local_model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
            
            train_loss /= len(train_loader.dataset)
            
            local_model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = local_model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)
            
            val_loss /= len(val_loader.dataset)
        
        return local_model, train_loss, val_loss
    
    def aggregate_models(self, site_type, client_models):
        """聚合客户端模型"""
        global_model = self.global_models[site_type]
        global_dict = global_model.state_dict()
        
        # 初始化聚合后的权重
        aggregated_dict = {k: torch.zeros_like(v) for k, v in global_dict.items()}
        
        # 聚合所有客户端模型的权重
        for local_dict in client_models:
            for k in aggregated_dict.keys():
                aggregated_dict[k] += local_dict[k] / len(client_models)
        
        # 更新全局模型
        global_model.load_state_dict(aggregated_dict)
        self.global_models[site_type] = global_model
        
        return global_model
    
    def select_clients(self, site_type):
        """选择参与本轮训练的客户端"""
        # 获取所有同类型站点
        all_sites = [site for site in self.datasets.keys() if self.get_site_type(site) == site_type]
        
        # 随机选择指定数量的客户端
        selected_sites = np.random.choice(all_sites, min(self.num_clients_per_round, len(all_sites)), replace=False)
        
        return selected_sites
    
    def federated_training(self):
        """执行联邦学习训练"""
        # 按站点类型分组
        site_types = ['wind', 'solar']
        
        for site_type in site_types:
            print(f"\n" + "="*60)
            print(f"FEDERATED TRAINING FOR {site_type.upper()} SITES")
            print("="*60)
            
            # 获取该类型的所有站点
            all_sites = [site for site in self.datasets.keys() if self.get_site_type(site) == site_type]
            print(f"Total {site_type} sites: {len(all_sites)}")
            print(f"Selected sites for federated training: {all_sites}")
            
            # 初始化全局模型
            if site_type not in self.global_models:
                feature_count = self.get_feature_count(site_type)
                self.global_models[site_type] = LSTMModel(feature_count, self.hidden_size, self.num_layers)
            
            # 联邦学习主循环
            for round_num in range(self.num_rounds):
                print(f"\n--- Round {round_num+1}/{self.num_rounds} ---")
                
                # 选择客户端
                selected_clients = self.select_clients(site_type)
                print(f"Selected clients: {selected_clients}")
                
                # 收集客户端模型
                client_models = []
                
                for client in selected_clients:
                    print(f"\nProcessing client: {client}")
                    
                    # 准备客户端数据
                    X, y = self.prepare_features(self.datasets[client], client)
                    
                    # 分割数据
                    split1 = int(0.7 * len(X))
                    split2 = int(0.85 * len(X))
                    X_train, X_val, _ = X[:split1], X[split1:split2], X[split2:]
                    y_train, y_val, _ = y[:split1], y[split1:split2], y[split2:]
                    
                    # 创建本地模型副本
                    local_model = LSTMModel(self.get_feature_count(site_type), self.hidden_size, self.num_layers)
                    local_model.load_state_dict(self.global_models[site_type].state_dict())
                    
                    # 在客户端上训练本地模型
                    local_model, final_train_loss, final_val_loss = self.train_local_model(
                        client, local_model, X_train, y_train, X_val, y_val
                    )
                    
                    print(f"Client {client} training completed - Train Loss: {final_train_loss:.6f}, Val Loss: {final_val_loss:.6f}")
                    
                    # 收集本地模型权重
                    client_models.append(local_model.cpu().state_dict())
                
                # 聚合客户端模型
                if client_models:
                    self.global_models[site_type] = self.aggregate_models(site_type, client_models)
                    print(f"Round {round_num+1} aggregation completed.")
            
            # 保存最终全局模型
            model_path = os.path.join(self.results_dir, "models", f"fedavg_global_{site_type}_model.pth")
            torch.save(self.global_models[site_type].state_dict(), model_path)
            print(f"Saved {site_type} global model to {model_path}")
    
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
    
    def plot_loss_curve(self, train_losses, val_losses, site_type):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_losses)+1), train_losses, label="Train Loss")
        plt.plot(range(1, len(val_losses)+1), val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("MSE Loss")
        plt.title(f"Loss Curve - Federated {site_type.upper()} Model")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.results_dir, "figures", f"fedavg_{site_type}_global_loss.png"))
        plt.close()
    
    def plot_predictions(self, site_name, y_true, y_pred):
        plt.figure(figsize=(15, 8))
        plt.plot(y_true, label="True Power", linewidth=2)
        plt.plot(y_pred, label="Predicted Power", linewidth=2, alpha=0.7)
        plt.xlabel("Time Steps")
        plt.ylabel("Power Output")
        plt.title(f"Power Prediction - {site_name} (FedAvg)")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.results_dir, "figures", f"{site_name}_fedavg_predictions.png"))
        plt.close()
    
    def plot_scatter(self, site_name, y_true, y_pred):
        plt.figure(figsize=(10, 8))
        plt.scatter(y_true, y_pred, alpha=0.6, s=50)
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', linewidth=2)
        plt.xlabel("True Power")
        plt.ylabel("Predicted Power")
        plt.title(f"True vs Predicted - {site_name} (FedAvg)")
        plt.grid(True)
        plt.savefig(os.path.join(self.results_dir, "figures", f"{site_name}_fedavg_scatter.png"))
        plt.close()
    
    def evaluate_all_sites(self):
        """评估所有站点"""
        print(f"\n" + "="*60)
        print("EVALUATING ALL SITES WITH FEDERATED MODELS")
        print("="*60)
        
        results = {}
        
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
            site_type = self.get_site_type(site_name)
            model = self.global_models[site_type]
            
            # 检查特征数量是否匹配
            expected_feature_count = self.get_feature_count(site_type)
            if X_test.shape[2] != expected_feature_count:
                print(f"Warning: Incompatible feature count for {site_type} site {site_name}. Expected {expected_feature_count}, got {X_test.shape[2]}. Skipping evaluation.")
                continue
            
            eval_results = self.evaluate_model(model, site_name, X_test, y_test)
            
            # 保存结果
            results[site_name] = eval_results
            
            print(f"Site {site_name} evaluation completed.")
            
            # 生成并保存图片结果
            self.plot_predictions(site_name, eval_results["y_true"], eval_results["y_pred"])
            self.plot_scatter(site_name, eval_results["y_true"], eval_results["y_pred"])
            
            print(f"Site: {site_name} Evaluation Results:")
            print(f"MSE: {eval_results['mse']:.4f}")
            print(f"RMSE: {eval_results['rmse']:.4f}")
            print(f"MAE: {eval_results['mae']:.4f}")
            print(f"R²: {eval_results['r2']:.4f}")
        
        # 生成结果报告
        self.generate_results_report(results)
        
        return results
    
    def generate_results_report(self, results):
        """生成结果报告"""
        report_path = os.path.join(self.results_dir, "fedavg_experiment_results.md")
        
        with open(report_path, "w") as f:
            f.write("# 联邦平均训练实验结果\n\n")
            f.write("## 实验概述\n")
            f.write(f"本实验使用联邦平均算法，对所有风能和光伏站点进行了联邦模型训练和评估。\n")
            f.write(f"- 联邦学习轮次：{self.num_rounds}\n")
            f.write(f"- 每轮参与客户端数量：{self.num_clients_per_round}\n")
            f.write(f"- 本地训练轮次：{self.epochs}\n\n")
            
            f.write("## 评估结果\n\n")
            f.write("### 评估指标说明\n")
            f.write("- **MSE（均方误差）**：衡量预测值与真实值之间的平均平方差，值越小越好\n")
            f.write("- **RMSE（均方根误差）**：MSE的平方根，单位与目标变量一致，值越小越好\n")
            f.write("- **MAE（平均绝对误差）**：衡量预测值与真实值之间的平均绝对差，值越小越好\n")
            f.write("- **R²（决定系数）**：衡量模型对数据的拟合程度，范围[-∞, 1]，越接近1越好\n\n")
            
            f.write("### 所有站点评估结果\n\n")
            f.write("| 站点名称 | MSE | RMSE | MAE | R² |\n")
            f.write("|---|---:|---:|---:|---:|\n")
            
            for site_name, result in results.items():
                f.write(f"| {site_name} | {result['mse']:.4f} | {result['rmse']:.4f} | {result['mae']:.4f} | {result['r2']:.4f} |\n")
            
            f.write("\n### 风能站点平均结果\n")
            wind_results = [r for s, r in results.items() if 'wind' in s]
            if wind_results:
                avg_mse = np.mean([r['mse'] for r in wind_results])
                avg_rmse = np.mean([r['rmse'] for r in wind_results])
                avg_mae = np.mean([r['mae'] for r in wind_results])
                avg_r2 = np.mean([r['r2'] for r in wind_results])
                f.write(f"- **平均MSE**：{avg_mse:.4f}\n")
                f.write(f"- **平均RMSE**：{avg_rmse:.4f}\n")
                f.write(f"- **平均MAE**：{avg_mae:.4f}\n")
                f.write(f"- **平均R²**：{avg_r2:.4f}\n")
            
            f.write("\n### 光伏站点平均结果\n")
            solar_results = [r for s, r in results.items() if 'solar' in s]
            if solar_results:
                avg_mse = np.mean([r['mse'] for r in solar_results])
                avg_rmse = np.mean([r['rmse'] for r in solar_results])
                avg_mae = np.mean([r['mae'] for r in solar_results])
                avg_r2 = np.mean([r['r2'] for r in solar_results])
                f.write(f"- **平均MSE**：{avg_mse:.4f}\n")
                f.write(f"- **平均RMSE**：{avg_rmse:.4f}\n")
                f.write(f"- **平均MAE**：{avg_mae:.4f}\n")
                f.write(f"- **平均R²**：{avg_r2:.4f}\n")
        
        print(f"Results report generated at {report_path}")
    
    def run(self):
        print("Starting Federated Average training...")
        self.load_data()
        
        if not self.datasets:
            print("Error: No datasets loaded!")
            return
        
        # 初始化全局模型
        self.initialize_global_models()
        
        # 执行联邦学习训练
        self.federated_training()
        
        # 评估所有站点
        self.evaluate_all_sites()
        
        print(f"\n=== Federated Average Training Completed Successfully! ===")

if __name__ == "__main__":
    # 使用用户指定的数据路径
    data_path = "/home/plasmid/Project/FLMixedPrediction/data/Renewable-energy-generation-input-feature-variables-analysis-main/data_processed/"
    trainer = FedAvgTrainer(data_path=data_path)
    trainer.run()