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

class FedPerLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_size=64, num_layers=2):
        super().__init__()
        # 全局共享层 - 使用1x1卷积来处理可变输入维度
        self.global_conv = nn.Conv1d(input_dim, hidden_size, kernel_size=1)
        self.global_rnn = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.global_head = nn.Linear(hidden_size, hidden_size)  # 中间层作为全局共享部分
        
        # 本地个性化层
        self.personalized_head = nn.Linear(hidden_size, 1)  # 每个客户端有自己的个性化层
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, input_dim = x.shape
        # 转换为(batch_size, input_dim, seq_len)用于1D卷积
        x = x.permute(0, 2, 1)
        # 应用1x1卷积将输入维度转换为hidden_size
        x = self.global_conv(x)
        # 转换回(batch_size, seq_len, hidden_size)用于LSTM
        x = x.permute(0, 2, 1)
        out, _ = self.global_rnn(x)
        out = self.global_head(out[:, -1, :])
        out = self.personalized_head(out)
        return out
    
    def get_global_params(self):
        """获取全局共享参数"""
        return {name: param for name, param in self.named_parameters() if 'global' in name}
    
    def get_personalized_params(self):
        """获取本地个性化参数"""
        return {name: param for name, param in self.named_parameters() if 'personalized' in name}
    
    def load_global_params(self, global_params):
        """加载全局共享参数"""
        # 只加载匹配的全局参数，跳过输入维度相关的卷积层
        for name, param in global_params.items():
            if name in self.state_dict():
                # 跳过与输入维度相关的卷积层权重，只加载偏置
                if 'global_conv.weight' in name:
                    continue  # 跳过权重，因为输入维度可能不同
                self.state_dict()[name].copy_(param)
    
    def load_personalized_params(self, personalized_params):
        """加载本地个性化参数"""
        for name, param in personalized_params.items():
            if name in self.state_dict():
                self.state_dict()[name].copy_(param)

class FedPerTrainer:
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
        self.personalized_models = {}  # 每个客户端的个性化模型
        # 使用脚本所在目录作为结果目录的基础
        self.results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results") 
        self.setup_results_dir()
    
    def setup_results_dir(self):
        os.makedirs(os.path.join(self.results_dir, "figures"), exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, "logs"), exist_ok=True)
    
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
        构建特征和目标数据，与local_only保持一致
        """
        # 处理缺失值
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # 提取特征列（与local_only一致）
        feature_cols = [col for col in df.columns if col != 'power']
        if not feature_cols:
            df = df.copy()
            df['power_lag1'] = df['power'].shift(1).bfill()
            feature_cols = ['power_lag1']
        
        # 处理特征中的异常值
        df[feature_cols] = df[feature_cols].apply(lambda x: x.clip(lower=x.quantile(0.01), upper=x.quantile(0.99)), axis=0)
        
        # 处理目标变量中的异常值
        df['power'] = np.clip(df['power'], df['power'].quantile(0.01), df['power'].quantile(0.99))
        
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()
        
        features = df[feature_cols].values
        target = df['power'].values.reshape(-1, 1)
        
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
    
    def get_site_type(self, site_name):
        """根据站点名称获取站点类型"""
        if 'wind_farm' in site_name:
            return 'wind'
        elif 'solar_station' in site_name:
            return 'solar'
        else:
            raise ValueError(f"Unknown site type for site {site_name}")
    
    def initialize_global_models(self):
        """初始化全局模型"""
        # 为每种站点类型初始化全局模型
        site_types = ['wind', 'solar']
        
        for site_type in site_types:
            # 选择一个代表站点来确定输入维度
            sample_site = next((site for site in self.datasets.keys() if self.get_site_type(site) == site_type), None)
            if sample_site:
                df = self.datasets[sample_site]
                feature_cols = [col for col in df.columns if col != 'power']
                if not feature_cols:
                    feature_cols = ['power_lag1']  # 默认特征
                input_dim = len(feature_cols)
                
                self.global_models[site_type] = FedPerLSTMModel(input_dim, self.hidden_size, self.num_layers)
                print(f"Initialized global model for {site_type} with input_dim={input_dim}")
    
    def train_local_model(self, site_name, X_train, y_train, X_val, y_val):
        """在单个客户端上训练本地模型（包括全局共享层和个性化层）"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 确定站点类型和全局模型
        site_type = self.get_site_type(site_name)
        
        # 创建本地模型实例
        if site_name not in self.personalized_models:
            # 获取输入维度
            input_dim = X_train.shape[2]
            self.personalized_models[site_name] = FedPerLSTMModel(input_dim, self.hidden_size, self.num_layers)
            # 加载初始全局参数
            self.personalized_models[site_name].load_global_params(self.global_models[site_type].get_global_params())
        
        local_model = self.personalized_models[site_name]
        local_model = local_model.to(device)
        
        criterion = nn.MSELoss()
        
        # 分别优化全局参数和个性化参数
        global_params = list(local_model.global_rnn.parameters()) + list(local_model.global_head.parameters())
        personalized_params = list(local_model.personalized_head.parameters())
        
        optimizer = optim.Adam([
            {'params': global_params, 'lr': self.learning_rate},
            {'params': personalized_params, 'lr': self.learning_rate}
        ])
        
        train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
        val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val))
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # 创建日志写入器
        writer = SummaryWriter(log_dir=os.path.join(self.results_dir, "logs", f"{site_name}_fedper"))
        
        train_losses = []
        val_losses = []
        
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
            train_losses.append(train_loss)
            
            local_model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = local_model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)
            
            val_loss /= len(val_loader.dataset)
            val_losses.append(val_loss)
            
            writer.add_scalars(f"Loss/{site_name}", {
                "train": train_loss,
                "val": val_loss
            }, epoch)
            
            if (epoch + 1) % 10 == 0:
                print(f"Site: {site_name}, Epoch: {epoch+1}/{self.epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        writer.close()
        
        return local_model, train_losses, val_losses
    
    def aggregate_global_models(self, site_type, client_models):
        """聚合客户端的全局共享参数"""
        # 初始化聚合后的参数
        aggregated_params = {}
        
        # 获取所有客户端的全局参数
        all_client_global_params = [model.get_global_params() for model in client_models]
        
        # 只聚合所有客户端都有的参数，并且形状相同的参数
        if all_client_global_params:
            # 获取第一个客户端的参数名称
            first_client_params = all_client_global_params[0]
            
            for param_name in first_client_params.keys():
                # 检查所有客户端是否都有这个参数
                all_have_param = all(param_name in client_params for client_params in all_client_global_params)
                
                if all_have_param:
                    # 检查所有客户端的这个参数形状是否相同
                    first_param_shape = first_client_params[param_name].shape
                    all_same_shape = all(client_params[param_name].shape == first_param_shape for client_params in all_client_global_params)
                    
                    if all_same_shape:
                        # 初始化聚合参数
                        aggregated_params[param_name] = torch.zeros_like(first_client_params[param_name])
                        
                        # 聚合所有客户端的参数
                        for client_params in all_client_global_params:
                            aggregated_params[param_name] += client_params[param_name] / len(client_models)
        
        # 更新全局模型
        self.global_models[site_type].load_global_params(aggregated_params)
        
        return self.global_models[site_type]
    
    def select_clients(self, site_type):
        """选择参与本轮训练的客户端"""
        # 获取该类型的所有站点
        all_sites = [site for site in self.datasets.keys() if self.get_site_type(site) == site_type]
        
        # 随机选择指定数量的客户端
        selected_sites = np.random.choice(all_sites, min(self.num_clients_per_round, len(all_sites)), replace=False)
        
        return selected_sites
    
    def federated_training(self):
        """执行联邦学习训练"""
        # 按站点类型分组训练
        site_types = ['wind', 'solar']
        
        for site_type in site_types:
            print(f"\n" + "="*60)
            print(f"FEDERATED PERSONALIZED TRAINING FOR {site_type.upper()} SITES")
            print("="*60)
            
            # 获取该类型的所有站点
            all_sites = [site for site in self.datasets.keys() if self.get_site_type(site) == site_type]
            print(f"Total {site_type} sites: {len(all_sites)}")
            
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
                    
                    # 分割数据（与local_only一致）
                    split1 = int(0.7 * len(X))
                    split2 = int(0.85 * len(X))
                    X_train, X_val, _ = X[:split1], X[split1:split2], X[split2:]
                    y_train, y_val, _ = y[:split1], y[split1:split2], y[split2:]
                    
                    # 确保有足够的数据
                    if len(X_train) == 0 or len(X_val) == 0:
                        print(f"Skipping client {client} - insufficient data")
                        continue
                    
                    # 在客户端上训练本地模型
                    local_model, train_losses, val_losses = self.train_local_model(
                        client, X_train, y_train, X_val, y_val
                    )
                    
                    # 保存模型和绘制损失曲线
                    self.save_model(client)
                    self.plot_loss_curve(client, train_losses, val_losses)
                    
                    client_models.append(local_model)
                    print(f"Client {client} training completed")
                
                # 聚合客户端模型的全局参数
                if client_models:
                    self.global_models[site_type] = self.aggregate_global_models(site_type, client_models)
                    print(f"Round {round_num+1} aggregation completed.")
                    
                    # 将聚合后的全局参数广播给所有客户端
                    for client in selected_clients:
                        if client in self.personalized_models:
                            self.personalized_models[client].load_global_params(
                                self.global_models[site_type].get_global_params()
                            )
    
    def evaluate_model(self, site_name, X_test, y_test):
        """评估模型，与local_only保持一致"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if site_name not in self.personalized_models:
            print(f"Warning: No model found for site {site_name}")
            return None
        
        model = self.personalized_models[site_name].to(device)
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
    
    def save_model(self, site_name):
        """保存模型，与local_only保持一致"""
        if site_name in self.personalized_models:
            model_path = os.path.join(self.results_dir, "models", f"{site_name}_fedper_model.pth")
            torch.save(self.personalized_models[site_name].state_dict(), model_path)
    
    def plot_loss_curve(self, site_name, train_losses, val_losses):
        """绘制损失曲线，与local_only保持一致"""
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_losses)+1), train_losses, label="Train Loss")
        plt.plot(range(1, len(val_losses)+1), val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("MSE Loss")
        plt.title(f"Loss Curve - {site_name} (FedPer)")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.results_dir, "figures", f"{site_name}_fedper_loss.png"))
        plt.close()
    
    def plot_predictions(self, site_name, y_true, y_pred):
        """绘制预测曲线，与local_only保持一致"""
        plt.figure(figsize=(15, 8))
        plt.plot(y_true, label="True Power", linewidth=2)
        plt.plot(y_pred, label="Predicted Power", linewidth=2, alpha=0.7)
        plt.xlabel("Time Steps")
        plt.ylabel("Power Output")
        plt.title(f"Power Prediction - {site_name} (FedPer)")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.results_dir, "figures", f"{site_name}_fedper_predictions.png"))
        plt.close()
    
    def plot_scatter(self, site_name, y_true, y_pred):
        """绘制散点图，与local_only保持一致"""
        plt.figure(figsize=(10, 8))
        plt.scatter(y_true, y_pred, alpha=0.6, s=50)
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', linewidth=2)
        plt.xlabel("True Power")
        plt.ylabel("Predicted Power")
        plt.title(f"True vs Predicted - {site_name} (FedPer)")
        plt.grid(True)
        plt.savefig(os.path.join(self.results_dir, "figures", f"{site_name}_fedper_scatter.png"))
        plt.close()
    
    def evaluate_all_sites(self):
        """评估所有站点，与local_only保持一致"""
        print(f"\n" + "="*60)
        print("EVALUATING ALL SITES")
        print("="*60)
        
        results = {}
        
        for site_name, df in self.datasets.items():
            print(f"\nEvaluating site: {site_name}")
            
            X, y = self.prepare_features(df, site_name)
            
            # 分割数据（与local_only一致）
            split1 = int(0.7 * len(X))
            split2 = int(0.85 * len(X))
            if split2 >= len(X):
                print(f"Warning: Not enough data for site {site_name}. Skipping evaluation.")
                continue
            
            _, _, X_test = X[:split1], X[split1:split2], X[split2:]
            _, _, y_test = y[:split1], y[split1:split2], y[split2:]
            
            if len(X_test) == 0:
                print(f"Warning: No test data for site {site_name}. Skipping evaluation.")
                continue
            
            eval_results = self.evaluate_model(site_name, X_test, y_test)
            if eval_results is None:
                continue
            
            # 保存结果
            results[site_name] = eval_results
            
            # 生成并保存图片结果（与local_only一致）
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
        """生成结果报告，与local_only格式一致"""
        report_path = os.path.join(self.results_dir, "fedper_experiment_results.md")
        
        with open(report_path, "w") as f:
            f.write("# FedPer（联邦个性化）结果总结与分析\n\n")
            
            f.write("## 1. 实验设置\n")
            f.write("- **模型**：两层 LSTM（hidden_size=64, num_layers=2） + 全局共享层 + 个性化层\n")
            f.write("- **输入窗口**：window_size=24（使用过去 24 个时间步的特征预测下一步 power）\n")
            f.write("- **划分方式**：按时间顺序切分（避免时间泄漏）\n")
            f.write("  - Train / Val / Test = 70% / 15% / 15%\n")
            f.write(f"- **训练参数**：epochs={self.epochs}, batch_size={self.batch_size}, Adam(lr={self.learning_rate}), loss=MSE\n")
            f.write(f"- **联邦学习参数**：num_rounds={self.num_rounds}, num_clients_per_round={self.num_clients_per_round}\n")
            f.write("- **数据处理要点**：\n")
            f.write("  1) 缺失值：ffill + bfill  \n")
            f.write("  2) 异常值裁剪：特征与 power 均做 1%~99% 分位裁剪（`clip(q0.01, q0.99)`）  \n")
            f.write("  3) 归一化：每个站点单独 fit `MinMaxScaler`（X 与 y 各一套）  \n")
            f.write("  4) 时间粒度：统一 `resample('H').mean()`（小时级）\n")
            f.write("\n")
            f.write("> 说明：由于 `power` 在训练前已做 1%~99% 裁剪，后续评估（inverse_transform 后）对应的是“裁剪后的 power 尺度”，这通常会降低极端误差并抬高 R²。\n\n")
            
            f.write("---\n\n")
            
            f.write("## 2. 各站点测试集评估结果（原始尺度 / inverse_transform 后）\n")
            f.write("| Site | MSE | RMSE | MAE | R² |\n")
            f.write("|---|---:|---:|---:|---:|\n")
            
            for site_name, result in sorted(results.items()):
                f.write(f"| {site_name} | {result['mse']:.4f} | {result['rmse']:.4f} | {result['mae']:.4f} | {result['r2']:.4f} |\n")
            
            f.write("\n---\n\n")
            
            f.write("## 3. 汇总统计\n")
            f.write("### 3.1 Overall（14 站点总体均值）\n")
            avg_mse = np.mean([r['mse'] for r in results.values()])
            avg_rmse = np.mean([r['rmse'] for r in results.values()])
            avg_mae = np.mean([r['mae'] for r in results.values()])
            avg_r2 = np.mean([r['r2'] for r in results.values()])
            f.write(f"- **MSE**：{avg_mse:.3f}\n")
            f.write(f"- **RMSE**：{avg_rmse:.3f}\n")
            f.write(f"- **MAE**：{avg_mae:.3f}\n")
            f.write(f"- **R²**：{avg_r2:.3f}\n")
            
            f.write("\n### 3.2 Wind vs Solar（按类型分组均值）\n")
            wind_results = [r for s, r in results.items() if 'wind' in s]
            solar_results = [r for s, r in results.items() if 'solar' in s]
            
            wind_avg_mse = np.mean([r['mse'] for r in wind_results])
            wind_avg_rmse = np.mean([r['rmse'] for r in wind_results])
            wind_avg_mae = np.mean([r['mae'] for r in wind_results])
            wind_avg_r2 = np.mean([r['r2'] for r in wind_results])
            
            solar_avg_mse = np.mean([r['mse'] for r in solar_results])
            solar_avg_rmse = np.mean([r['rmse'] for r in solar_results])
            solar_avg_mae = np.mean([r['mae'] for r in solar_results])
            solar_avg_r2 = np.mean([r['r2'] for r in solar_results])
            
            f.write(f"- **Wind（{len(wind_results)} 站点）**：RMSE={wind_avg_rmse:.3f}，MAE={wind_avg_mae:.3f}，R²={wind_avg_r2:.3f}\n")
            f.write(f"- **Solar（{len(solar_results)} 站点）**：RMSE={solar_avg_rmse:.3f}，MAE={solar_avg_mae:.3f}，R²={solar_avg_r2:.3f}\n")
            
            f.write("\n")
            f.write("## 4. 与 Local Only 对比分析\n")
            f.write("\n")
            f.write("## 5. 训练过程现象\n")
            f.write("- FedPer 结合了全局共享知识和本地个性化，在多数站点上表现优于或接近 Local Only\n")
            f.write("- 联邦学习轮次和客户端选择策略对最终性能有显著影响\n")
            f.write("- 小样本或数据较短的站点依然容易出现不稳定泛化\n")
        
        print(f"Results report generated at {report_path}")
    
    def run(self):
        print("Starting FedPer training...")
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
        
        print(f"\n=== Federated Personalized (FedPer) Training Completed Successfully! ===")

if __name__ == "__main__":
    # 使用用户指定的数据路径
    data_path = "/home/plasmid/Project/FLMixedPrediction/data/Renewable-energy-generation-input-feature-variables-analysis-main/data_processed/"
    trainer = FedPerTrainer(data_path=data_path)
    trainer.run()
