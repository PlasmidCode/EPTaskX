import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
import logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from src.model import TinyLSTM
from src.data_processor import RenewableEnergyDataProcessor

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

class LSTMTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'Using device: {self.device}')
        
        # 获取项目根目录
        self.project_root = Path(__file__).parent.parent
        
        # 创建figure文件夹
        self.figure_dir = self.project_root / 'figure'
        self.figure_dir.mkdir(exist_ok=True)
        logger.info(f'Created figure directory: {self.figure_dir}')
        
        # 创建tensorboard日志目录
        self.log_dir = self.project_root / 'logs' / f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        self.log_dir.parent.mkdir(exist_ok=True)
        self.writer = SummaryWriter(str(self.log_dir))
        
        # 初始化数据处理器
        self.data_processor = RenewableEnergyDataProcessor(config['data_dir'])
        
        # 初始化模型
        self.model = None
        self.scaler = None
    
    def load_data(self, site_type, site_index=0):
        """
        加载指定站点的数据
        
        Args:
            site_type: 站点类型，'wind'或'solar'
            site_index: 站点索引
            
        Returns:
            X_train, y_train, X_val, y_val, feature_cols
        """
        logger.info(f'Loading {site_type} data...')
        
        # 获取站点列表
        if site_type == 'wind':
            sites = self.data_processor.list_wind_farms()
        else:
            sites = self.data_processor.list_solar_stations()
        
        # 选择指定站点
        selected_site = sites[site_index]
        logger.info(f'Selected {site_type} site: {Path(selected_site).stem}')
        
        # 获取模型数据
        X, y, feature_cols = self.data_processor.get_data_for_model(
            selected_site, site_type, 
            seq_len=self.config['seq_len'], 
            horizon=self.config['horizon']
        )
        
        logger.info(f'Raw data shape - X: {X.shape}, y: {y.shape}')
        
        # 数据标准化
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        
        # 划分训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=self.config['val_size'], random_state=self.config['seed']
        )
        
        logger.info(f'Train data shape - X: {X_train.shape}, y: {y_train.shape}')
        logger.info(f'Validation data shape - X: {X_val.shape}, y: {y_val.shape}')
        
        return X_train, y_train, X_val, y_val, feature_cols, Path(selected_site).stem
    
    def prepare_dataloaders(self, X_train, y_train, X_val, y_val):
        """
        准备数据加载器
        
        Args:
            X_train: 训练特征
            y_train: 训练目标
            X_val: 验证特征
            y_val: 验证目标
            
        Returns:
            train_loader, val_loader
        """
        # 转换为tensor
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), 
            torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val), 
            torch.FloatTensor(y_val)
        )
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False
        )
        
        return train_loader, val_loader
    
    def train(self, train_loader, val_loader, site_name, site_type):
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            site_name: 站点名称
            site_type: 站点类型
            
        Returns:
            train_losses, val_losses
        """
        logger.info(f'Starting training for {site_name}...')
        
        # 获取输入维度
        input_dim = next(iter(train_loader))[0].shape[-1]
        
        # 初始化模型
        self.model = TinyLSTM(
            input_dim=input_dim, 
            hidden=self.config['hidden_size'], 
            num_layers=self.config['num_layers']
        ).to(self.device)
        
        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config['lr'])
        
        train_losses = []
        val_losses = []
        
        # 开始训练
        for epoch in range(self.config['epochs']):
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                # 前向传播
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * batch_X.size(0)
            
            # 计算平均训练损失
            train_loss = train_loss / len(train_loader.dataset)
            train_losses.append(train_loss)
            
            # 验证模型
            val_loss = self.evaluate(val_loader, criterion)
            val_losses.append(val_loss)
            
            # 记录tensorboard
            self.writer.add_scalar(f'{site_type}/{site_name}/Loss/train', train_loss, epoch)
            self.writer.add_scalar(f'{site_type}/{site_name}/Loss/val', val_loss, epoch)
            
            # 记录日志
            if (epoch + 1) % self.config['log_interval'] == 0:
                logger.info(f'Epoch [{epoch+1}/{self.config["epochs"]}], ' +
                           f'Train Loss: {train_loss:.4f}, ' +
                           f'Val Loss: {val_loss:.4f}')
        
        # 关闭tensorboard writer
        self.writer.close()
        
        logger.info(f'Training completed for {site_name}!')
        return train_losses, val_losses
    
    def evaluate(self, data_loader, criterion):
        """
        评估模型
        
        Args:
            data_loader: 数据加载器
            criterion: 损失函数
            
        Returns:
            平均损失
        """
        self.model.eval()
        loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in data_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_X)
                batch_loss = criterion(outputs, batch_y)
                loss += batch_loss.item() * batch_X.size(0)
        
        return loss / len(data_loader.dataset)
    
    def predict(self, X):
        """
        模型预测
        
        Args:
            X: 输入数据
            
        Returns:
            预测结果
        """
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            return outputs.cpu().numpy()
    
    def calculate_metrics(self, y_true, y_pred):
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
    
    def plot_loss(self, train_losses, val_losses, site_name, site_type):
        """
        绘制损失曲线
        
        Args:
            train_losses: 训练损失列表
            val_losses: 验证损失列表
            site_name: 站点名称
            site_type: 站点类型
        """
        plt.figure(figsize=(12, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title(f'{site_type.upper()} - {site_name} - Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.grid(True)
        
        # 保存图片
        plt.savefig(str(self.figure_dir / f'{site_type}_{site_name}_loss.png'))
        plt.close()
        logger.info(f'Loss plot saved: {site_type}_{site_name}_loss.png')
    
    def plot_predictions(self, y_true, y_pred, site_name, site_type):
        """
        绘制预测结果
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            site_name: 站点名称
            site_type: 站点类型
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
        plt.savefig(str(self.figure_dir / f'{site_type}_{site_name}_predictions.png'))
        plt.close()
        logger.info(f'Predictions plot saved: {site_type}_{site_name}_predictions.png')
        
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
        plt.savefig(str(self.figure_dir / f'{site_type}_{site_name}_scatter.png'))
        plt.close()
        logger.info(f'Scatter plot saved: {site_type}_{site_name}_scatter.png')
    
    def print_metrics(self, metrics, site_name, site_type):
        """
        打印评估指标
        
        Args:
            metrics: 评估指标字典
            site_name: 站点名称
            site_type: 站点类型
        """
        logger.info(f'\n--- {site_type.upper()} - {site_name} Evaluation Metrics ---')
        for metric_name, value in metrics.items():
            logger.info(f'{metric_name}: {value:.4f}')
            self.writer.add_scalar(f'{site_type}/{site_name}/Metrics/{metric_name}', value, 0)
    
    def run_experiment(self, site_type, site_index=0):
        """
        运行完整的实验流程
        
        Args:
            site_type: 站点类型，'wind'或'solar'
            site_index: 站点索引
        """
        # 加载数据
        X_train, y_train, X_val, y_val, feature_cols, site_name = self.load_data(site_type, site_index)
        
        # 准备数据加载器
        train_loader, val_loader = self.prepare_dataloaders(X_train, y_train, X_val, y_val)
        
        # 训练模型
        train_losses, val_losses = self.train(train_loader, val_loader, site_name, site_type)
        
        # 绘制损失曲线
        self.plot_loss(train_losses, val_losses, site_name, site_type)
        
        # 模型预测
        y_val_pred = self.predict(X_val)
        
        # 计算评估指标
        metrics = self.calculate_metrics(y_val, y_val_pred)
        
        # 打印评估指标
        self.print_metrics(metrics, site_name, site_type)
        
        # 绘制预测结果
        self.plot_predictions(y_val, y_val_pred, site_name, site_type)
        
        # 保存模型
        model_path = self.project_root / 'models' / f'{site_type}_{site_name}_model.pth'
        model_path.parent.mkdir(exist_ok=True)
        torch.save(self.model.state_dict(), str(model_path))
        logger.info(f'Model saved: {model_path}')
        
        return metrics


def main():
    print("Starting main function...")
    # 配置参数
    config = {
        'data_dir': '/home/plasmid/Project/FLMixedPrediction/data/Renewable-energy-generation-input-feature-variables-analysis-main/data_processed',
        'seq_len': 24,
        'horizon': 1,
        'hidden_size': 64,
        'num_layers': 2,
        'batch_size': 32,
        'lr': 0.001,
        'epochs': 50,
        'val_size': 0.2,
        'seed': 42,
        'log_interval': 5
    }
    
    # 获取项目根目录
    project_root = Path(__file__).parent.parent
    
    # 创建figure文件夹
    figure_dir = project_root / 'figure'
    figure_dir.mkdir(exist_ok=True)
    print(f"Created figure directory: {figure_dir}")
    
    # 创建models文件夹
    models_dir = project_root / 'models'
    models_dir.mkdir(exist_ok=True)
    print(f"Created models directory: {models_dir}")
    
    # 运行风能站点实验
    print('\n' + '='*50)
    print('Running Wind Farm Experiment')
    print('='*50)
    logger.info('\n' + '='*50)
    logger.info('Running Wind Farm Experiment')
    logger.info('='*50)
    
    wind_trainer = LSTMTrainer(config)
    wind_metrics = wind_trainer.run_experiment('wind', site_index=0)
    
    # 运行光伏站点实验
    logger.info('\n' + '='*50)
    logger.info('Running Solar Station Experiment')
    logger.info('='*50)
    
    solar_trainer = LSTMTrainer(config)
    solar_metrics = solar_trainer.run_experiment('solar', site_index=0)
    
    # 汇总结果
    logger.info('\n' + '='*50)
    logger.info('Experiment Results Summary')
    logger.info('='*50)
    
    logger.info('\n--- Wind Farm Metrics ---')
    for metric_name, value in wind_metrics.items():
        logger.info(f'{metric_name}: {value:.4f}')
    
    logger.info('\n--- Solar Station Metrics ---')
    for metric_name, value in solar_metrics.items():
        logger.info(f'{metric_name}: {value:.4f}')
    
    logger.info('\nAll experiments completed successfully!')
    logger.info('Results saved in ./figure directory')
    logger.info('Models saved in ./models directory')
    logger.info(f'Tensorboard logs saved in {wind_trainer.log_dir}')


if __name__ == '__main__':
    main()
