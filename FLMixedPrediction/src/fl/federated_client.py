import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple
from models import FLMoEModel

class FederatedClient:
    """联邦学习客户端"""
    
    def __init__(self, client_id: str, model: FLMoEModel, train_data: Tuple[torch.Tensor, torch.Tensor],
                 val_data: Tuple[torch.Tensor, torch.Tensor], device: str = "cuda",
                 lr: float = 0.001, batch_size: int = 128, local_epochs: int = 5,
                 strategy: str = "fedavg", fedprox_mu: float = 0.01):
        """
        Args:
            client_id: 客户端ID
            model: 客户端模型
            train_data: 训练数据 (X, y)
            val_data: 验证数据 (X, y)
            device: 设备类型
            lr: 学习率
            batch_size: 批次大小
            local_epochs: 本地训练轮数
            strategy: 联邦学习策略 (fedavg, fedprox)
            fedprox_mu: FedProx的mu参数
        """
        self.client_id = client_id
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.strategy = strategy
        self.fedprox_mu = fedprox_mu
        
        # 数据加载
        self.train_data = self._create_data_loader(train_data, batch_size, shuffle=True)
        self.val_data = self._create_data_loader(val_data, batch_size, shuffle=False)
        
        # 模型
        self.model = model.to(device)
        
        # 优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # 损失函数
        self.criterion = nn.MSELoss()
    
    def _create_data_loader(self, data: Tuple[torch.Tensor, torch.Tensor], batch_size: int,
                          shuffle: bool = True) -> torch.utils.data.DataLoader:
        """
        创建数据加载器
        
        Args:
            data: 数据元组 (X, y)
            batch_size: 批次大小
            shuffle: 是否打乱数据
            
        Returns:
            数据加载器
        """
        X, y = data
        dataset = torch.utils.data.TensorDataset(X, y)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def train(self, global_model_state: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], float, int]:
        """
        本地训练
        
        Args:
            global_model_state: 全局模型状态
            
        Returns:
            本地模型状态
            平均训练损失
            训练样本数量
        """
        # 加载全局模型参数
        self.model.load_state_dict(global_model_state)
        
        # 保存全局模型参数用于FedProx
        global_params = {name: param.clone().detach() for name, param in global_model_state.items()}
        
        total_loss = 0.0
        total_samples = 0
        
        # 本地训练
        self.model.train()
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            epoch_samples = 0
            
            for batch_idx, (X, y) in enumerate(self.train_data):
                X, y = X.to(self.device), y.to(self.device)
                
                # 梯度清零
                self.optimizer.zero_grad()
                
                # 模型前向传播
                outputs, load_balancing_loss = self.model(X)
                
                # 计算主损失
                loss = self.criterion(outputs, y)
                
                # 添加负载均衡损失
                total_batch_loss = loss + load_balancing_loss
                
                # FedProx: 添加近端项
                if self.strategy == "fedprox":
                    for name, param in self.model.named_parameters():
                        if name in global_params:
                            total_batch_loss += self.fedprox_mu / 2 * torch.norm(param - global_params[name]) ** 2
                
                # 反向传播和优化
                total_batch_loss.backward()
                self.optimizer.step()
                
                # 累计损失
                epoch_loss += total_batch_loss.item() * X.size(0)
                epoch_samples += X.size(0)
            
            avg_epoch_loss = epoch_loss / epoch_samples
            total_loss += avg_epoch_loss
            total_samples = epoch_samples
            
            # 打印本地训练信息
            print(f"Client {self.client_id}, Epoch {epoch+1}/{self.local_epochs}, Loss: {avg_epoch_loss:.6f}")
        
        avg_loss = total_loss / self.local_epochs
        
        return self.model.state_dict(), avg_loss, total_samples
    
    def evaluate(self, model_state: Dict[str, torch.Tensor] = None) -> Dict[str, float]:
        """
        评估模型
        
        Args:
            model_state: 模型状态（可选，默认使用当前模型）
            
        Returns:
            评估指标字典
        """
        if model_state is not None:
            self.model.load_state_dict(model_state)
        
        self.model.eval()
        
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for X, y in self.val_data:
                X, y = X.to(self.device), y.to(self.device)
                
                # 模型前向传播
                outputs, _ = self.model(X)
                
                # 计算损失
                loss = self.criterion(outputs, y)
                
                # 累计损失
                total_loss += loss.item() * X.size(0)
                total_samples += X.size(0)
        
        avg_loss = total_loss / total_samples
        
        return {
            "loss": avg_loss,
            "rmse": torch.sqrt(torch.tensor(avg_loss)).item()
        }
    
    def get_model_size(self) -> int:
        """
        获取模型大小（参数数量）
        
        Returns:
            参数数量
        """
        return sum(p.numel() for p in self.model.parameters())
    
    def get_gate_weights(self) -> torch.Tensor:
        """
        获取门控权重（用于分析）
        
        Returns:
            门控权重，形状为 [batch_size, num_experts]
        """
        if hasattr(self.model, 'gate_weights'):
            return self.model.gate_weights.detach().cpu()
        return None
