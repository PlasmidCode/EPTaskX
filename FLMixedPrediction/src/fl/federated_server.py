import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
from models import FLMoEModel
from .federated_client import FederatedClient

class FederatedServer:
    """联邦学习服务器"""
    
    def __init__(self, model: FLMoEModel, clients: List[FederatedClient], device: str = "cuda",
                 rounds: int = 100, client_fraction: float = 0.5, strategy: str = "fedavg"):
        """
        Args:
            model: 全局模型
            clients: 客户端列表
            device: 设备类型
            rounds: 联邦学习轮数
            client_fraction: 每轮参与的客户端比例
            strategy: 联邦学习策略 (fedavg, fedprox)
        """
        self.global_model = model.to(device)
        self.clients = clients
        self.device = device
        self.rounds = rounds
        self.client_fraction = client_fraction
        self.strategy = strategy
        
        # 每轮参与的客户端数量
        self.num_clients_per_round = max(1, int(len(clients) * client_fraction))
        
        # 历史记录
        self.round_history = []
        
        # 损失函数
        self.criterion = nn.MSELoss()
    
    def select_clients(self) -> List[FederatedClient]:
        """
        选择参与本轮训练的客户端
        
        Returns:
            选中的客户端列表
        """
        if self.num_clients_per_round >= len(self.clients):
            return self.clients
        
        # 随机选择客户端
        selected_indices = np.random.choice(len(self.clients), self.num_clients_per_round, replace=False)
        return [self.clients[i] for i in selected_indices]
    
    def aggregate_models(self, client_updates: List[Tuple[Dict[str, torch.Tensor], int]]) -> Dict[str, torch.Tensor]:
        """
        聚合客户端模型更新
        
        Args:
            client_updates: 客户端更新列表，每个元素为 (模型状态, 样本数量)
            
        Returns:
            聚合后的模型状态
        """
        # 初始化聚合后的模型参数
        aggregated_state = {}
        total_samples = 0
        
        # 计算总样本数
        for state, samples in client_updates:
            total_samples += samples
        
        # 初始化所有参数为0
        first_state, _ = client_updates[0]
        for name, param in first_state.items():
            aggregated_state[name] = torch.zeros_like(param, device=self.device)
        
        # 加权聚合
        for state, samples in client_updates:
            weight = samples / total_samples
            for name, param in state.items():
                aggregated_state[name] += param.to(self.device) * weight
        
        return aggregated_state
    
    def evaluate_global_model(self, test_data: Dict[str, Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, Dict[str, float]]:
        """
        评估全局模型
        
        Args:
            test_data: 测试数据字典，键为客户端ID，值为 (X, y)
            
        Returns:
            评估结果字典
        """
        results = {}
        
        self.global_model.eval()
        
        with torch.no_grad():
            for client_id, (X, y) in test_data.items():
                X = X.to(self.device)
                y = y.to(self.device)
                
                # 模型前向传播
                outputs, _ = self.global_model(X)
                
                # 计算损失
                loss = self.criterion(outputs, y)
                
                # 计算MAE
                mae = nn.L1Loss()(outputs, y)
                
                # 计算RMSE
                rmse = torch.sqrt(loss)
                
                results[client_id] = {
                    "loss": loss.item(),
                    "mae": mae.item(),
                    "rmse": rmse.item()
                }
        
        return results
    
    def train(self, test_data: Dict[str, Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, List[float]]:
        """
        开始联邦学习训练
        
        Args:
            test_data: 测试数据字典，键为客户端ID，值为 (X, y)
            
        Returns:
            训练历史记录
        """
        print(f"开始联邦学习训练，共 {self.rounds} 轮，每轮 {self.num_clients_per_round} 个客户端")
        
        for round_idx in range(self.rounds):
            print(f"\n=== 轮次 {round_idx+1}/{self.rounds} ===")
            
            # 1. 选择客户端
            selected_clients = self.select_clients()
            print(f"选中的客户端: {[client.client_id for client in selected_clients]}")
            
            # 2. 获取全局模型状态
            global_state = self.global_model.state_dict()
            
            # 3. 客户端本地训练
            client_updates = []
            round_loss = 0.0
            
            for client in selected_clients:
                # 本地训练
                client_state, client_loss, client_samples = client.train(global_state)
                
                # 保存更新
                client_updates.append((client_state, client_samples))
                
                # 累计损失
                round_loss += client_loss
            
            # 4. 聚合模型
            aggregated_state = self.aggregate_models(client_updates)
            
            # 5. 更新全局模型
            self.global_model.load_state_dict(aggregated_state)
            
            # 6. 评估全局模型
            test_results = self.evaluate_global_model(test_data)
            
            # 7. 计算平均指标
            avg_loss = sum(result["loss"] for result in test_results.values()) / len(test_results)
            avg_mae = sum(result["mae"] for result in test_results.values()) / len(test_results)
            avg_rmse = sum(result["rmse"] for result in test_results.values()) / len(test_results)
            
            # 8. 记录历史
            round_info = {
                "round": round_idx + 1,
                "clients": [client.client_id for client in selected_clients],
                "loss": avg_loss,
                "mae": avg_mae,
                "rmse": avg_rmse,
                "client_loss": round_loss / len(selected_clients)
            }
            
            self.round_history.append(round_info)
            
            # 9. 打印轮次信息
            print(f"轮次 {round_idx+1} 结果:")
            print(f"  平均损失: {avg_loss:.6f}")
            print(f"  平均MAE: {avg_mae:.6f}")
            print(f"  平均RMSE: {avg_rmse:.6f}")
            print(f"  客户端平均损失: {round_loss / len(selected_clients):.6f}")
        
        return self.round_history
    
    def get_global_model(self) -> FLMoEModel:
        """
        获取全局模型
        
        Returns:
            全局模型
        """
        return self.global_model
    
    def save_global_model(self, path: str) -> None:
        """
        保存全局模型
        
        Args:
            path: 保存路径
        """
        torch.save({
            "model_state": self.global_model.state_dict(),
            "round_history": self.round_history
        }, path)
    
    def load_global_model(self, path: str) -> None:
        """
        加载全局模型
        
        Args:
            path: 加载路径
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.global_model.load_state_dict(checkpoint["model_state"])
        self.round_history = checkpoint.get("round_history", [])
    
    def get_client_gate_weights(self) -> Dict[str, torch.Tensor]:
        """
        获取所有客户端的门控权重（用于分析）
        
        Returns:
            门控权重字典，键为客户端ID，值为门控权重张量
        """
        gate_weights = {}
        for client in self.clients:
            weights = client.get_gate_weights()
            if weights is not None:
                gate_weights[client.client_id] = weights
        
        return gate_weights
    
    def analyze_expert_usage(self) -> Dict[str, float]:
        """
        分析专家使用情况
        
        Returns:
            专家使用比例字典，键为专家ID，值为使用比例
        """
        gate_weights = self.get_client_gate_weights()
        
        if not gate_weights:
            return {}
        
        # 合并所有门控权重
        all_weights = torch.cat(list(gate_weights.values()), dim=0)
        
        # 计算每个专家的平均使用比例
        expert_usage = all_weights.mean(dim=0).cpu().numpy()
        
        return {f"expert_{i}": usage for i, usage in enumerate(expert_usage)}
