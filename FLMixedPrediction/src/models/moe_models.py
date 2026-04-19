import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class SharedEncoder(nn.Module):
    """共享编码器模块，支持TCN、LSTM和Transformer"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int,
                 model_type: str = 'tcn', dropout: float = 0.2):
        super().__init__()
        self.model_type = model_type
        self.hidden_dim = hidden_dim
        
        if model_type == 'tcn':
            # TCN编码器
            from torch.nn.utils import weight_norm
            
            class TCNBlock(nn.Module):
                def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2):
                    super().__init__()
                    self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                                     stride=stride, padding=int((kernel_size-1)*dilation),
                                                     dilation=dilation))
                    self.relu1 = nn.ReLU()
                    self.dropout1 = nn.Dropout(dropout)
                    self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                                     stride=stride, padding=int((kernel_size-1)*dilation),
                                                     dilation=dilation))
                    self.relu2 = nn.ReLU()
                    self.dropout2 = nn.Dropout(dropout)
                    
                    self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
                    self.relu = nn.ReLU()
                    self.init_weights()
                
                def init_weights(self):
                    self.conv1.weight.data.normal_(0, 0.01)
                    self.conv2.weight.data.normal_(0, 0.01)
                    if self.downsample is not None:
                        self.downsample.weight.data.normal_(0, 0.01)
                
                def forward(self, x):
                    out = self.conv1(x)
                    out = self.relu1(out)
                    out = self.dropout1(out)
                    out = self.conv2(out)
                    out = self.relu2(out)
                    out = self.dropout2(out)
                    
                    res = x if self.downsample is None else self.downsample(x)
                    return self.relu(out + res)
            
            layers = []
            in_channels = input_dim
            for i in range(num_layers):
                dilation_size = 2 ** i
                layers.append(TCNBlock(in_channels, hidden_dim, kernel_size=3, stride=1, 
                                      dilation=dilation_size, dropout=dropout))
                in_channels = hidden_dim
            
            self.network = nn.Sequential(*layers)
            
        elif model_type == 'lstm':
            # LSTM编码器
            self.network = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, 
                                  batch_first=True, dropout=dropout)
            
        elif model_type == 'transformer':
            # Transformer编码器
            encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=4, 
                                                     dim_feedforward=hidden_dim, dropout=dropout, 
                                                     batch_first=True)
            self.network = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def forward(self, x):
        """
        Args:
            x: 输入序列，形状为 [batch_size, seq_len, input_dim]
            
        Returns:
            编码器输出，形状为 [batch_size, seq_len, hidden_dim] 或 [batch_size, hidden_dim]
        """
        if self.model_type == 'tcn':
            # TCN需要 [batch_size, input_dim, seq_len] 格式
            x = x.permute(0, 2, 1)  # 转换为 [batch_size, input_dim, seq_len]
            x = self.network(x)
            x = x.permute(0, 2, 1)  # 转换回 [batch_size, seq_len, hidden_dim]
        elif self.model_type == 'lstm':
            # LSTM返回输出序列和隐藏状态
            out, (h_n, c_n) = self.network(x)
            # 使用最后一个时间步的隐藏状态
            x = h_n[-1]  # [batch_size, hidden_dim]
        else:  # transformer
            # Transformer返回整个序列
            x = self.network(x)
            # 使用最后一个时间步的输出
            x = x[:, -1, :]  # [batch_size, hidden_dim]
        
        return x


class ExpertHead(nn.Module):
    """专家网络头，用于生成预测"""
    
    def __init__(self, input_dim: int, horizon: int, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon)
        )
    
    def forward(self, x):
        """
        Args:
            x: 编码器输出，形状为 [batch_size, hidden_dim]
            
        Returns:
            预测结果，形状为 [batch_size, horizon]
        """
        return self.network(x)


class TopKGating(nn.Module):
    """Top-k 门控机制"""
    
    def __init__(self, input_dim: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # 门控网络
        self.gate = nn.Linear(input_dim, num_experts)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        """
        Args:
            x: 编码器输出，形状为 [batch_size, hidden_dim]
            
        Returns:
            门控权重，形状为 [batch_size, num_experts]
        """
        # 计算门控分数
        logits = self.gate(x)
        
        # 应用top-k选择
        top_logits, top_indices = torch.topk(logits, self.top_k, dim=1)
        
        # 仅对top-k专家应用softmax
        zeros = torch.zeros_like(logits)
        gated_logits = zeros.scatter(1, top_indices, top_logits)
        
        # 计算权重
        weights = self.softmax(gated_logits)
        
        return weights


class FLMoEModel(nn.Module):
    """联邦学习混合专家模型"""
    
    def __init__(self, input_dim: int, seq_len: int, horizons: List[int], 
                 num_experts: int = 8, top_k: int = 2, model_type: str = 'tcn',
                 hidden_dim: int = 64, num_layers: int = 3, dropout: float = 0.2,
                 gating_type: str = 'local', load_balancing_loss_weight: float = 0.1):
        super().__init__()
        
        # 配置参数
        self.num_experts = num_experts
        self.top_k = top_k
        self.horizons = horizons
        self.num_horizons = len(horizons)
        self.gating_type = gating_type
        self.load_balancing_loss_weight = load_balancing_loss_weight
        
        # 共享编码器
        self.encoder = SharedEncoder(input_dim, hidden_dim, num_layers, model_type, dropout)
        
        # 专家网络
        self.experts = nn.ModuleList([
            ExpertHead(hidden_dim, max(horizons), hidden_dim, dropout) 
            for _ in range(num_experts)
        ])
        
        # 门控网络
        if gating_type == 'local':
            # 本地门控（每个客户端有自己的门控）
            # 在FL设置中，每个客户端会有自己的门控权重
            self.gating = TopKGating(hidden_dim, num_experts, top_k)
        else:  # global
            # 全局门控（所有客户端共享）
            self.gating = TopKGating(hidden_dim, num_experts, top_k)
    
    def forward(self, x):
        """
        Args:
            x: 输入序列，形状为 [batch_size, seq_len, input_dim]
            
        Returns:
            预测结果，形状为 [batch_size, max_horizon]
            负载均衡损失，标量
        """
        # 编码器输出
        encoder_output = self.encoder(x)
        
        # 计算门控权重
        gate_weights = self.gating(encoder_output)
        
        # 获取每个专家的输出
        expert_outputs = torch.stack([expert(encoder_output) for expert in self.experts], dim=1)
        
        # 加权求和得到最终输出
        output = torch.sum(expert_outputs * gate_weights.unsqueeze(-1), dim=1)
        
        # 计算负载均衡损失（cv_squared）
        importance = gate_weights.sum(0)
        cv_squared = self._cv_squared(importance)
        load_balancing_loss = self.load_balancing_loss_weight * cv_squared
        
        # 保存门控权重用于分析
        self.gate_weights = gate_weights
        
        return output, load_balancing_loss
    
    def _cv_squared(self, x):
        """
        计算变异系数的平方
        """
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)
    
    def get_predictions(self, x, horizon_idx: int = 0):
        """
        获取特定horizon的预测结果
        
        Args:
            x: 输入序列，形状为 [batch_size, seq_len, input_dim]
            horizon_idx: 预测步长索引（0: 1h, 1: 6h, 2: 24h）
            
        Returns:
            预测结果，形状为 [batch_size, horizons[horizon_idx]]
        """
        output, _ = self.forward(x)
        # 获取对应horizon的预测结果
        horizon = self.horizons[horizon_idx]
        return output[:, :horizon]
