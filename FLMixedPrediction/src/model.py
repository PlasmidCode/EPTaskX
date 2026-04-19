import torch, torch.nn as nn
import math
import torch.nn.functional as F

class TinyLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64, num_layers: int = 1):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


class MOE(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64, num_experts: int = 3, num_layers: int = 1):
        super().__init__()
        self.num_experts = num_experts
        
        # 创建多个专家网络
        self.experts = nn.ModuleList([
            TinyLSTM(input_dim, hidden, num_layers) for _ in range(num_experts)
        ])
        
        # 门控网络
        self.gate_rnn = nn.LSTM(input_dim, hidden, num_layers=num_layers, batch_first=True)
        self.gate_linear = nn.Linear(hidden, num_experts)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # 获取每个专家的输出
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        
        # 计算门控权重
        gate_out, _ = self.gate_rnn(x)
        gate_last = gate_out[:, -1, :]
        gate_weights = self.softmax(self.gate_linear(gate_last))
        
        # 加权求和得到最终输出
        output = torch.sum(expert_outputs * gate_weights, dim=1)
        
        # 保存门控权重用于分析
        self.gate_weights = gate_weights
        
        return output


# 即插即用模块集合 - AMD架构的核心组件
# 这些模块可以独立使用或组合使用，用于时间序列预测任务


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str, target_slice=None):
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x, target_slice)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x, target_slice=None):
        if self.affine:
            x = x - self.affine_bias[target_slice]
            x = x / (self.affine_weight + self.eps * self.eps)[target_slice]
        x = x * self.stdev[:, :, target_slice]
        x = x + self.mean[:, :, target_slice]
        return x


class MDM(nn.Module):
    def __init__(self, input_shape, k=3, c=2, layernorm=True):
        super().__init__()
        self.seq_len = input_shape[0]
        self.k = k
        if self.k > 0:
            self.k_list = [c**i for i in range(k, 0, -1)]
            self.avg_pools = nn.ModuleList([nn.AvgPool1d(kernel_size=k, stride=k) for k in self.k_list])
            self.linears = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(self.seq_len // k, self.seq_len // k),
                        nn.GELU(),
                        nn.Linear(self.seq_len // k, self.seq_len * c // k),
                    )
                    for k in self.k_list
                ]
            )
        self.layernorm = layernorm
        if self.layernorm:
            self.norm = nn.BatchNorm1d(input_shape[0] * input_shape[-1])

    def forward(self, x):
        if self.layernorm:
            x = self.norm(torch.flatten(x, 1, -1)).reshape(x.shape)
        if self.k == 0:
            return x
        sample_x = []
        for i, k in enumerate(self.k_list):
            sample_x.append(self.avg_pools[i](x))
        sample_x.append(x)
        n = len(sample_x)
        for i in range(n - 1):
            tmp = self.linears[i](sample_x[i])
            sample_x[i + 1] = torch.add(sample_x[i + 1], tmp, alpha=1.0)
        return sample_x[n - 1]


class DDI(nn.Module):
    def __init__(self, input_shape, dropout=0.2, patch=12, alpha=0.0, layernorm=True):
        super().__init__()
        self.input_shape = input_shape
        if alpha > 0.0:
            self.ff_dim = 2 ** math.ceil(math.log2(self.input_shape[-1]))
            self.fc_block = nn.Sequential(
                nn.Linear(self.input_shape[-1], self.ff_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.ff_dim, self.input_shape[-1]),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        self.n_history = 1
        self.alpha = alpha
        self.patch = patch
        self.layernorm = layernorm
        if self.layernorm:
            self.norm = nn.BatchNorm1d(self.input_shape[0] * self.input_shape[-1])
        self.norm1 = nn.BatchNorm1d(self.n_history * patch * self.input_shape[-1])
        if self.alpha > 0.0:
            self.norm2 = nn.BatchNorm1d(self.patch * self.input_shape[-1])
        self.agg = nn.Linear(self.n_history * self.patch, self.patch)
        self.dropout_t = nn.Dropout(dropout)

    def forward(self, x):
        if self.layernorm:
            x = self.norm(torch.flatten(x, 1, -1)).reshape(x.shape)
        output = torch.zeros_like(x)
        output[:, :, : self.n_history * self.patch] = x[:, :, : self.n_history * self.patch].clone()
        for i in range(self.n_history * self.patch, self.input_shape[0], self.patch):
            context = output[:, :, i - self.n_history * self.patch : i]
            context = self.norm1(torch.flatten(context, 1, -1)).reshape(context.shape)
            context = F.gelu(self.agg(context))
            context = self.dropout_t(context)
            tmp = context + x[:, :, i : i + self.patch]
            res = tmp
            if self.alpha > 0.0:
                tmp = self.norm2(torch.flatten(tmp, 1, -1)).reshape(tmp.shape)
                tmp = torch.transpose(tmp, 1, 2)
                tmp = self.fc_block(tmp)
                tmp = torch.transpose(tmp, 1, 2)
            output[:, :, i : i + self.patch] = res + self.alpha * tmp
        return output


class TopKGating(nn.Module):
    def __init__(self, input_dim, num_experts, top_k=2, noise_epsilon=1e-5):
        super().__init__()
        self.gate = nn.Linear(input_dim, num_experts)
        self.top_k = top_k
        self.noise_epsilon = noise_epsilon
        self.num_experts = num_experts
        self.w_noise = nn.Parameter(torch.zeros(num_experts, num_experts), requires_grad=True)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)

    def decompostion_tp(self, x, alpha=10):
        output = torch.zeros_like(x)
        kth_largest_val, _ = torch.kthvalue(x, self.num_experts - self.top_k + 1)
        kth_largest_mat = kth_largest_val.unsqueeze(1).expand(-1, self.num_experts)
        mask = x < kth_largest_mat
        x = self.softmax(x)
        output[mask] = alpha * torch.log(x[mask] + 1)
        output[~mask] = alpha * (torch.exp(x[~mask]) - 1)
        return output

    def forward(self, x):
        x = self.gate(x)
        clean_logits = x
        if self.training:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + self.noise_epsilon
            logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
        else:
            logits = clean_logits
        logits = self.decompostion_tp(logits)
        gates = self.softmax(logits)
        return gates


class Expert(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout=0.2):
        super().__init__()
        # 使用LSTM作为专家网络，替换原来的线性模型
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        # x形状: [batch_size, seq_len]
        # 直接将x作为LSTM的输入，seq_len作为特征维度
        # 首先调整形状为: [batch_size, 1, seq_len]，将seq_len作为特征维度
        x = x.unsqueeze(1)
        # LSTM输出: [batch_size, 1, hidden_dim]
        lstm_out, _ = self.lstm(x)
        # 取唯一时间步的输出
        out = lstm_out[:, 0, :]
        # 全连接层输出
        return self.fc(out)


class AMS(nn.Module):
    def __init__(self, input_shape, pred_len, ff_dim=2048, dropout=0.2, loss_coef=1.0, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        # 确保pred_len为1，与horizon参数匹配
        self.pred_len = 1
        self.gating = TopKGating(input_shape[0], num_experts, top_k)
        self.experts = nn.ModuleList(
            [Expert(input_shape[0], self.pred_len, hidden_dim=ff_dim, dropout=dropout) for _ in range(num_experts)]
        )
        self.loss_coef = loss_coef
        assert self.top_k <= self.num_experts

    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def forward(self, x, time_embedding):
        batch_size = x.shape[0]
        feature_num = x.shape[1]
        x = torch.transpose(x, 0, 1)
        time_embedding = torch.transpose(time_embedding, 0, 1)
        output = torch.zeros(feature_num, batch_size, self.pred_len).to(x.device)
        loss = 0
        for i in range(feature_num):
            seq_input = x[i]
            time_info = time_embedding[i]
            gates = self.gating(time_info)
            expert_outputs = torch.zeros(self.num_experts, batch_size, self.pred_len).to(x.device)
            for j in range(self.num_experts):
                expert_outputs[j, :, :] = self.experts[j](seq_input)
            expert_outputs = torch.transpose(expert_outputs, 0, 1)
            gates = gates.unsqueeze(-1).expand(-1, -1, self.pred_len)
            batch_output = (gates * expert_outputs).sum(1)
            output[i, :, :] = batch_output
            importance = gates.sum(0)
            loss += self.loss_coef * self.cv_squared(importance)
        output = torch.transpose(output, 0, 1)
        return output, loss


class AMD(nn.Module):
    def __init__(
        self,
        input_shape,
        pred_len,
        n_block,
        dropout,
        patch,
        k,
        c,
        alpha,
        target_slice,
        norm=True,
        layernorm=True,
    ):
        super().__init__()
        self.target_slice = target_slice
        self.norm = norm
        if self.norm:
            self.rev_norm = RevIN(input_shape[-1])
        self.pastmixing = MDM(input_shape, k=k, c=c, layernorm=layernorm)
        self.fc_blocks = nn.ModuleList(
            [DDI(input_shape, dropout=dropout, patch=patch, alpha=alpha, layernorm=layernorm) for _ in range(n_block)]
        )
        self.moe = AMS(input_shape, pred_len, ff_dim=2048, dropout=dropout, num_experts=8, top_k=2)

    def forward(self, x):
        if self.norm:
            x = self.rev_norm(x, "norm")
        x = torch.transpose(x, 1, 2)
        time_embedding = self.pastmixing(x)
        for fc_block in self.fc_blocks:
            x = fc_block(x)
        x, moe_loss = self.moe(x, time_embedding)
        x = torch.transpose(x, 1, 2)
        if self.norm:
            x = self.rev_norm(x, "denorm", self.target_slice)
        if self.target_slice:
            x = x[:, :, self.target_slice]
        return x, moe_loss
