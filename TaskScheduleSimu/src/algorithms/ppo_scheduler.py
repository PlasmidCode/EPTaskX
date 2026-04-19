import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .base import BaseScheduler

class PPOScheduler(BaseScheduler):
    """
    基于 PPO 的调度算法
    """
    def __init__(self, problem, episodes=5000, lr=1e-4, gamma=0.95, clip_eps=0.1, value_coef=0.3, entropy_coef=0.05):
        """
        初始化 PPO 调度器
        
        参数:
        problem: 问题模型实例
        episodes: 训练轮数
        lr: 学习率
        gamma: 折扣因子
        clip_eps: 剪辑参数
        value_coef: 价值函数损失系数
        entropy_coef: 熵损失系数
        """
        super().__init__(problem)
        self.episodes = episodes
        self.lr = lr
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # 初始化网络
        self.policy = PolicyNetwork(problem.D, problem.T, len(problem.tasks))
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
    
    def solve(self):
        """
        使用 PPO 求解调度问题
        """
        D = self.problem.D
        T = self.problem.T
        J = self.problem.J
        R_hat = self.problem.R_hat
        Cap = self.problem.Cap
        Base = self.problem.Base
        BW_tot = self.problem.BW_tot
        tasks = self.problem.tasks
        alpha = self.problem.alpha
        E_base = self.problem.E_base
        
        # 训练 PPO
        for episode in range(self.episodes):
            # 初始化状态
            state = self._get_initial_state()
            done = False
            trajectory = []
            
            while not done:
                # 获取动作
                action, log_prob, value = self.policy(state)
                
                # 执行动作
                next_state, reward, done = self._step(state, action)
                
                # 存储轨迹
                trajectory.append((state, action, log_prob, value, reward, done))
                
                state = next_state
            
            # 计算回报
            returns = []
            advantages = []
            G = 0
            
            # 从后向前计算回报和优势函数
            for i in reversed(range(len(trajectory))):
                state, action, log_prob, value, reward, done = trajectory[i]
                G = reward + self.gamma * G * (1 - done)
                returns.insert(0, G)
                
                # 计算优势函数
                if i == len(trajectory) - 1:
                    advantage = G - value.item()
                else:
                    _, _, _, next_value, _, _ = trajectory[i+1]
                    advantage = reward + self.gamma * next_value.item() * (1 - done) - value.item()
                advantages.insert(0, advantage)
            
            # 转换为张量
            states = torch.stack([t[0] for t in trajectory])
            actions = torch.stack([t[1] for t in trajectory])
            old_log_probs = torch.stack([t[2] for t in trajectory])
            returns = torch.tensor(returns, dtype=torch.float32)
            advantages = torch.tensor(advantages, dtype=torch.float32)
            
            # 标准化优势函数
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # 更新策略
            for _ in range(4):  # PPO 多次更新
                # 前向传播
                action_probs, values = self.policy.forward(states.detach())
                
                # 计算新的对数概率
                action_indices = actions.long().unsqueeze(1)
                new_log_probs = torch.log(action_probs.gather(1, action_indices)).squeeze(1)
                
                # 计算比率
                ratio = torch.exp(new_log_probs - old_log_probs.detach())
                
                # 计算剪辑损失
                surr1 = ratio * advantages.detach()
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages.detach()
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 计算价值损失
                value_loss = self.value_coef * nn.MSELoss()(values.squeeze(), returns.detach())
                
                # 计算熵损失
                entropy_loss = -self.entropy_coef * (action_probs * torch.log(action_probs + 1e-8)).sum(1).mean()
                
                # 总损失
                total_loss = policy_loss + value_loss + entropy_loss
                
                # 反向传播
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
        
        # 使用训练好的策略进行调度
        x = np.zeros((J, D, T))  # 执行分配量
        g = np.zeros((D, T))  # 棕电补能
        u = np.zeros(J)  # 违约未完成量
        m = np.zeros((J, T))  # 迁移数据量
        
        # 初始化任务状态
        current_domains = [task['src_j'] - 1 for task in tasks]  # 转换为0索引
        remaining_work = [task['W_j'] for task in tasks]
        
        # 执行调度
        for t in range(T):
            # 获取当前状态
            state = self._get_state(t, current_domains, remaining_work)
            state = torch.tensor(state, dtype=torch.float32)
            
            # 获取动作
            action_probs, _ = self.policy.forward(state.unsqueeze(0))
            action = torch.argmax(action_probs, dim=1).item()
            
            # 解析动作
            j = action // D
            d = action % D
            
            if j < J and remaining_work[j] > 0:
                # 计算可分配的执行量
                available_capacity = Cap[d, t] - Base[d, t] - np.sum(x[:, d, t])
                if available_capacity > 0:
                    assign_work = min(remaining_work[j], available_capacity)
                    x[j, d, t] = assign_work
                    remaining_work[j] -= assign_work
                    
                    # 记录迁移
                    if d != current_domains[j]:
                        m[j, t] = tasks[j]['S_j']
                        current_domains[j] = d
        
        # 计算能耗和棕电补能
        for d in range(D):
            for t in range(T):
                energy_consumption = E_base[d, t] + alpha[d] * np.sum(x[:, d, t])
                if energy_consumption > R_hat[d, t]:
                    g[d, t] = energy_consumption - R_hat[d, t]
        
        # 计算未完成的工作量
        for j in range(J):
            task = tasks[j]
            a_j = task['a_j'] - 1
            d_j = task['d_j'] - 1
            completed_work = np.sum(x[j, :, a_j:d_j+1])
            u[j] = max(0, task['W_j'] - completed_work)
        
        # 计算目标函数值
        objective = np.sum(g) + 0.1 * np.sum(m[:, 1:]) + np.sum([tasks[j]['pi_j'] * u[j] for j in range(J)])
        
        return {
            'status': 'optimal',
            'x': x,
            'g': g,
            'u': u,
            'm': m,
            'objective': objective,
            'algorithm': 'ppo'
        }
    
    def _get_initial_state(self):
        """
        获取初始状态
        """
        D = self.problem.D
        T = self.problem.T
        J = self.problem.J
        
        # 状态包括：当前时间槽、每个任务的剩余工作量、每个任务的当前域
        state = []
        state.append(0)  # 当前时间槽
        
        for task in self.problem.tasks:
            state.append(task['W_j'])  # 剩余工作量
            state.append(task['src_j'] - 1)  # 当前域
        
        return torch.tensor(state, dtype=torch.float32)
    
    def _get_state(self, t, current_domains, remaining_work):
        """
        获取当前状态
        """
        state = []
        state.append(t)  # 当前时间槽
        
        for j in range(len(self.problem.tasks)):
            state.append(remaining_work[j])  # 剩余工作量
            state.append(current_domains[j])  # 当前域
        
        return state
    
    def _step(self, state, action):
        """
        执行动作并返回下一个状态、奖励和是否完成
        """
        D = self.problem.D
        T = self.problem.T
        J = self.problem.J
        
        # 解析动作
        j = action // D
        d = action % D
        
        # 解析状态
        t = int(state[0].item())
        remaining_work = state[1::2].numpy()
        current_domains = state[2::2].numpy().astype(int)
        
        # 计算奖励
        reward = 0
        
        if j < J and remaining_work[j] > 0 and t < T:
            # 计算可分配的执行量
            available_capacity = self.problem.Cap[d, t] - self.problem.Base[d, t]
            assign_work = min(remaining_work[j], available_capacity)
            
            # 计算能源使用情况
            energy_consumption = self.problem.E_base[d, t] + self.problem.alpha[d] * assign_work
            if energy_consumption <= self.problem.R_hat[d, t]:
                reward += 5.0  # 使用清洁能源的奖励
            else:
                reward -= 5.0  # 使用棕电的惩罚
            
            # 计算迁移惩罚
            if d != current_domains[j]:
                reward -= 0.5 * self.problem.tasks[j]['S_j'] / 100  # 迁移惩罚
            
            # 更新剩余工作量
            remaining_work[j] -= assign_work
            
            # 任务完成奖励
            if remaining_work[j] <= 0:
                reward += 20.0  # 任务完成奖励
            
            # 任务进度奖励
            reward += assign_work / self.problem.tasks[j]['W_j'] * 5.0  # 按完成比例奖励
        
        # 更新状态
        t += 1
        done = t >= T
        
        # 构建下一个状态
        next_state = []
        next_state.append(t)
        
        for i in range(J):
            next_state.append(remaining_work[i])
            if i == j:
                next_state.append(d)
            else:
                next_state.append(current_domains[i])
        
        return torch.tensor(next_state, dtype=torch.float32), reward, done

class PolicyNetwork(nn.Module):
    """
    策略网络
    """
    def __init__(self, D, T, J):
        """
        初始化策略网络
        
        参数:
        D: 域数量
        T: 时间槽数量
        J: 任务数量
        """
        super(PolicyNetwork, self).__init__()
        
        # 状态维度
        state_dim = 1 + 2 * J  # 当前时间槽 + 每个任务的剩余工作量和当前域
        # 动作维度
        action_dim = J * D  # 每个任务选择一个域
        
        # 网络结构
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc_pi = nn.Linear(64, action_dim)
        self.fc_v = nn.Linear(64, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        前向传播
        """
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        
        # 策略输出
        pi = self.fc_pi(x)
        pi = torch.softmax(pi, dim=-1)
        
        # 价值输出
        v = self.fc_v(x)
        
        return pi, v
    
    def __call__(self, state):
        """
        获取动作、对数概率和价值
        """
        # 确保状态是 2D 张量
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        pi, v = self.forward(state)
        
        # 采样动作
        action = torch.multinomial(pi, 1)
        
        # 计算对数概率
        log_prob = torch.log(pi.gather(1, action)).squeeze()
        action = action.squeeze()
        
        return action, log_prob, v
