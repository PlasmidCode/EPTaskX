import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .base import BaseScheduler

class MAPPOScheduler(BaseScheduler):
    """
    基于 MAPPO 的调度算法
    """
    def __init__(self, problem, episodes=5000, lr=1e-4, gamma=0.95, clip_eps=0.1, value_coef=0.3, entropy_coef=0.05):
        """
        初始化 MAPPO 调度器
        
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
        self.J = len(problem.tasks)
        
        # 初始化网络（每个任务一个智能体）
        self.policies = [PolicyNetwork(problem.D, problem.T) for _ in range(self.J)]
        self.optimizer = optim.Adam(
            [param for policy in self.policies for param in policy.parameters()],
            lr=lr
        )
    
    def solve(self):
        """
        使用 MAPPO 求解调度问题
        """
        D = self.problem.D
        T = self.problem.T
        J = self.J
        R_hat = self.problem.R_hat
        Cap = self.problem.Cap
        Base = self.problem.Base
        BW_tot = self.problem.BW_tot
        tasks = self.problem.tasks
        alpha = self.problem.alpha
        E_base = self.problem.E_base
        
        # 训练 MAPPO
        for episode in range(self.episodes):
            # 初始化状态
            states = [self._get_initial_state(j) for j in range(J)]
            done = False
            trajectories = [[] for _ in range(J)]
            
            while not done:
                # 获取每个智能体的动作
                actions = []
                log_probs = []
                values = []
                
                for j in range(J):
                    action, log_prob, value = self.policies[j](states[j])
                    actions.append(action)
                    log_probs.append(log_prob)
                    values.append(value)
                
                # 执行动作
                next_states, rewards, done = self._step(states, actions)
                
                # 存储轨迹
                for j in range(J):
                    trajectories[j].append((states[j], actions[j], log_probs[j], values[j], rewards[j], done))
                
                states = next_states
            
            # 计算回报和优势函数
            all_returns = []
            all_advantages = []
            all_states = []
            all_actions = []
            all_old_log_probs = []
            
            for j in range(J):
                trajectory = trajectories[j]
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
                states_j = torch.stack([t[0] for t in trajectory])
                actions_j = torch.stack([t[1] for t in trajectory])
                old_log_probs_j = torch.stack([t[2] for t in trajectory])
                returns_j = torch.tensor(returns, dtype=torch.float32)
                advantages_j = torch.tensor(advantages, dtype=torch.float32)
                
                # 标准化优势函数
                advantages_j = (advantages_j - advantages_j.mean()) / (advantages_j.std() + 1e-8)
                
                all_states.append(states_j)
                all_actions.append(actions_j)
                all_old_log_probs.append(old_log_probs_j)
                all_returns.append(returns_j)
                all_advantages.append(advantages_j)
            
            # 更新策略
            for _ in range(4):  # MAPPO 多次更新
                total_loss = 0
                
                for j in range(J):
                    # 前向传播
                    action_probs, values = self.policies[j].forward(all_states[j].detach())
                    
                    # 计算新的对数概率
                    action_indices = all_actions[j].long().unsqueeze(1)
                    new_log_probs = torch.log(action_probs.gather(1, action_indices)).squeeze(1)
                    
                    # 计算比率
                    ratio = torch.exp(new_log_probs - all_old_log_probs[j].detach())
                    
                    # 计算剪辑损失
                    surr1 = ratio * all_advantages[j].detach()
                    surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * all_advantages[j].detach()
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # 计算价值损失
                    value_loss = self.value_coef * nn.MSELoss()(values.squeeze(), all_returns[j].detach())
                    
                    # 计算熵损失
                    entropy_loss = -self.entropy_coef * (action_probs * torch.log(action_probs + 1e-8)).sum(1).mean()
                    
                    # 总损失
                    agent_loss = policy_loss + value_loss + entropy_loss
                    total_loss += agent_loss
                
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
            # 获取每个智能体的动作
            actions = []
            
            for j in range(J):
                if remaining_work[j] <= 0:
                    actions.append(0)  # 任务已完成，选择默认动作
                    continue
                
                # 获取当前状态
                state = self._get_state(j, t, current_domains[j], remaining_work[j])
                state = torch.tensor(state, dtype=torch.float32)
                
                # 获取动作
                action_probs, _ = self.policies[j].forward(state.unsqueeze(0))
                action = torch.argmax(action_probs, dim=1).item()
                actions.append(action)
            
            # 执行动作
            for j in range(J):
                if remaining_work[j] <= 0:
                    continue
                
                d = actions[j]
                
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
            'algorithm': 'mappo'
        }
    
    def _get_initial_state(self, j):
        """
        获取第 j 个智能体的初始状态
        """
        task = self.problem.tasks[j]
        
        # 状态包括：当前时间槽、剩余工作量、当前域
        state = [0, task['W_j'], task['src_j'] - 1]
        
        return torch.tensor(state, dtype=torch.float32)
    
    def _get_state(self, j, t, current_domain, remaining_work):
        """
        获取第 j 个智能体的当前状态
        """
        state = [t, remaining_work, current_domain]
        return state
    
    def _step(self, states, actions):
        """
        执行动作并返回下一个状态、奖励和是否完成
        """
        D = self.problem.D
        T = self.problem.T
        J = self.J
        
        # 解析状态
        t = int(states[0][0].item())
        
        # 计算奖励
        rewards = []
        next_states = []
        
        for j in range(J):
            state = states[j]
            action = actions[j]
            task = self.problem.tasks[j]
            
            # 解析状态
            remaining_work = state[1].item()
            current_domain = int(state[2].item())
            
            d = action
            reward = 0
            
            if remaining_work > 0 and t < T:
                # 计算可分配的执行量
                available_capacity = self.problem.Cap[d, t] - self.problem.Base[d, t]
                assign_work = min(remaining_work, available_capacity)
                
                # 计算能源使用情况
                energy_consumption = self.problem.E_base[d, t] + self.problem.alpha[d] * assign_work
                if energy_consumption <= self.problem.R_hat[d, t]:
                    reward += 1.0  # 使用清洁能源的奖励
                else:
                    reward -= 1.0  # 使用棕电的惩罚
                
                # 计算迁移惩罚
                if d != current_domain:
                    reward -= 0.1 * task['S_j'] / 100  # 迁移惩罚
                
                # 更新剩余工作量
                remaining_work -= assign_work
                
                # 任务完成奖励
                if remaining_work <= 0:
                    reward += 10.0  # 任务完成奖励
            
            rewards.append(reward)
            
            # 构建下一个状态
            next_t = t + 1
            next_state = [next_t, remaining_work, d]
            next_states.append(torch.tensor(next_state, dtype=torch.float32))
        
        done = t >= T - 1
        
        return next_states, rewards, done

class PolicyNetwork(nn.Module):
    """
    策略网络
    """
    def __init__(self, D, T):
        """
        初始化策略网络
        
        参数:
        D: 域数量
        T: 时间槽数量
        """
        super(PolicyNetwork, self).__init__()
        
        # 状态维度
        state_dim = 3  # 当前时间槽、剩余工作量、当前域
        # 动作维度
        action_dim = D  # 选择一个域
        
        # 网络结构
        self.fc1 = nn.Linear(state_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc_pi = nn.Linear(32, action_dim)
        self.fc_v = nn.Linear(32, 1)
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
