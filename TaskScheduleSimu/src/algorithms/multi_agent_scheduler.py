import numpy as np
from .base import BaseScheduler

class MultiAgentScheduler(BaseScheduler):
    """
    基于多智能体协商的调度算法
    """
    def __init__(self, problem, max_iterations=100):
        """
        初始化多智能体调度器
        
        参数:
        problem: 问题模型实例
        max_iterations: 最大迭代次数
        """
        super().__init__(problem)
        self.max_iterations = max_iterations
    
    def solve(self):
        """
        使用多智能体协商策略求解调度问题
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
        
        # 初始化结果变量
        x = np.zeros((J, D, T))  # 执行分配量
        g = np.zeros((D, T))  # 棕电补能量
        u = np.zeros(J)  # 违约未完成量
        m = np.zeros((J, T))  # 迁移数据量
        
        # 初始化任务智能体的当前域
        current_domains = [task['src_j'] - 1 for task in tasks]  # 转换为0索引
        
        # 计算每个域和时间槽的资源情况
        available_capacity = np.zeros((D, T))
        energy_availability = np.zeros((D, T))
        
        for d in range(D):
            for t in range(T):
                available_capacity[d, t] = Cap[d, t] - Base[d, t]
                energy_availability[d, t] = R_hat[d, t] / (E_base[d, t] + alpha[d] * (Cap[d, t] - Base[d, t]))
        
        # 多智能体协商过程
        for iteration in range(self.max_iterations):
            # 计算当前的资源使用情况
            used_capacity = np.zeros((D, T))
            for j in range(J):
                for d in range(D):
                    for t in range(T):
                        used_capacity[d, t] += x[j, d, t]
            
            # 每个任务作为智能体进行协商
            for j in range(J):
                task = tasks[j]
                a_j = task['a_j'] - 1
                d_j = task['d_j'] - 1
                W_j = task['W_j']
                S_j = task['S_j']
                
                # 计算当前任务的完成情况
                current_completed = np.sum(x[j, :, a_j:d_j+1])
                remaining_work = W_j - current_completed
                
                if remaining_work <= 0:
                    continue
                
                # 寻找最优的域和时间槽组合
                best_score = -float('inf')
                best_d = current_domains[j]
                best_t = a_j
                
                for t in range(a_j, d_j + 1):
                    for d in range(D):
                        # 计算可用容量
                        available = available_capacity[d, t] - used_capacity[d, t]
                        if available <= 0:
                            continue
                        
                        # 计算能源可用性分数
                        energy_score = energy_availability[d, t]
                        
                        # 计算迁移成本（如果从当前域迁移）
                        migration_cost = 0
                        if d != current_domains[j]:
                            migration_cost = S_j / 100  # 迁移成本因子
                        
                        # 计算综合分数
                        score = energy_score - migration_cost
                        
                        if score > best_score:
                            best_score = score
                            best_d = d
                            best_t = t
                
                # 分配任务到最优域和时间槽
                if best_score > -float('inf'):
                    available = available_capacity[best_d, best_t] - used_capacity[best_d, best_t]
                    assign_work = min(remaining_work, available)
                    
                    # 更新执行分配
                    x[j, best_d, best_t] += assign_work
                    used_capacity[best_d, best_t] += assign_work
                    
                    # 记录迁移
                    if best_d != current_domains[j]:
                        m[j, best_t] = S_j
                        current_domains[j] = best_d
        
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
        
        # 计算目标函数值（简化版）
        objective = np.sum(g) + 0.1 * np.sum(m[:, 1:]) + np.sum([tasks[j]['pi_j'] * u[j] for j in range(J)])
        
        return {
            'status': 'optimal',
            'x': x,
            'g': g,
            'u': u,
            'm': m,
            'objective': objective,
            'algorithm': 'multi_agent'
        }
