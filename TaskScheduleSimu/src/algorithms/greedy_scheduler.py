import numpy as np
from .base import BaseScheduler

class GreedyScheduler(BaseScheduler):
    """
    基于贪心策略的调度算法
    """
    def __init__(self, problem):
        """
        初始化贪心调度器
        
        参数:
        problem: 问题模型实例
        """
        super().__init__(problem)
    
    def solve(self):
        """
        使用贪心策略求解调度问题
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
        
        # 计算每个域和时间槽的清洁能源利用率
        energy_utilization = np.zeros((D, T))
        for d in range(D):
            for t in range(T):
                energy_utilization[d, t] = R_hat[d, t] / (E_base[d, t] + alpha[d] * (Cap[d, t] - Base[d, t]))
        
        # 按任务到达时间排序
        sorted_tasks = sorted(enumerate(tasks), key=lambda x: x[1]['a_j'])
        
        for j_idx, (j, task) in enumerate(sorted_tasks):
            a_j = task['a_j'] - 1
            d_j = task['d_j'] - 1
            W_j = task['W_j']
            S_j = task['S_j']
            src_j = task['src_j'] - 1  # 转换为0索引
            
            remaining_work = W_j
            current_domain = src_j
            
            # 为每个时间槽分配任务
            for t in range(a_j, d_j + 1):
                if remaining_work <= 0:
                    break
                
                # 找到当前时间槽中清洁能源利用率最高的域
                available_domains = []
                for d in range(D):
                    available_capacity = Cap[d, t] - Base[d, t] - np.sum(x[:, d, t])
                    if available_capacity > 0:
                        available_domains.append((-energy_utilization[d, t], d))  # 负号用于排序
                
                if not available_domains:
                    continue
                
                # 按清洁能源利用率排序，选择最高的
                available_domains.sort()
                best_domain = available_domains[0][1]
                
                # 计算可分配的执行量
                available_capacity = Cap[best_domain, t] - Base[best_domain, t] - np.sum(x[:, best_domain, t])
                assign_work = min(remaining_work, available_capacity)
                
                # 分配执行量
                x[j_idx, best_domain, t] = assign_work
                remaining_work -= assign_work
                
                # 计算能耗和棕电补能
                energy_consumption = E_base[best_domain, t] + alpha[best_domain] * np.sum(x[:, best_domain, t])
                if energy_consumption > R_hat[best_domain, t]:
                    g[best_domain, t] = energy_consumption - R_hat[best_domain, t]
                
                # 计算迁移数据量
                if t > a_j and best_domain != current_domain:
                    m[j_idx, t] = S_j
                    current_domain = best_domain
        
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
            'algorithm': 'greedy'
        }
