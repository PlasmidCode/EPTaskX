import cvxpy as cp
import numpy as np
from .base import BaseScheduler

class OptimizationScheduler(BaseScheduler):
    """
    基于优化的调度算法
    """
    def __init__(self, problem, beta=1.0, lambda1=0.1, lambda2=1.0):
        """
        初始化优化调度器
        
        参数:
        problem: 问题模型实例
        beta: 迁移惩罚系数
        lambda1: 迁移惩罚权重
        lambda2: 违约惩罚权重
        """
        super().__init__(problem)
        self.beta = beta
        self.lambda1 = lambda1
        self.lambda2 = lambda2
    
    def solve(self):
        """
        求解优化问题
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
        
        # 决策变量
        x = cp.Variable((J, D, T), nonneg=True)  # 执行分配量
        g = cp.Variable((D, T), nonneg=True)  # 棕电补能量
        u = cp.Variable(J, nonneg=True)  # 违约未完成量
        m = cp.Variable((J, T), nonneg=True)  # 迁移数据量
        
        # 目标函数
        objective = cp.Minimize(
            cp.sum(g) + 
            self.lambda1 * self.beta * cp.sum(m[:, 1:]) + 
            self.lambda2 * cp.sum(cp.multiply([task['pi_j'] for task in tasks], u))
        )
        
        # 约束条件
        constraints = []
        
        # 1. 任务完成约束
        for j in range(J):
            task = tasks[j]
            a_j = task['a_j'] - 1  # 转换为0索引
            d_j = task['d_j'] - 1
            constraints.append(cp.sum(x[j, :, a_j:d_j+1]) + u[j] == task['W_j'])
        
        # 2. 域容量约束
        for d in range(D):
            for t in range(T):
                constraints.append(cp.sum(x[:, d, t]) <= Cap[d, t] - Base[d, t])
        
        # 3. 能耗定义与供能平衡
        for d in range(D):
            for t in range(T):
                E_dt = E_base[d, t] + alpha[d] * cp.sum(x[:, d, t])
                constraints.append(E_dt <= R_hat[d, t] + g[d, t])
        
        # 4. 迁移数据量约束
        for j in range(J):
            task = tasks[j]
            a_j = task['a_j'] - 1
            S_j = task['S_j']
            W_j = task['W_j']
            
            # 累计执行量
            cum_x = cp.cumsum(x[j, :, a_j:], axis=1)  # [D, T - a_j]
            
            # 计算相邻时间槽的执行分布变化
            for t in range(1, T - a_j):
                delta = cum_x[:, t] - cum_x[:, t-1]
                abs_delta = cp.abs(delta)
                constraints.append(m[j, a_j + t] >= (S_j / W_j) * 0.5 * cp.sum(abs_delta))
        
        # 5. 带宽约束
        for t in range(1, T):
            constraints.append(cp.sum(m[:, t]) <= BW_tot[t])
        
        # 求解
        problem = cp.Problem(objective, constraints)
        problem.solve()  # 使用默认求解器
        
        # 提取结果
        if problem.status == cp.OPTIMAL or problem.status == cp.OPTIMAL_INACCURATE:
            return {
                'status': problem.status,
                'x': x.value,
                'g': g.value,
                'u': u.value,
                'm': m.value,
                'objective': problem.value,
                'algorithm': 'optimization'
            }
        else:
            return {'status': problem.status, 'objective': None, 'algorithm': 'optimization'}
