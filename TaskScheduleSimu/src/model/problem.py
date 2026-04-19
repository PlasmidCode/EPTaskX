import numpy as np

class ProblemModel:
    """
    问题模型定义
    """
    def __init__(self, D, T, R_hat, Cap, Base, BW_tot, tasks, alpha, E_base):
        """
        初始化问题模型
        
        参数:
        D: 域数量
        T: 时间槽数量
        R_hat: 清洁能源发电量预测 [D, T]
        Cap: 域算力容量 [D, T]
        Base: 域背景负载 [D, T]
        BW_tot: 总迁移带宽预算 [T]
        tasks: 任务列表，每个任务包含 {a_j, d_j, W_j, S_j, src_j, pi_j}
        alpha: 能耗转换系数 [D]
        E_base: 域基础能耗 [D, T]
        """
        self.D = D
        self.T = T
        self.R_hat = R_hat
        self.Cap = Cap
        self.Base = Base
        self.BW_tot = BW_tot
        self.tasks = tasks
        self.alpha = alpha
        self.E_base = E_base
        self.J = len(tasks)
    
    def get_task_info(self, j):
        """
        获取任务信息
        """
        return self.tasks[j]
    
    def get_domain_info(self, d):
        """
        获取域信息
        """
        return {
            'Cap': self.Cap[d],
            'Base': self.Base[d],
            'alpha': self.alpha[d],
            'E_base': self.E_base[d],
            'R_hat': self.R_hat[d]
        }
