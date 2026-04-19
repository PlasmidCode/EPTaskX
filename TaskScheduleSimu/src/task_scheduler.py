import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import os

class TaskScheduler:
    def __init__(self, D, T, R_hat, Cap, Base, BW_tot, tasks, alpha, E_base, beta=1.0, lambda1=0.1, lambda2=1.0):
        """
        初始化任务调度器
        
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
        beta: 迁移惩罚系数
        lambda1: 迁移惩罚权重
        lambda2: 违约惩罚权重
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
        self.beta = beta
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.J = len(tasks)
        
    def solve(self):
        """
        求解优化问题
        """
        # 决策变量
        x = cp.Variable((self.J, self.D, self.T), nonneg=True)  # 执行分配量
        g = cp.Variable((self.D, self.T), nonneg=True)  # 棕电补能量
        u = cp.Variable(self.J, nonneg=True)  # 违约未完成量
        m = cp.Variable((self.J, self.T), nonneg=True)  # 迁移数据量
        
        # 目标函数
        objective = cp.Minimize(
            cp.sum(g) + 
            self.lambda1 * self.beta * cp.sum(m[:, 1:]) + 
            self.lambda2 * cp.sum(cp.multiply([task['pi_j'] for task in self.tasks], u))
        )
        
        # 约束条件
        constraints = []
        
        # 1. 任务完成约束
        for j in range(self.J):
            task = self.tasks[j]
            a_j = task['a_j'] - 1  # 转换为0索引
            d_j = task['d_j'] - 1
            constraints.append(cp.sum(x[j, :, a_j:d_j+1]) + u[j] == task['W_j'])
        
        # 2. 域容量约束
        for d in range(self.D):
            for t in range(self.T):
                constraints.append(cp.sum(x[:, d, t]) <= self.Cap[d, t] - self.Base[d, t])
        
        # 3. 能耗定义与供能平衡
        for d in range(self.D):
            for t in range(self.T):
                E_dt = self.E_base[d, t] + self.alpha[d] * cp.sum(x[:, d, t])
                constraints.append(E_dt <= self.R_hat[d, t] + g[d, t])
        
        # 4. 迁移数据量约束
        for j in range(self.J):
            task = self.tasks[j]
            a_j = task['a_j'] - 1
            S_j = task['S_j']
            W_j = task['W_j']
            
            # 累计执行量
            cum_x = cp.cumsum(x[j, :, a_j:], axis=1)  # [D, T - a_j]
            
            # 计算相邻时间槽的执行分布变化
            for t in range(1, self.T - a_j):
                delta = cum_x[:, t] - cum_x[:, t-1]
                abs_delta = cp.abs(delta)
                constraints.append(m[j, a_j + t] >= (S_j / W_j) * 0.5 * cp.sum(abs_delta))
        
        # 5. 带宽约束
        for t in range(1, self.T):
            constraints.append(cp.sum(m[:, t]) <= self.BW_tot[t])
        
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
                'objective': problem.value
            }
        else:
            return {'status': problem.status, 'objective': None}
    
    def visualize_results(self, result, output_dir='results'):
        """
        可视化实验结果
        
        参数:
        result: 求解结果
        output_dir: 输出目录
        """
        if result['status'] not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            print("无法可视化：求解未成功")
            return
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 任务执行分配可视化
        plt.figure(figsize=(12, 8))
        for j in range(self.J):
            for d in range(self.D):
                plt.bar(np.arange(self.T) + d * 0.2, result['x'][j, d, :], width=0.2, label=f'任务{j+1} 域{d+1}')
        plt.xlabel('时间槽')
        plt.ylabel('执行量')
        plt.title('任务执行分配')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'task_allocation.png'))
        plt.close()
        
        # 2. 棕电补能可视化
        plt.figure(figsize=(12, 6))
        for d in range(self.D):
            plt.plot(np.arange(self.T), result['g'][d, :], marker='o', label=f'域{d+1}')
        plt.xlabel('时间槽')
        plt.ylabel('棕电补能量 (kWh)')
        plt.title('棕电补能情况')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'grid_energy.png'))
        plt.close()
        
        # 3. 迁移数据量可视化
        plt.figure(figsize=(12, 6))
        for j in range(self.J):
            plt.plot(np.arange(self.T), result['m'][j, :], marker='o', label=f'任务{j+1}')
        plt.xlabel('时间槽')
        plt.ylabel('迁移数据量 (GB)')
        plt.title('迁移数据量')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'migration.png'))
        plt.close()
        
        # 4. 清洁能源使用情况
        plt.figure(figsize=(12, 6))
        for d in range(self.D):
            # 计算实际能耗
            energy_consumption = self.E_base[d, :] + self.alpha[d] * np.sum(result['x'][:, d, :], axis=0)
            plt.plot(np.arange(self.T), energy_consumption, marker='o', label=f'域{d+1} 实际能耗')
            plt.plot(np.arange(self.T), self.R_hat[d, :], marker='s', label=f'域{d+1} 清洁能源供给')
        plt.xlabel('时间槽')
        plt.ylabel('能量 (kWh)')
        plt.title('清洁能源使用情况')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'clean_energy_usage.png'))
        plt.close()
        
        print(f"可视化结果已保存到 {output_dir} 目录")

# 示例代码
if __name__ == '__main__':
    # 示例参数
    D = 2  # 2个域
    T = 4  # 4个时间槽
    
    # 清洁能源发电量预测 [D, T]
    R_hat = np.array([
        [100, 150, 80, 120],  # 域1
        [120, 90, 140, 100]   # 域2
    ])
    
    # 域算力容量 [D, T]
    Cap = np.array([
        [50, 50, 50, 50],  # 域1
        [60, 60, 60, 60]   # 域2
    ])
    
    # 域背景负载 [D, T]
    Base = np.array([
        [10, 10, 10, 10],  # 域1
        [15, 15, 15, 15]   # 域2
    ])
    
    # 总迁移带宽预算 [T]
    BW_tot = np.array([0, 50, 50, 50])  # 第0个时间槽无迁移
    
    # 任务列表
    tasks = [
        {
            'a_j': 1,  # 到达时间槽
            'd_j': 4,  # 截止期
            'W_j': 100,  # 总计算需求
            'S_j': 20,  # 状态大小
            'src_j': 1,  # 初始域
            'pi_j': 1.0  # 任务权重
        },
        {
            'a_j': 2,
            'd_j': 4,
            'W_j': 80,
            'S_j': 15,
            'src_j': 2,
            'pi_j': 1.0
        }
    ]
    
    # 能耗转换系数 [D]
    alpha = np.array([0.5, 0.6])  # kWh/计算单位
    
    # 域基础能耗 [D, T]
    E_base = np.array([
        [10, 10, 10, 10],  # 域1
        [12, 12, 12, 12]   # 域2
    ])
    
    # 创建调度器
    scheduler = TaskScheduler(D, T, R_hat, Cap, Base, BW_tot, tasks, alpha, E_base)
    
    # 求解
    result = scheduler.solve()
    
    # 输出结果
    print(f"求解状态: {result['status']}")
    print(f"目标函数值: {result['objective']}")
    
    if result['status'] in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        print("\n执行分配量 x [任务, 域, 时间槽]:")
        print(result['x'])
        
        print("\n棕电补能量 g [域, 时间槽]:")
        print(result['g'])
        
        print("\n任务违约未完成量 u:")
        print(result['u'])
        
        print("\n迁移数据量 m [任务, 时间槽]:")
        print(result['m'])
        
        # 可视化结果
        scheduler.visualize_results(result)
