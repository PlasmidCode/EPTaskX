import numpy as np

class TestDataGenerator:
    """
    测试数据生成器
    """
    def __init__(self, seed=42):
        """
        初始化测试数据生成器
        
        参数:
        seed: 随机种子
        """
        np.random.seed(seed)
    
    def generate_problem(self, D=3, T=6, J=5):
        """
        生成测试问题数据
        
        参数:
        D: 域数量
        T: 时间槽数量
        J: 任务数量
        
        返回:
        问题参数字典
        """
        # 清洁能源发电量预测 [D, T]
        R_hat = np.random.uniform(80, 200, size=(D, T))
        
        # 域算力容量 [D, T]
        Cap = np.random.uniform(40, 80, size=(D, T))
        
        # 域背景负载 [D, T]
        Base = np.random.uniform(5, 20, size=(D, T))
        
        # 总迁移带宽预算 [T]
        BW_tot = np.random.uniform(30, 60, size=T)
        BW_tot[0] = 0  # 第0个时间槽无迁移
        
        # 任务列表
        tasks = []
        for j in range(J):
            a_j = np.random.randint(1, T-2)  # 到达时间槽
            d_j = np.random.randint(a_j+1, T+1)  # 截止期
            W_j = np.random.uniform(50, 150)  # 总计算需求
            S_j = np.random.uniform(10, 30)  # 状态大小
            src_j = np.random.randint(1, D+1)  # 初始域
            pi_j = np.random.uniform(0.5, 1.5)  # 任务权重
            
            tasks.append({
                'a_j': a_j,
                'd_j': d_j,
                'W_j': W_j,
                'S_j': S_j,
                'src_j': src_j,
                'pi_j': pi_j
            })
        
        # 能耗转换系数 [D]
        alpha = np.random.uniform(0.4, 0.7, size=D)  # kWh/计算单位
        
        # 域基础能耗 [D, T]
        E_base = np.random.uniform(5, 15, size=(D, T))
        
        return {
            'D': D,
            'T': T,
            'R_hat': R_hat,
            'Cap': Cap,
            'Base': Base,
            'BW_tot': BW_tot,
            'tasks': tasks,
            'alpha': alpha,
            'E_base': E_base
        }
    
    def generate_scenarios(self):
        """
        生成多种测试场景
        
        返回:
        场景列表
        """
        scenarios = []
        
        # 场景1: 小规模问题
        scenarios.append({
            'name': 'small_scale',
            'params': self.generate_problem(D=2, T=4, J=3)
        })
        
        # 场景2: 中等规模问题
        scenarios.append({
            'name': 'medium_scale',
            'params': self.generate_problem(D=3, T=6, J=5)
        })
        
        # 场景3: 大规模问题
        scenarios.append({
            'name': 'large_scale',
            'params': self.generate_problem(D=4, T=8, J=8)
        })
        
        # 场景4: 高清洁能源波动
        high_volatility = self.generate_problem(D=3, T=6, J=5)
        # 增加清洁能源的波动性
        high_volatility['R_hat'] = np.random.uniform(50, 250, size=(3, 6))
        scenarios.append({
            'name': 'high_volatility',
            'params': high_volatility
        })
        
        # 场景5: 高任务密度
        high_density = self.generate_problem(D=3, T=6, J=8)
        # 增加任务的计算需求
        for task in high_density['tasks']:
            task['W_j'] *= 1.5
        scenarios.append({
            'name': 'high_density',
            'params': high_density
        })
        
        return scenarios
