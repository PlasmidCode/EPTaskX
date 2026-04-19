import numpy as np
from model.problem import ProblemModel
from algorithms.optimization_scheduler import OptimizationScheduler
from algorithms.greedy_scheduler import GreedyScheduler
from algorithms.multi_agent_scheduler import MultiAgentScheduler
from algorithms.improved_multi_agent_scheduler import ImprovedMultiAgentScheduler
from algorithms.ppo_scheduler import PPOScheduler
from algorithms.mappo_scheduler import MAPPOScheduler
from algorithms.dial_scheduler import DIALScheduler
from algorithms.transformer_scheduler import TransformerScheduler
from visualization.visualizer import Visualizer

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
    
    # 创建问题模型
    problem = ProblemModel(D, T, R_hat, Cap, Base, BW_tot, tasks, alpha, E_base)
    
    # 创建可视化器
    visualizer = Visualizer(problem)
    
    # 初始化不同的调度算法
        schedulers = [
            OptimizationScheduler(problem),
            GreedyScheduler(problem),
            MultiAgentScheduler(problem),
            ImprovedMultiAgentScheduler(problem),
            PPOScheduler(problem),
            MAPPOScheduler(problem),
            DIALScheduler(problem),
            TransformerScheduler(problem)
        ]
    
    # 运行所有算法并收集结果
    results = []
    for scheduler in schedulers:
        print(f"运行算法: {scheduler.__class__.__name__}")
        result = scheduler.solve()
        results.append(result)
        print(f"求解状态: {result['status']}")
        print(f"目标函数值: {result['objective']}")
        print()
        
        # 可视化单个算法的结果
        visualizer.visualize_results(result)
    
    # 可视化算法比较
    visualizer.visualize_comparison(results)
    
    # 输出算法比较结果
    print("算法比较结果:")
    print("-" * 60)
    print(f"{'算法':<20} {'目标函数值':<15} {'棕电补能':<15} {'迁移数据量':<15} {'任务完成率':<15}")
    print("-" * 60)
    
    for result in results:
        algorithm = result['algorithm']
        objective = result['objective']
        grid_energy = np.sum(result['g'])
        migration = np.sum(result['m'][:, 1:])
        
        # 计算任务完成率
        total_completed = np.sum([tasks[j]['W_j'] - result['u'][j] for j in range(len(tasks))])
        total_work = np.sum([tasks[j]['W_j'] for j in range(len(tasks))])
        completion_rate = total_completed / total_work
        
        print(f"{algorithm:<20} {objective:<15.4f} {grid_energy:<15.4f} {migration:<15.4f} {completion_rate:<15.4f}")
    
    print("-" * 60)
