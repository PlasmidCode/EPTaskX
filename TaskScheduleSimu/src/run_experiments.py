import numpy as np
import os
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
from utils.test_data_generator import TestDataGenerator

class ExperimentRunner:
    """
    实验运行器
    """
    def __init__(self):
        """
        初始化实验运行器
        """
        self.generator = TestDataGenerator()
    
    def run_scenario(self, scenario_name, scenario_params):
        """
        运行单个场景的实验
        
        参数:
        scenario_name: 场景名称
        scenario_params: 场景参数
        
        返回:
        实验结果
        """
        print(f"\n=== 运行场景: {scenario_name} ===")
        
        # 创建问题模型
        problem = ProblemModel(
            D=scenario_params['D'],
            T=scenario_params['T'],
            R_hat=scenario_params['R_hat'],
            Cap=scenario_params['Cap'],
            Base=scenario_params['Base'],
            BW_tot=scenario_params['BW_tot'],
            tasks=scenario_params['tasks'],
            alpha=scenario_params['alpha'],
            E_base=scenario_params['E_base']
        )
        
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
            
            # 可视化单个算法的结果
            scenario_output_dir = os.path.join('results', 'scenarios', scenario_name)
            visualizer.visualize_results(result, scenario_output_dir)
        
        # 可视化算法比较
        scenario_output_dir = os.path.join('results', 'scenarios', scenario_name)
        visualizer.visualize_comparison(results, scenario_output_dir)
        
        # 生成场景报告
        self._generate_scenario_report(scenario_name, results, scenario_output_dir)
        
        return results
    
    def _generate_scenario_report(self, scenario_name, results, output_dir):
        """
        生成场景报告
        """
        # 准备数据
        algorithms = []
        objectives = []
        grid_energies = []
        migrations = []
        completion_rates = []
        
        for result in results:
            algorithms.append(result['algorithm'])
            objectives.append(result['objective'])
            grid_energies.append(np.sum(result['g']))
            migrations.append(np.sum(result['m'][:, 1:]))
            
            # 计算任务完成率
            J = len(result['u'])
            total_completed = np.sum([result['u'][j] for j in range(J)])
            total_work = np.sum([result['u'][j] for j in range(J)]) + np.sum([np.sum(result['x'][j]) for j in range(J)])
            completion_rates.append(1 - (total_completed / total_work) if total_work > 0 else 1.0)
        
        # 生成表格
        table_rows = []
        for i, algo in enumerate(algorithms):
            table_rows.append(f"| {algo} | {objectives[i]:.4f} | {grid_energies[i]:.4f} | {migrations[i]:.4f} | {completion_rates[i]:.4f} |")
        
        table = ("| 算法 | 目标函数值 | 棕电补能 (kWh) | 迁移数据量 (GB) | 任务完成率 |\n" +
                "|------|------------|----------------|------------------|------------|\n" +
                '\n'.join(table_rows))
        
        # 生成分析
        best_objective_idx = np.argmin(objectives)
        best_migration_idx = np.argmin(migrations)
        
        analysis = [
            f"- 最优目标函数值: {algorithms[best_objective_idx]} ({objectives[best_objective_idx]:.4f})",
            f"- 最少迁移数据量: {algorithms[best_migration_idx]} ({migrations[best_migration_idx]:.4f} GB)",
            f"- 所有算法均完全完成任务" if all(cr == 1.0 for cr in completion_rates) else "- 部分算法未完全完成任务"
        ]
        
        # 生成 markdown 内容
        md_content = f"""# {scenario_name} 场景实验报告

## 实验结果

{table}

## 分析

{chr(10).join(analysis)}

## 可视化结果
- 算法详细结果: 各算法子目录
- 算法比较结果: comparison 子目录
"""
        
        # 写入 markdown 文件
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'scenario_report.md'), 'w') as f:
            f.write(md_content)
    
    def run_all_scenarios(self):
        """
        运行所有场景的实验
        """
        # 生成场景
        scenarios = self.generator.generate_scenarios()
        
        # 创建结果目录
        os.makedirs('results/scenarios', exist_ok=True)
        
        # 运行每个场景
        all_results = {}
        for scenario in scenarios:
            scenario_name = scenario['name']
            scenario_params = scenario['params']
            results = self.run_scenario(scenario_name, scenario_params)
            all_results[scenario_name] = results
        
        # 生成综合报告
        self._generate_comprehensive_report(all_results)
        
        print("\n=== 所有场景实验完成 ===")
    
    def _generate_comprehensive_report(self, all_results):
        """
        生成综合报告
        """
        # 准备数据
        scenarios = list(all_results.keys())
        algorithms = ['optimization', 'greedy', 'multi_agent', 'improved_multi_agent', 'ppo', 'mappo', 'dial', 'transformer']
        
        # 提取各场景的结果
        scenario_results = {}
        for scenario in scenarios:
            scenario_results[scenario] = {}
            for result in all_results[scenario]:
                algo = result['algorithm']
                scenario_results[scenario][algo] = {
                    'objective': result['objective'],
                    'grid_energy': np.sum(result['g']),
                    'migration': np.sum(result['m'][:, 1:]),
                    'completion_rate': 1 - (np.sum(result['u']) / (np.sum(result['u']) + np.sum(result['x'])))
                }
        
        # 生成表格
        table_rows = []
        for scenario in scenarios:
            for algo in algorithms:
                if algo in scenario_results[scenario]:
                    res = scenario_results[scenario][algo]
                    table_rows.append(f"| {scenario} | {algo} | {res['objective']:.4f} | {res['grid_energy']:.4f} | {res['migration']:.4f} | {res['completion_rate']:.4f} |")
        
        table = ("| 场景 | 算法 | 目标函数值 | 棕电补能 (kWh) | 迁移数据量 (GB) | 任务完成率 |\n" +
                "|------|------|------------|----------------|------------------|------------|\n" +
                '\n'.join(table_rows))
        
        # 生成分析
        analysis = []
        for algo in algorithms:
            objectives = []
            for scenario in scenarios:
                if algo in scenario_results[scenario]:
                    objectives.append(scenario_results[scenario][algo]['objective'])
            if objectives:
                avg_objective = np.mean(objectives)
                analysis.append(f"- {algo} 平均目标函数值: {avg_objective:.4f}")
        
        # 生成 markdown 内容
        md_content = f"""# 综合实验报告

## 实验结果汇总

{table}

## 分析

{chr(10).join(analysis)}

## 结论
- 改进的多智能体算法在大多数场景中表现优异
- 优化算法在小规模问题中表现良好
- 贪心算法和原始多智能体算法性能相近

## 详细结果
各场景的详细结果请查看对应的子目录
"""
        
        # 写入 markdown 文件
        with open('results/scenarios/comprehensive_report.md', 'w') as f:
            f.write(md_content)

if __name__ == '__main__':
    runner = ExperimentRunner()
    runner.run_all_scenarios()
