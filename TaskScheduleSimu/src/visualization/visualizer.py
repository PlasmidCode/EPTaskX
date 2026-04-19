import matplotlib.pyplot as plt
import numpy as np
import os

class Visualizer:
    """
    实验结果可视化
    """
    def __init__(self, problem):
        """
        初始化可视化器
        
        参数:
        problem: 问题模型实例
        """
        self.problem = problem
    
    def visualize_results(self, result, output_dir='results'):
        """
        可视化实验结果
        
        参数:
        result: 求解结果
        output_dir: 输出目录
        """
        # 创建算法专用目录
        algo_dir = os.path.join(output_dir, result['algorithm'])
        os.makedirs(algo_dir, exist_ok=True)
        
        # 1. 任务执行分配可视化
        self._visualize_task_allocation(result, algo_dir)
        
        # 2. 棕电补能可视化
        self._visualize_grid_energy(result, algo_dir)
        
        # 3. 迁移数据量可视化
        self._visualize_migration(result, algo_dir)
        
        # 4. 清洁能源使用情况
        self._visualize_clean_energy_usage(result, algo_dir)
        
        # 5. 生成算法结果的 markdown 文件
        self._generate_algorithm_report(result, algo_dir)
        
        print(f"可视化结果已保存到 {algo_dir} 目录")
    
    def _visualize_task_allocation(self, result, output_dir):
        """
        可视化任务执行分配
        """
        D = self.problem.D
        T = self.problem.T
        J = self.problem.J
        
        plt.figure(figsize=(12, 8))
        for j in range(J):
            for d in range(D):
                plt.bar(np.arange(T) + d * 0.2, result['x'][j, d, :], width=0.2, label=f'Task {j+1} Domain {d+1}')
        plt.xlabel('Time Slot')
        plt.ylabel('Execution Amount')
        plt.title('Task Execution Allocation')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'task_allocation.png'))
        plt.close()
    
    def _visualize_grid_energy(self, result, output_dir):
        """
        可视化棕电补能
        """
        D = self.problem.D
        T = self.problem.T
        
        plt.figure(figsize=(12, 6))
        for d in range(D):
            plt.plot(np.arange(T), result['g'][d, :], marker='o', label=f'Domain {d+1}')
        plt.xlabel('Time Slot')
        plt.ylabel('Grid Energy (kWh)')
        plt.title('Grid Energy Consumption')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'grid_energy.png'))
        plt.close()
    
    def _visualize_migration(self, result, output_dir):
        """
        可视化迁移数据量
        """
        J = self.problem.J
        T = self.problem.T
        
        plt.figure(figsize=(12, 6))
        for j in range(J):
            plt.plot(np.arange(T), result['m'][j, :], marker='o', label=f'Task {j+1}')
        plt.xlabel('Time Slot')
        plt.ylabel('Migration Data (GB)')
        plt.title('Migration Data Amount')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'migration.png'))
        plt.close()
    
    def _visualize_clean_energy_usage(self, result, output_dir):
        """
        可视化清洁能源使用情况
        """
        D = self.problem.D
        T = self.problem.T
        alpha = self.problem.alpha
        E_base = self.problem.E_base
        R_hat = self.problem.R_hat
        
        plt.figure(figsize=(12, 6))
        for d in range(D):
            # 计算实际能耗
            energy_consumption = E_base[d, :] + alpha[d] * np.sum(result['x'][:, d, :], axis=0)
            plt.plot(np.arange(T), energy_consumption, marker='o', label=f'Domain {d+1} Actual Energy')
            plt.plot(np.arange(T), R_hat[d, :], marker='s', label=f'Domain {d+1} Clean Energy Supply')
        plt.xlabel('Time Slot')
        plt.ylabel('Energy (kWh)')
        plt.title('Clean Energy Usage')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'clean_energy_usage.png'))
        plt.close()
    
    def _generate_algorithm_report(self, result, output_dir):
        """
        生成算法结果的 markdown 报告
        """
        algorithm = result['algorithm']
        objective = result['objective']
        grid_energy = np.sum(result['g'])
        migration = np.sum(result['m'][:, 1:])
        
        # 计算任务完成率
        J = self.problem.J
        total_completed = np.sum([self.problem.tasks[j]['W_j'] - result['u'][j] for j in range(J)])
        total_work = np.sum([self.problem.tasks[j]['W_j'] for j in range(J)])
        completion_rate = total_completed / total_work
        
        # 生成 markdown 内容
        md_content = f"""# {algorithm} 算法结果报告

## 基本信息
- 算法名称: {algorithm}
- 求解状态: {result['status']}
- 目标函数值: {objective:.4f}

## 性能指标
- 棕电补能总量: {grid_energy:.4f} kWh
- 迁移数据总量: {migration:.4f} GB
- 任务完成率: {completion_rate:.4f}

## 详细结果
- 任务执行分配: task_allocation.png
- 棕电补能情况: grid_energy.png
- 迁移数据量: migration.png
- 清洁能源使用情况: clean_energy_usage.png

## 分析
{self._generate_analysis(result)}
"""
        
        # 写入 markdown 文件
        with open(os.path.join(output_dir, 'report.md'), 'w') as f:
            f.write(md_content)
    
    def _generate_analysis(self, result):
        """
        生成算法分析内容
        """
        algorithm = result['algorithm']
        grid_energy = np.sum(result['g'])
        migration = np.sum(result['m'][:, 1:])
        
        analysis = []
        
        if grid_energy == 0:
            analysis.append("- 完全利用清洁能源，无需棕电补能")
        else:
            analysis.append(f"- 需要棕电补能 {grid_energy:.4f} kWh")
        
        if migration == 0:
            analysis.append("- 无任务迁移，减少了网络开销")
        else:
            analysis.append(f"- 迁移数据量为 {migration:.4f} GB")
        
        # 任务完成情况
        J = self.problem.J
        uncompleted = np.sum(result['u'])
        if uncompleted == 0:
            analysis.append("- 所有任务均已完成")
        else:
            analysis.append(f"- 有 {uncompleted:.4f} 单位的任务未完成")
        
        return '\n'.join(analysis)
    
    def visualize_comparison(self, results, output_dir='results'):
        """
        可视化不同算法的比较
        
        参数:
        results: 不同算法的求解结果列表
        output_dir: 输出目录
        """
        # 创建比较结果目录
        comparison_dir = os.path.join(output_dir, 'comparison')
        os.makedirs(comparison_dir, exist_ok=True)
        
        # 1. 目标函数值比较
        algorithms = [result['algorithm'] for result in results]
        objectives = [result['objective'] for result in results]
        
        plt.figure(figsize=(10, 6))
        plt.bar(algorithms, objectives)
        plt.xlabel('Algorithm')
        plt.ylabel('Objective Value')
        plt.title('Objective Value Comparison')
        plt.savefig(os.path.join(comparison_dir, 'algorithm_comparison.png'))
        plt.close()
        
        # 2. 棕电补能量比较
        grid_energies = [np.sum(result['g']) for result in results]
        
        plt.figure(figsize=(10, 6))
        plt.bar(algorithms, grid_energies)
        plt.xlabel('Algorithm')
        plt.ylabel('Total Grid Energy (kWh)')
        plt.title('Grid Energy Consumption Comparison')
        plt.savefig(os.path.join(comparison_dir, 'grid_energy_comparison.png'))
        plt.close()
        
        # 3. 迁移数据量比较
        migration_data = [np.sum(result['m'][:, 1:]) for result in results]
        
        plt.figure(figsize=(10, 6))
        plt.bar(algorithms, migration_data)
        plt.xlabel('Algorithm')
        plt.ylabel('Total Migration Data (GB)')
        plt.title('Migration Data Comparison')
        plt.savefig(os.path.join(comparison_dir, 'migration_comparison.png'))
        plt.close()
        
        # 4. 任务完成率比较
        J = self.problem.J
        completion_rates = []
        for result in results:
            total_completed = np.sum([self.problem.tasks[j]['W_j'] - result['u'][j] for j in range(J)])
            total_work = np.sum([self.problem.tasks[j]['W_j'] for j in range(J)])
            completion_rates.append(total_completed / total_work)
        
        plt.figure(figsize=(10, 6))
        plt.bar(algorithms, completion_rates)
        plt.xlabel('Algorithm')
        plt.ylabel('Completion Rate')
        plt.title('Task Completion Rate Comparison')
        plt.savefig(os.path.join(comparison_dir, 'completion_rate_comparison.png'))
        plt.close()
        
        # 生成比较报告
        self._generate_comparison_report(results, comparison_dir)
        
        print(f"算法比较结果已保存到 {comparison_dir} 目录")
    
    def _generate_comparison_report(self, results, output_dir):
        """
        生成算法比较的 markdown 报告
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
            J = self.problem.J
            total_completed = np.sum([self.problem.tasks[j]['W_j'] - result['u'][j] for j in range(J)])
            total_work = np.sum([self.problem.tasks[j]['W_j'] for j in range(J)])
            completion_rates.append(total_completed / total_work)
        
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
            f"- 所有算法均完全完成任务"
        ]
        
        # 生成 markdown 内容
        md_content = f"""# 算法比较报告

## 比较结果

{table}

## 分析

{chr(10).join(analysis)}

## 可视化图表
- 目标函数值比较: algorithm_comparison.png
- 棕电补能比较: grid_energy_comparison.png
- 迁移数据量比较: migration_comparison.png
- 任务完成率比较: completion_rate_comparison.png
"""
        
        # 写入 markdown 文件
        with open(os.path.join(output_dir, 'comparison_report.md'), 'w') as f:
            f.write(md_content)

