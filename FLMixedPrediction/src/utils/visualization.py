import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


def plot_training_history(history: List[Dict[str, float]], save_path: str = None) -> None:
    """
    绘制训练历史曲线
    
    Args:
        history: 训练历史记录，包含每轮的损失和指标
        save_path: 保存路径，若为None则不保存
    """
    rounds = [entry['round'] for entry in history]
    losses = [entry['loss'] for entry in history]
    maes = [entry['mae'] for entry in history]
    rmses = [entry['rmse'] for entry in history]
    
    plt.figure(figsize=(15, 5))
    
    # 损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(rounds, losses, 'b-', label='Loss')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.legend()
    
    # MAE曲线
    plt.subplot(1, 3, 2)
    plt.plot(rounds, maes, 'r-', label='MAE')
    plt.xlabel('Round')
    plt.ylabel('MAE')
    plt.title('Mean Absolute Error')
    plt.grid(True)
    plt.legend()
    
    # RMSE曲线
    plt.subplot(1, 3, 3)
    plt.plot(rounds, rmses, 'g-', label='RMSE')
    plt.xlabel('Round')
    plt.ylabel('RMSE')
    plt.title('Root Mean Squared Error')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_multi_horizon_results(results: Dict[str, Dict[str, float]], metrics: List[str] = None, save_path: str = None) -> None:
    """
    绘制多步预测结果对比
    
    Args:
        results: 多步预测结果，键为步长，值为指标字典
        metrics: 要绘制的指标列表，默认使用所有指标
        save_path: 保存路径，若为None则不保存
    """
    if metrics is None:
        metrics = list(next(iter(results.values())).keys())
    
    horizons = list(results.keys())
    num_metrics = len(metrics)
    
    plt.figure(figsize=(15, 5 * num_metrics))
    
    for i, metric in enumerate(metrics):
        plt.subplot(num_metrics, 1, i + 1)
        
        metric_values = [results[horizon][metric] for horizon in horizons]
        
        plt.bar(horizons, metric_values, color='skyblue')
        plt.xlabel('Horizon')
        plt.ylabel(metric.upper())
        plt.title(f'{metric.upper()} Across Different Horizons')
        
        # 添加数值标签
        for j, v in enumerate(metric_values):
            plt.text(j, v, f'{v:.4f}', ha='center', va='bottom')
        
        plt.grid(True, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_expert_usage(expert_usage: Dict[str, float], save_path: str = None) -> None:
    """
    绘制专家使用比例
    
    Args:
        expert_usage: 专家使用比例字典，键为专家ID，值为使用比例
        save_path: 保存路径，若为None则不保存
    """
    experts = list(expert_usage.keys())
    usage = list(expert_usage.values())
    
    plt.figure(figsize=(12, 6))
    
    plt.pie(usage, labels=experts, autopct='%1.1f%%', startangle=90, 
            colors=plt.cm.Set3(np.linspace(0, 1, len(experts))))
    plt.title('Expert Usage Distribution')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, horizon: int = 96, sample_idx: int = 0, save_path: str = None) -> None:
    """
    绘制预测结果与真实值对比
    
    Args:
        y_true: 真实值，形状为 [样本数, 预测步长]
        y_pred: 预测值，形状为 [样本数, 预测步长]
        horizon: 预测步长
        sample_idx: 要绘制的样本索引
        save_path: 保存路径，若为None则不保存
    """
    plt.figure(figsize=(15, 6))
    
    # 时间步
    time_steps = np.arange(horizon)
    
    # 绘制真实值和预测值
    plt.plot(time_steps, y_true[sample_idx, :horizon], 'b-', label='True Value')
    plt.plot(time_steps, y_pred[sample_idx, :horizon], 'r--', label='Predicted Value')
    
    plt.xlabel('Time Step')
    plt.ylabel('Power')
    plt.title(f'Prediction vs True Value (Horizon: {horizon} steps)')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_rmse_per_step(rmse_values: List[float], save_path: str = None) -> None:
    """
    绘制每个预测步长的RMSE值
    
    Args:
        rmse_values: 每个步长的RMSE值列表
        save_path: 保存路径，若为None则不保存
    """
    plt.figure(figsize=(15, 6))
    
    time_steps = np.arange(1, len(rmse_values) + 1)
    
    plt.plot(time_steps, rmse_values, 'b-', marker='o')
    plt.xlabel('Prediction Step')
    plt.ylabel('RMSE')
    plt.title('RMSE Per Prediction Step')
    plt.grid(True)
    
    # 添加数值标签
    for i, v in enumerate(rmse_values):
        plt.text(i + 1, v, f'{v:.4f}', ha='center', va='bottom')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()
