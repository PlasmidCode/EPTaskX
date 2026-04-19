import numpy as np
from typing import Dict, List, Tuple


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算平均绝对误差
    
    Args:
        y_true: 真实值，形状为 [样本数, 预测步长]
        y_pred: 预测值，形状为 [样本数, 预测步长]
        
    Returns:
        MAE值
    """
    return np.mean(np.abs(y_true - y_pred))


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算均方根误差
    
    Args:
        y_true: 真实值，形状为 [样本数, 预测步长]
        y_pred: 预测值，形状为 [样本数, 预测步长]
        
    Returns:
        RMSE值
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def calculate_nrmse(y_true: np.ndarray, y_pred: np.ndarray, norm_type: str = "range") -> float:
    """
    计算归一化均方根误差
    
    Args:
        y_true: 真实值，形状为 [样本数, 预测步长]
        y_pred: 预测值，形状为 [样本数, 预测步长]
        norm_type: 归一化类型，可选 "range", "mean", "std"
        
    Returns:
        nRMSE值
    """
    rmse = calculate_rmse(y_true, y_pred)
    
    if norm_type == "range":
        # 基于数据范围归一化
        y_min = np.min(y_true)
        y_max = np.max(y_true)
        norm_factor = y_max - y_min if y_max != y_min else 1.0
    elif norm_type == "mean":
        # 基于均值归一化
        y_mean = np.mean(y_true)
        norm_factor = y_mean if y_mean != 0 else 1.0
    elif norm_type == "std":
        # 基于标准差归一化
        y_std = np.std(y_true)
        norm_factor = y_std if y_std != 0 else 1.0
    else:
        raise ValueError(f"Unsupported norm_type: {norm_type}")
    
    return rmse / norm_factor


def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算对称平均绝对百分比误差
    
    Args:
        y_true: 真实值，形状为 [样本数, 预测步长]
        y_pred: 预测值，形状为 [样本数, 预测步长]
        
    Returns:
        sMAPE值
    """
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    
    # 避免除以零
    denominator = np.where(denominator == 0, 1.0, denominator)
    
    return np.mean(numerator / denominator) * 100


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, metrics: List[str] = None, 
                   norm_type: str = "range") -> Dict[str, float]:
    """
    评估模型性能
    
    Args:
        y_true: 真实值，形状为 [样本数, 预测步长]
        y_pred: 预测值，形状为 [样本数, 预测步长]
        metrics: 要计算的指标列表，默认计算所有指标
        norm_type: nRMSE的归一化类型
        
    Returns:
        指标字典
    """
    if metrics is None:
        metrics = ["mae", "rmse", "nrmse"]
    
    results = {}
    
    if "mae" in metrics:
        results["mae"] = calculate_mae(y_true, y_pred)
    
    if "rmse" in metrics:
        results["rmse"] = calculate_rmse(y_true, y_pred)
    
    if "nrmse" in metrics:
        results["nrmse"] = calculate_nrmse(y_true, y_pred, norm_type)
    
    if "smape" in metrics:
        results["smape"] = calculate_smape(y_true, y_pred)
    
    return results


def evaluate_multi_horizon(y_true: np.ndarray, y_pred: np.ndarray, horizons: List[int],
                           metrics: List[str] = None, norm_type: str = "range") -> Dict[str, Dict[str, float]]:
    """
    评估多步预测性能
    
    Args:
        y_true: 真实值，形状为 [样本数, 最大预测步长]
        y_pred: 预测值，形状为 [样本数, 最大预测步长]
        horizons: 预测步长列表，例如 [4, 24, 96]
        metrics: 要计算的指标列表
        norm_type: nRMSE的归一化类型
        
    Returns:
        指标字典，键为预测步长，值为指标结果
    """
    results = {}
    
    for horizon in horizons:
        # 获取对应步长的预测结果
        y_true_horizon = y_true[:, :horizon]
        y_pred_horizon = y_pred[:, :horizon]
        
        # 计算指标
        horizon_results = evaluate_model(y_true_horizon, y_pred_horizon, metrics, norm_type)
        results[f"{horizon}_steps"] = horizon_results
    
    return results


def evaluate_per_horizon_step(y_true: np.ndarray, y_pred: np.ndarray, metrics: List[str] = None,
                              norm_type: str = "range") -> Dict[str, List[float]]:
    """
    评估每个预测步长的性能
    
    Args:
        y_true: 真实值，形状为 [样本数, 预测步长]
        y_pred: 预测值，形状为 [样本数, 预测步长]
        metrics: 要计算的指标列表
        norm_type: nRMSE的归一化类型
        
    Returns:
        指标字典，键为指标名称，值为每个步长的指标值列表
    """
    if metrics is None:
        metrics = ["mae", "rmse", "nrmse"]
    
    results = {metric: [] for metric in metrics}
    num_steps = y_true.shape[1]
    
    for step in range(num_steps):
        y_true_step = y_true[:, step]
        y_pred_step = y_pred[:, step]
        
        # 扩展维度以适应现有函数
        y_true_step = y_true_step[:, np.newaxis]
        y_pred_step = y_pred_step[:, np.newaxis]
        
        step_results = evaluate_model(y_true_step, y_pred_step, metrics, norm_type)
        
        for metric in metrics:
            results[metric].append(step_results[metric])
    
    return results


def calculate_metrics_by_horizon(y_true: np.ndarray, y_pred: np.ndarray, 
                                 horizons: List[int] = [4, 24, 96]) -> Dict[str, Dict[str, float]]:
    """
    按预测范围计算指标
    
    Args:
        y_true: 真实值，形状为 [样本数, 最大预测步长]
        y_pred: 预测值，形状为 [样本数, 最大预测步长]
        horizons: 预测步长列表，例如 [4, 24, 96] 对应 1h, 6h, 24h
        
    Returns:
        指标字典，键为时间范围，值为指标结果
    """
    horizon_names = {
        4: "1h",
        24: "6h",
        96: "24h"
    }
    
    results = {}
    
    for horizon in horizons:
        name = horizon_names.get(horizon, f"{horizon}steps")
        results[name] = evaluate_multi_horizon(y_true, y_pred, [horizon])[f"{horizon}_steps"]
    
    return results
