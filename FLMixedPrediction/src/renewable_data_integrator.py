import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import torch
from torch.utils.data import DataLoader

from .data_processor import RenewableEnergyDataProcessor
from .utils import SeqDS, split_train_val


def load_renewable_data(data_dir: str, data_type: str = 'all', seq_len: int = 24, horizon: int = 1) -> Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]]:
    """
    加载可再生能源数据
    
    Args:
        data_dir: 数据目录路径
        data_type: 数据类型，'wind'、'solar'或'all'
        seq_len: 序列长度
        horizon: 预测步长
        
    Returns:
        站点数据字典，键为站点名称，值为(X, y, feature_cols)
    """
    processor = RenewableEnergyDataProcessor(data_dir)
    
    all_data = {}
    
    # 加载风场数据
    if data_type in ['wind', 'all']:
        wind_data = processor.get_all_sites_data('wind', seq_len, horizon)
        all_data.update(wind_data)
    
    # 加载光伏电站数据
    if data_type in ['solar', 'all']:
        solar_data = processor.get_all_sites_data('solar', seq_len, horizon)
        all_data.update(solar_data)
    
    return all_data


def build_renewable_loaders(X: np.ndarray, y: np.ndarray, batch: int) -> Tuple[DataLoader, DataLoader]:
    """
    构建数据加载器
    
    Args:
        X: 特征数据
        y: 目标数据
        batch: 批次大小
        
    Returns:
        训练数据加载器和验证数据加载器
    """
    if len(y) < 100:
        raise ValueError('Too few samples after windowing; reduce --seq or use more data.')
    
    (Xtr, ytr), (Xva, yva) = split_train_val(X, y, 0.2)
    
    return (
        DataLoader(SeqDS(Xtr, ytr), batch_size=batch, shuffle=True),
        DataLoader(SeqDS(Xva, yva), batch_size=batch, shuffle=False)
    )


def get_all_sites_dataframes(data_dir: str, data_type: str = 'all') -> List[pd.DataFrame]:
    """
    获取所有站点的数据帧，用于与现有代码兼容
    
    Args:
        data_dir: 数据目录路径
        data_type: 数据类型，'wind'、'solar'或'all'
        
    Returns:
        数据帧列表
    """
    processor = RenewableEnergyDataProcessor(data_dir)
    
    all_dfs = []
    
    # 加载风场数据
    if data_type in ['wind', 'all']:
        wind_files = processor.list_wind_farms()
        for file in wind_files:
            df = processor.read_wind_farm_data(file)
            df = processor.preprocess_data(df)
            # 自动确定目标列
            target_col = None
            possible_target_cols = ['Power (MW)', 'power', 'ActivePower', 'Generated Power']
            for col in possible_target_cols:
                if col in df.columns:
                    target_col = col
                    break
            if target_col is None:
                target_col = df.columns[-1]
            # 重命名目标列为'power'以与现有代码兼容
            df = df.rename(columns={target_col: 'power'})
            # 移除时间列，因为现有代码使用索引
            time_columns = ['Time(year-month-day h:m:s)', 'Time', 'time']
            for col in time_columns:
                if col in df.columns:
                    df = df.drop(columns=[col])
                    break
            all_dfs.append(df)
    
    # 加载光伏电站数据
    if data_type in ['solar', 'all']:
        solar_files = processor.list_solar_stations()
        for file in solar_files:
            df = processor.read_solar_station_data(file)
            df = processor.preprocess_data(df)
            # 自动确定目标列
            target_col = None
            possible_target_cols = ['Power (MW)', 'power', 'ActivePower', 'Generated Power']
            for col in possible_target_cols:
                if col in df.columns:
                    target_col = col
                    break
            if target_col is None:
                target_col = df.columns[-1]
            # 重命名目标列为'power'以与现有代码兼容
            df = df.rename(columns={target_col: 'power'})
            # 移除时间列，因为现有代码使用索引
            time_columns = ['Time(year-month-day h:m:s)', 'Time', 'time']
            for col in time_columns:
                if col in df.columns:
                    df = df.drop(columns=[col])
                    break
            all_dfs.append(df)
    
    return all_dfs
