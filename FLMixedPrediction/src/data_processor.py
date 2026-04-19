import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict

class RenewableEnergyDataProcessor:
    """
    可再生能源数据处理器，用于读取、预处理风电和光伏数据
    """
    
    def __init__(self, data_dir: str):
        """
        初始化数据处理器
        
        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = Path(data_dir)
        self.wind_farms_dir = self.data_dir / "wind_farms"
        self.solar_stations_dir = self.data_dir / "solar_stations"
        
    def list_wind_farms(self) -> List[str]:
        """
        列出所有风场数据文件
        
        Returns:
            风场文件路径列表
        """
        return sorted([str(file) for file in self.wind_farms_dir.glob("*.xlsx")])
    
    def list_solar_stations(self) -> List[str]:
        """
        列出所有光伏电站数据文件
        
        Returns:
            光伏电站文件路径列表
        """
        return sorted([str(file) for file in self.solar_stations_dir.glob("*.xlsx")])
    
    def read_wind_farm_data(self, file_path: str) -> pd.DataFrame:
        """
        读取风场数据
        
        Args:
            file_path: 风场数据文件路径
            
        Returns:
            风场数据DataFrame
        """
        return pd.read_excel(file_path)
    
    def read_solar_station_data(self, file_path: str) -> pd.DataFrame:
        """
        读取光伏电站数据
        
        Args:
            file_path: 光伏电站数据文件路径
            
        Returns:
            光伏电站数据DataFrame
        """
        return pd.read_excel(file_path)
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        预处理数据
        
        Args:
            df: 原始数据DataFrame
            
        Returns:
            预处理后的数据DataFrame
        """
        # 复制数据，避免修改原始数据
        df = df.copy()
        
        # 确保时间列为datetime类型
        time_columns = ['Time(year-month-day h:m:s)', 'Time', 'time']
        for time_col in time_columns:
            if time_col in df.columns:
                # 判断时间列是否为字符串类型
                if df[time_col].dtype == 'object':
                    # 处理24:00:00的情况，转换为第二天的00:00:00
                    df[time_col] = df[time_col].str.replace(' 24:', ' 00:')
                    df[time_col] = pd.to_datetime(df[time_col])
                else:
                    # 已经是datetime类型，直接转换
                    df[time_col] = pd.to_datetime(df[time_col])
                break
        
        # 处理缺失值
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # 确保所有数值列都是float类型
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        df[numeric_cols] = df[numeric_cols].astype(float)
        
        return df
    
    def build_features(self, df: pd.DataFrame, target_col: str = 'power', seq_len: int = 24, horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        构建特征和目标数据
        
        Args:
            df: 预处理后的数据DataFrame
            target_col: 目标列名称
            seq_len: 序列长度
            horizon: 预测步长
            
        Returns:
            X: 特征数据，形状为[样本数, 序列长度, 特征数]
            y: 目标数据，形状为[样本数]
        """
        # 提取特征列（排除目标列和时间列）
        time_columns = ['Time(year-month-day h:m:s)', 'Time', 'time']
        exclude_cols = time_columns + [target_col]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # 确保特征列存在
        if not feature_cols:
            # 如果没有其他特征，使用目标列的滞后特征
            df['power_lag1'] = df[target_col].shift(1).bfill()
            feature_cols = ['power_lag1']
        
        # 构建序列数据
        X, y = [], []
        for i in range(seq_len, len(df) - horizon + 1):
            # 获取输入序列
            x_seq = df[feature_cols].iloc[i - seq_len:i].values
            # 获取目标值
            y_val = df[target_col].iloc[i + horizon - 1]
            
            X.append(x_seq)
            y.append(y_val)
        
        return np.array(X), np.array(y)
    
    def get_data_for_model(self, file_path: str, data_type: str = 'wind', seq_len: int = 24, horizon: int = 1) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        获取模型可用的数据
        
        Args:
            file_path: 数据文件路径
            data_type: 数据类型，'wind'或'solar'
            seq_len: 序列长度
            horizon: 预测步长
            
        Returns:
            X: 特征数据，形状为[样本数, 序列长度, 特征数]
            y: 目标数据，形状为[样本数]
            feature_cols: 特征列名称列表
        """
        # 读取数据
        if data_type == 'wind':
            df = self.read_wind_farm_data(file_path)
        elif data_type == 'solar':
            df = self.read_solar_station_data(file_path)
        else:
            raise ValueError(f"未知的数据类型: {data_type}")
        
        # 根据数据列名自动确定目标列
        target_col = None
        possible_target_cols = ['Power (MW)', 'power', 'ActivePower', 'Generated Power']
        for col in possible_target_cols:
            if col in df.columns:
                target_col = col
                break
        
        # 如果没有找到目标列，默认使用最后一列
        if target_col is None:
            target_col = df.columns[-1]
        
        # 构建特征和目标
        X, y = self.build_features(df, target_col, seq_len, horizon)
        
        # 获取特征列名称
        time_columns = ['Time(year-month-day h:m:s)', 'Time', 'time']
        feature_cols = [col for col in df.columns if col not in time_columns + [target_col]]
        if not feature_cols:
            feature_cols = ['power_lag1']
        
        return X, y, feature_cols
    
    def get_all_sites_data(self, data_type: str = 'wind', seq_len: int = 24, horizon: int = 1) -> Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]]:
        """
        获取所有站点的数据
        
        Args:
            data_type: 数据类型，'wind'或'solar'
            seq_len: 序列长度
            horizon: 预测步长
            
        Returns:
            站点数据字典，键为站点名称，值为(X, y, feature_cols)
        """
        sites_data = {}
        
        if data_type == 'wind':
            files = self.list_wind_farms()
        elif data_type == 'solar':
            files = self.list_solar_stations()
        else:
            raise ValueError(f"未知的数据类型: {data_type}")
        
        for file_path in files:
            # 提取站点名称
            site_name = Path(file_path).stem
            
            # 获取模型数据
            X, y, feature_cols = self.get_data_for_model(file_path, data_type, seq_len, horizon)
            
            # 保存数据
            sites_data[site_name] = (X, y, feature_cols)
        
        return sites_data


def main():
    """
    主函数，用于测试数据处理器
    """
    # 初始化数据处理器
    data_dir = "/home/plasmid/Project/FLMixedPrediction/data/Renewable-energy-generation-input-feature-variables-analysis-main/data_processed"
    processor = RenewableEnergyDataProcessor(data_dir)
    
    # 测试风场数据
    wind_files = processor.list_wind_farms()
    print(f"找到 {len(wind_files)} 个风场数据文件")
    
    # 测试光伏电站数据
    solar_files = processor.list_solar_stations()
    print(f"找到 {len(solar_files)} 个光伏电站数据文件")
    
    # 测试单个风场数据处理
    if wind_files:
        wind_file = wind_files[0]
        print(f"\n测试风场数据: {Path(wind_file).stem}")
        
        # 读取并预处理数据
        df = processor.read_wind_farm_data(wind_file)
        df = processor.preprocess_data(df)
        print(f"原始数据形状: {df.shape}")
        print(f"数据列: {list(df.columns)}")
        
        # 自动确定目标列
        target_col = None
        possible_target_cols = ['Power (MW)', 'power', 'ActivePower', 'Generated Power']
        for col in possible_target_cols:
            if col in df.columns:
                target_col = col
                break
        
        # 如果没有找到目标列，默认使用最后一列
        if target_col is None:
            target_col = df.columns[-1]
        
        print(f"使用目标列: {target_col}")
        
        # 构建特征
        X, y = processor.build_features(df, target_col=target_col)
        print(f"特征数据形状: {X.shape}")
        print(f"目标数据形状: {y.shape}")
    
    # 测试单个光伏电站数据处理
    if solar_files:
        solar_file = solar_files[0]
        print(f"\n测试光伏电站数据: {Path(solar_file).stem}")
        
        # 读取并预处理数据
        df = processor.read_solar_station_data(solar_file)
        df = processor.preprocess_data(df)
        print(f"原始数据形状: {df.shape}")
        print(f"数据列: {list(df.columns)}")
        
        # 自动确定目标列
        target_col = None
        possible_target_cols = ['Power (MW)', 'power', 'ActivePower', 'Generated Power']
        for col in possible_target_cols:
            if col in df.columns:
                target_col = col
                break
        
        # 如果没有找到目标列，默认使用最后一列
        if target_col is None:
            target_col = df.columns[-1]
        
        print(f"使用目标列: {target_col}")
        
        # 构建特征
        X, y = processor.build_features(df, target_col=target_col)
        print(f"特征数据形状: {X.shape}")
        print(f"目标数据形状: {y.shape}")


if __name__ == "__main__":
    main()
