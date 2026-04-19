import sys
from pathlib import Path

# 将ref目录添加到Python路径
sys.path.append("ref")
from forecast_utils import load_site_xlsx, clean_and_impute, add_time_features, add_wind_vector_features, make_supervised_frame

# 设置数据路径
data_root = Path("data/Renewable-energy-generation-input-feature-variables-analysis-main/data_processed")

# 选择一个有Power(MW)列的风电场
wind_file = data_root / "wind_farms" / "Wind farm site 1 (Nominal capacity-99MW).xlsx"

# 加载数据
df = load_site_xlsx(wind_file, kind="wind")
print(f"原始数据大小: {df.shape}")
print(f"时间范围: {df.index.min()} 到 {df.index.max()}")
print(f"数据点数: {len(df)}")
print(f"列名: {df.columns.tolist()}")

# 查看原始数据的内容，特别是Air_P列
print(f"\n原始数据Air_P列的前10个值:")
print(df['Air_P'].head(10))
print(f"\n原始数据Power(MW)列的前10个值:")
print(df['Power(MW)'].head(10))

# 检查数据类型
print(f"\n数据类型:")
print(df.dtypes)

# 清洗和处理数据
df = clean_and_impute(df)
print(f"\n清洗后Air_P列的前10个值:")
print(df['Air_P'].head(10))

df = add_time_features(df)
df = add_wind_vector_features(df)

# 调试make_supervised_frame函数
def debug_make_supervised_frame(df, target_col="Power(MW)", horizon=12, lags=(1, 2, 3), rolling_windows=(4, 12), exog_lag=(1,)):
    """调试版本的make_supervised_frame函数"""
    d = df.copy()
    print(f"\n步骤1: 初始数据大小: {d.shape}")
    
    # 目标滞后
    for k in lags:
        d[f"{target_col}_lag_{k}"] = d[target_col].shift(k)
    print(f"步骤2: 添加目标滞后后大小: {d.shape}")
    print(f"NaN值数量: {d.isna().sum().sum()}")
    
    # 滚动统计
    for w in rolling_windows:
        d[f"{target_col}_roll_mean_{w}"] = d[target_col].shift(1).rolling(w).mean()
        d[f"{target_col}_roll_std_{w}"] = d[target_col].shift(1).rolling(w).std()
    print(f"步骤3: 添加滚动统计后大小: {d.shape}")
    print(f"NaN值数量: {d.isna().sum().sum()}")
    
    # 外部滞后
    exog_cols = [c for c in d.columns if c != target_col]
    for c in exog_cols[:3]:  # 只处理前3个外部列，减少输出
        for k in exog_lag:
            d[f"{c}_lag_{k}"] = d[c].shift(k)
    print(f"步骤4: 添加外部滞后后大小: {d.shape}")
    print(f"NaN值数量: {d.isna().sum().sum()}")
    
    # 多步目标
    y_cols = []
    for h in range(1, horizon + 1):
        col = f"y_t+{h}"
        d[col] = d[target_col].shift(-h)
        y_cols.append(col)
    print(f"步骤5: 添加多步目标后大小: {d.shape}")
    print(f"NaN值数量: {d.isna().sum().sum()}")
    
    # 删除NaN行
    print(f"步骤6: 删除NaN行前的NaN值分布:")
    print(d.isna().sum().sort_values(ascending=False).head(10))
    
    d_clean = d.dropna()
    print(f"步骤7: 删除NaN行后大小: {d_clean.shape}")
    
    return d_clean

# 运行调试函数
horizon = 12
supervised = debug_make_supervised_frame(df, target_col="Power(MW)", horizon=horizon)

