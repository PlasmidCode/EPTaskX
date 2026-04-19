import os
import pandas as pd

# 测试数据路径
DATA_PATH = "/home/plasmid/Project/FLMixedPrediction/data/renewable_energy/"

print(f"Testing data path: {DATA_PATH}")
print(f"Path exists: {os.path.exists(DATA_PATH)}")

if os.path.exists(DATA_PATH):
    print(f"Directory contents: {os.listdir(DATA_PATH)}")
    
    # 尝试读取一个文件
    csv_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".csv")]
    print(f"CSV files found: {csv_files}")
    
    if csv_files:
        test_file = os.path.join(DATA_PATH, csv_files[0])
        print(f"Trying to read: {test_file}")
        df = pd.read_csv(test_file, parse_dates=["datetime"])
        print(f"Successfully read file with shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print("First few rows:")
        print(df.head())
    else:
        print("No CSV files found")
else:
    print("Path does not exist")
    # 尝试创建路径
    print("Trying to create path...")
    os.makedirs(DATA_PATH, exist_ok=True)
    print(f"Path creation successful: {os.path.exists(DATA_PATH)}")
