import argparse
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

# 导入自定义模块
from config import init_config, get_config
from data import RenewableEnergyDataProcessor
from models import FLMoEModel
from fl import FederatedClient, FederatedServer
from evaluation import calculate_metrics_by_horizon


def setup_seed(seed: int = 42):
    """
    设置随机种子，确保实验可重复性
    
    Args:
        seed: 随机种子值
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_data(config: Dict, processor: RenewableEnergyDataProcessor) -> Tuple[Dict, Dict, Dict]:
    """
    准备训练、验证和测试数据
    
    Args:
        config: 配置字典
        processor: 数据处理器
        
    Returns:
        train_data: 训练数据字典
        val_data: 验证数据字典
        test_data: 测试数据字典
    """
    data_config = config.get_data_config()
    seq_len = data_config['seq_len']
    horizons = data_config['horizons']
    max_horizon = max(horizons)
    val_days = data_config.get('val_days', 30)
    test_days = data_config.get('test_days', 30)
    
    # 获取所有站点数据
    train_data = {}
    val_data = {}
    test_data = {}
    
    # 处理风电数据
    wind_files = processor.list_wind_farms()
    for file_path in wind_files:
        site_name = Path(file_path).stem
        
        # 读取并预处理数据
        df = processor.read_wind_farm_data(file_path)
        df = processor.preprocess_data(df)
        
        # 时间序列切分
        train_df, val_df, test_df = processor.split_time_series(df, val_days=val_days, test_days=test_days)
        
        # 构建特征和目标
        X_train, y_train = processor.build_features(train_df, seq_len=seq_len, horizon=max_horizon)
        X_val, y_val = processor.build_features(val_df, seq_len=seq_len, horizon=max_horizon)
        X_test, y_test = processor.build_features(test_df, seq_len=seq_len, horizon=max_horizon)
        
        # 转换为torch张量
        train_data[site_name] = (torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        val_data[site_name] = (torch.FloatTensor(X_val), torch.FloatTensor(y_val))
        test_data[site_name] = (torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    
    # 处理光伏数据
    solar_files = processor.list_solar_stations()
    for file_path in solar_files:
        site_name = Path(file_path).stem
        
        # 读取并预处理数据
        df = processor.read_solar_station_data(file_path)
        df = processor.preprocess_data(df)
        
        # 时间序列切分
        train_df, val_df, test_df = processor.split_time_series(df, val_days=val_days, test_days=test_days)
        
        # 构建特征和目标
        X_train, y_train = processor.build_features(train_df, seq_len=seq_len, horizon=max_horizon)
        X_val, y_val = processor.build_features(val_df, seq_len=seq_len, horizon=max_horizon)
        X_test, y_test = processor.build_features(test_df, seq_len=seq_len, horizon=max_horizon)
        
        # 转换为torch张量
        train_data[site_name] = (torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        val_data[site_name] = (torch.FloatTensor(X_val), torch.FloatTensor(y_val))
        test_data[site_name] = (torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    
    return train_data, val_data, test_data


def create_model(config: Dict, input_dim: int) -> FLMoEModel:
    """
    创建FLMoE模型
    
    Args:
        config: 配置字典
        input_dim: 输入特征维度
        
    Returns:
        FLMoEModel实例
    """
    model_config = config.get_model_config()
    moe_config = config.get_moe_config()
    data_config = config.get_data_config()
    
    # 模型参数
    seq_len = data_config['seq_len']
    horizons = data_config['horizons']
    model_type = model_config.get('type', 'tcn')
    hidden_dim = model_config.get('hidden_dim', 64)
    num_layers = model_config.get('num_layers', 3)
    dropout = model_config.get('dropout', 0.2)
    
    # MoE参数
    num_experts = moe_config.get('num_experts', 8)
    top_k = moe_config.get('top_k', 2)
    gating_type = moe_config.get('gating_type', 'local')
    load_balancing_loss_weight = moe_config.get('load_balancing_loss_weight', 0.1)
    
    # 创建模型
    model = FLMoEModel(
        input_dim=input_dim,
        seq_len=seq_len,
        horizons=horizons,
        num_experts=num_experts,
        top_k=top_k,
        model_type=model_type,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        gating_type=gating_type,
        load_balancing_loss_weight=load_balancing_loss_weight
    )
    
    return model


def create_clients(config: Dict, model: FLMoEModel, train_data: Dict, val_data: Dict) -> List[FederatedClient]:
    """
    创建联邦学习客户端
    
    Args:
        config: 配置字典
        model: 全局模型
        train_data: 训练数据字典
        val_data: 验证数据字典
        
    Returns:
        客户端列表
    """
    training_config = config.get_training_config()
    fl_config = config.get_federated_config()
    
    clients = []
    for client_id in train_data.keys():
        # 创建客户端模型（使用与全局模型相同的参数）
        client_model = FLMoEModel(
            input_dim=input_dim,
            seq_len=seq_len,
            horizons=model.horizons,
            num_experts=model.num_experts,
            top_k=model.top_k,
            model_type=model.encoder.model_type,
            hidden_dim=model.encoder.hidden_dim,
            num_layers=3,
            dropout=0.2,
            gating_type=model.gating_type,
            load_balancing_loss_weight=model.load_balancing_loss_weight
        )
        
        # 复制全局模型权重
        client_model.load_state_dict(model.state_dict())
        
        # 创建客户端
        client = FederatedClient(
            client_id=client_id,
            model=client_model,
            train_data=train_data[client_id],
            val_data=val_data[client_id],
            device=training_config.get('device', 'cuda'),
            lr=training_config.get('lr', 0.001),
            batch_size=training_config.get('batch_size', 128),
            local_epochs=fl_config.get('local_epochs', 5),
            strategy=fl_config.get('strategy', 'fedavg'),
            fedprox_mu=fl_config.get('fedprox_mu', 0.01)
        )
        
        clients.append(client)
    
    return clients


def main(config_path: str = "configs/base_config.yaml"):
    """
    主函数
    
    Args:
        config_path: 配置文件路径
    """
    # 初始化配置
    config = init_config(config_path)
    
    # 设置随机种子
    setup_seed(config.get_training_config().get('seed', 42))
    
    print("=== FL+MoE Renewable Energy Forecasting ===")
    print(f"配置文件: {config_path}")
    print(f"预测步长: {config.get_data_config()['horizons']}")
    print(f"联邦学习策略: {config.get_federated_config()['strategy']}")
    print(f"MoE专家数量: {config.get_moe_config()['num_experts']}")
    print(f"Top-k专家: {config.get_moe_config()['top_k']}")
    print("=" * 60)
    
    # 初始化数据处理器
    data_config = config.get_data_config()
    data_processor = RenewableEnergyDataProcessor(data_config['root_dir'])
    
    # 准备数据
    print("\n1. 准备数据...")
    train_data, val_data, test_data = prepare_data(config, data_processor)
    print(f"   客户端数量: {len(train_data)}")
    
    # 获取输入维度（从第一个客户端的数据中获取）
    first_client_id = list(train_data.keys())[0]
    input_dim = train_data[first_client_id][0].shape[-1]
    print(f"   输入特征维度: {input_dim}")
    
    # 创建模型
    print("\n2. 创建模型...")
    model = create_model(config, input_dim)
    print(f"   模型类型: {config.get_model_config().get('type', 'tcn')}")
    print(f"   模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建客户端
    print("\n3. 创建联邦学习客户端...")
    clients = create_clients(config, model, train_data, val_data)
    print(f"   客户端数量: {len(clients)}")
    
    # 创建服务器
    print("\n4. 创建联邦学习服务器...")
    fl_config = config.get_federated_config()
    server = FederatedServer(
        model=model,
        clients=clients,
        device=config.get_training_config().get('device', 'cuda'),
        rounds=fl_config.get('rounds', 100),
        client_fraction=fl_config.get('client_fraction', 0.5),
        strategy=fl_config.get('strategy', 'fedavg')
    )
    
    # 开始训练
    print("\n5. 开始联邦学习训练...")
    history = server.train(test_data)
    
    # 保存模型
    print("\n6. 保存模型...")
    server.save_global_model("outputs/global_model.pt")
    
    # 评估模型
    print("\n7. 评估模型...")
    test_results = server.evaluate_global_model(test_data)
    
    # 打印评估结果
    print("\n=== 评估结果 ===")
    for client_id, results in test_results.items():
        print(f"\n客户端 {client_id}:")
        print(f"  Loss: {results['loss']:.6f}")
        print(f"  MAE: {results['mae']:.6f}")
        print(f"  RMSE: {results['rmse']:.6f}")
    
    # 专家使用分析
    print("\n8. 专家使用情况分析...")
    expert_usage = server.analyze_expert_usage()
    print("   专家平均使用比例:")
    for expert_id, usage in expert_usage.items():
        print(f"   {expert_id}: {usage:.4f}")
    
    print("\n=== 训练完成 ===")
    print(f"总轮数: {config.get_federated_config()['rounds']}")
    print(f"最终全局损失: {history[-1]['loss']:.6f}")
    print(f"最终全局MAE: {history[-1]['mae']:.6f}")
    print(f"最终全局RMSE: {history[-1]['rmse']:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL+MoE Renewable Energy Forecasting")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml",
                      help="Path to configuration file")
    
    args = parser.parse_args()
    main(args.config)
