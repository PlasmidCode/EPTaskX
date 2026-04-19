import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import flwr as fl
from flwr.server import ServerConfig
import logging
from datetime import datetime

from .model import TinyLSTM, MOE, AMD
from .utils import load_csv, build_loaders
from .renewable_data_integrator import get_all_sites_dataframes


def create_model(model_type: str, in_dim: int, num_experts: int = 3, seq_len: int = 24, pred_len: int = 1):
    """根据参数创建相应的模型"""
    if model_type == "moe":
        return MOE(in_dim, num_experts=num_experts)
    elif model_type == "amd":
        # AMD模型需要输入形状、预测长度等参数
        input_shape = (seq_len, in_dim)
        # 确保pred_len为1，与horizon参数匹配
        pred_len = 1
        return AMD(
            input_shape=input_shape,
            pred_len=pred_len,
            n_block=2,
            dropout=0.1,
            patch=12,
            k=3,
            c=2,
            alpha=0.5,
            target_slice=None,
            norm=True,
            layernorm=True,
        )
    else:
        return TinyLSTM(in_dim)


class Client(fl.client.NumPyClient):
    def __init__(self, df, seq, horizon, batch, device, model_type="tiny_lstm", num_experts=3, client_id=0):
        self.device = device
        self.client_id = client_id
        self.tr_loader, self.va_loader, in_dim = build_loaders(df, seq, horizon, batch)
        
        # 根据模型类型创建模型实例
        self.model = create_model(model_type, in_dim, num_experts, seq, horizon).to(device)
        self.crit = nn.MSELoss()
        self.opt = optim.Adam(self.model.parameters(), lr=1e-3)

    # ---- Flower NumPyClient API ----
    def get_parameters(self, config):
        return [p.detach().cpu().numpy() for p in self.model.state_dict().values()]

    def set_parameters(self, params):
        sd = {k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), params)}
        self.model.load_state_dict(sd, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        epochs = int(config.get("local_epochs", 1))
        current_round = config.get("current_round", 0)
        
        total_loss = 0.0
        num_batches = 0
        
        total_moe_loss = 0.0
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_moe_loss = 0.0
            epoch_batches = 0
            for xb, yb in self.tr_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.opt.zero_grad()
                output = self.model(xb)
                # 检查模型输出是否包含损失项（如AMD模型）
                if isinstance(output, tuple):
                    pred, moe_loss = output
                    # 确保pred的形状与yb相同
                    # 如果pred是4D张量，取最后一个时间步、第一个特征值和第一个预测值
                    if pred.ndim == 4:
                        pred = pred[:, -1, 0, 0].squeeze()
                    # 如果pred是3D张量，取第一个时间步和第一个特征值
                    elif pred.ndim == 3:
                        pred = pred[:, 0, 0].squeeze()
                    # 如果pred是2D张量，取第一个特征值
                    elif pred.ndim == 2:
                        pred = pred[:, 0].squeeze()
                    # 主损失 + moe_loss
                    main_loss = self.crit(pred, yb)
                    loss = main_loss + moe_loss
                    epoch_moe_loss += moe_loss.item()
                else:
                    pred = output
                    loss = self.crit(pred, yb)
                loss.backward()
                self.opt.step()
                
                epoch_loss += loss.item()
                epoch_batches += 1
            
            avg_epoch_loss = epoch_loss / epoch_batches
            total_loss += avg_epoch_loss
            if epoch_batches > 0:
                avg_epoch_moe_loss = epoch_moe_loss / epoch_batches
                total_moe_loss += avg_epoch_moe_loss
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        # 如果是AMD模型，添加moe_loss指标
        metrics = {"train_loss": avg_loss, "client_id": self.client_id, "round": current_round}
        if total_moe_loss > 0:
            avg_moe_loss = total_moe_loss / num_batches
            metrics["moe_loss"] = avg_moe_loss
        return self.get_parameters({}), len(self.tr_loader.dataset), metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        mse, mae, n = 0.0, 0.0, 0
        total_preds = []
        total_ys = []
        current_round = config.get("current_round", 0)
        
        # 用于记录门控权重的变量
        gate_weights_list = []
        
        with torch.no_grad():
            for xb, yb in self.va_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                output = self.model(xb)
                
                # 检查模型输出是否包含损失项（如AMD模型）
                if isinstance(output, tuple):
                    pred, _ = output
                    # 确保pred的形状与yb相同
                    # 如果pred是4D张量，取最后一个时间步、第一个特征值和第一个预测值
                    if pred.ndim == 4:
                        pred = pred[:, -1, 0, 0].squeeze()
                    # 如果pred是3D张量，取第一个时间步和第一个特征值
                    elif pred.ndim == 3:
                        pred = pred[:, 0, 0].squeeze()
                    # 如果pred是2D张量，取第一个特征值
                    elif pred.ndim == 2:
                        pred = pred[:, 0].squeeze()
                else:
                    pred = output
                
                # 如果是MOE模型，收集门控权重
                if hasattr(self.model, 'gate_weights'):
                    gate_weights_list.append(self.model.gate_weights.cpu().numpy())
                
                # 计算MSE和MAE
                mse += torch.mean((pred - yb) ** 2).item() * len(yb)
                mae += torch.mean(torch.abs(pred - yb)).item() * len(yb)
                n += len(yb)
                
                # 保存所有预测值和真实值，用于计算R2
                total_preds.extend(pred.cpu().numpy())
                total_ys.extend(yb.cpu().numpy())
        
        # 计算平均指标
        mse_avg = float(mse / max(n, 1))
        mae_avg = float(mae / max(n, 1))
        rmse = float(np.sqrt(mse_avg))
        
        # 计算R2指标
        total_preds = np.array(total_preds)
        total_ys = np.array(total_ys)
        if len(total_ys) > 1:
            y_mean = np.mean(total_ys)
            ss_total = np.sum((total_ys - y_mean) ** 2)
            ss_residual = np.sum((total_ys - total_preds) ** 2)
            r2 = 1.0 - (ss_residual / (ss_total + 1e-10))
        else:
            r2 = 0.0
        
        # 准备返回的指标
        metrics = {"rmse": rmse, "mse": mse_avg, "mae": mae_avg, "r2": r2, "client_id": self.client_id, "round": current_round}
        
        # 如果有门控权重，计算统计信息并添加到指标中
        if gate_weights_list:
            all_gate_weights = np.concatenate(gate_weights_list, axis=0)
            for expert_idx in range(all_gate_weights.shape[1]):
                expert_weights = all_gate_weights[:, expert_idx]
                metrics[f"gate_weight_expert_{expert_idx}_mean"] = float(np.mean(expert_weights))
                metrics[f"gate_weight_expert_{expert_idx}_std"] = float(np.std(expert_weights))
        
        return mse_avg, n, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=24)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--local_epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--model", type=str, choices=["tiny_lstm", "moe", "amd"], default="tiny_lstm", help="Model type to use")
    ap.add_argument("--num_experts", type=int, default=3, help="Number of experts for MOE model")
    ap.add_argument("--tb_log_dir", type=str, default="logs", help="TensorBoard log directory")
    ap.add_argument("--use_renewable", action="store_true", help="Use renewable energy dataset (wind and solar farms)")
    ap.add_argument("--renewable_data_type", type=str, choices=["wind", "solar", "all"], default="all", help="Type of renewable energy data to use")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    
    # 显式使用 CPU 以避免 CUDA 错误
    device = torch.device("cpu")
    
    if args.use_renewable:
        # 使用可再生能源数据集
        renewable_data_dir = root / "data" / "Renewable-energy-generation-input-feature-variables-analysis-main" / "data_processed"
        print(f"Using renewable energy data from: {renewable_data_dir}")
        dfs = get_all_sites_dataframes(str(renewable_data_dir), args.renewable_data_type)
        print(f"Loaded {len(dfs)} sites data")
        
        # 检查所有客户端的特征维度是否相同
        if len(dfs) > 1:
            # 获取第一个客户端的特征维度
            _, _, in_dim_0 = build_loaders(dfs[0], args.seq, args.horizon, args.batch)
            print(f"First client feature dimension: {in_dim_0}")
            
            # 过滤出具有相同特征维度的客户端
            filtered_dfs = []
            for i, df in enumerate(dfs):
                _, _, in_dim_i = build_loaders(df, args.seq, args.horizon, args.batch)
                if in_dim_i == in_dim_0:
                    filtered_dfs.append(df)
                    print(f"Client {i}: feature dimension {in_dim_i} (kept)")
                else:
                    print(f"Client {i}: feature dimension {in_dim_i} (different, removed)")
            
            # 使用过滤后的客户端列表
            dfs = filtered_dfs
            print(f"After filtering, using {len(dfs)} clients with the same feature dimension")
    else:
        # 使用原始数据集
        pv = load_csv(root / "data" / "pv_site.csv")
        wd = load_csv(root / "data" / "wind_site.csv")
        dfs = [pv, wd]
    
    # 创建 TensorBoard 日志目录
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_log_dir = f"{args.tb_log_dir}/{current_time}_{args.model}_experts{args.num_experts}"
    writer = SummaryWriter(tb_log_dir)
    
    print(f"TensorBoard logs will be saved to: {tb_log_dir}")
    print(f"Run: tensorboard --logdir={args.tb_log_dir} to view results")

    # 修复点1：返回 Client 类型（由 NumPyClient 转换）
    def client_fn(cid: str):
        return Client(dfs[int(cid)], args.seq, args.horizon, args.batch, device, 
                     args.model, args.num_experts, client_id=int(cid)).to_client()

    # 定义一个自定义的 fit_config_fn，添加当前轮数信息
    def fit_config_fn(rnd):
        return {"local_epochs": args.local_epochs, "current_round": rnd}
    
    # 定义一个自定义的 evaluate_config_fn，添加当前轮数信息
    def evaluate_config_fn(rnd):
        return {"current_round": rnd}
    
    # 定义一个全局变量来存储指标
    global_metrics = []
    
    # 定义一个自定义的聚合函数，记录全局指标
    def aggregate_fit_metrics(metrics):
        if not metrics:
            return {}
        
        # 记录客户端指标
        client_metrics = []
        for num_examples, m in metrics:
            client_metrics.append({
                "client_id": m["client_id"],
                "round": m["round"],
                "train_loss": m["train_loss"],
                "num_examples": num_examples
            })
        
        # 保存到全局指标列表
        global_metrics.append({"type": "fit", "metrics": client_metrics})
        
        # 聚合指标
        total_loss = 0.0
        total_weight = 0
        
        for num_examples, m in metrics:
            total_loss += m["train_loss"] * num_examples
            total_weight += num_examples
        
        avg_loss = total_loss / total_weight if total_weight > 0 else 0
        
        return {"avg_train_loss": avg_loss}
    
    # 定义一个自定义的评估聚合函数，记录全局指标
    def aggregate_evaluate_metrics(metrics):
        if not metrics:
            return {}
        
        # 记录客户端指标
        client_metrics = []
        for num_examples, m in metrics:
            client_metrics.append({
                "client_id": m["client_id"],
                "round": m["round"],
                "mse": m["mse"],
                "rmse": m["rmse"],
                "num_examples": num_examples
            })
        
        # 保存到全局指标列表
        global_metrics.append({"type": "evaluate", "metrics": client_metrics})
        
        # 聚合指标
        total_rmse = 0.0
        total_mse = 0.0
        total_weight = 0
        
        for num_examples, m in metrics:
            total_rmse += m["rmse"] * num_examples
            total_mse += m["mse"] * num_examples
            total_weight += num_examples
        
        avg_rmse = total_rmse / total_weight
        avg_mse = total_mse / total_weight
        
        return {"rmse": avg_rmse, "mse": avg_mse}
    
    # 简化 TensorBoard 集成，直接在服务器端记录指标
    def log_metrics(writer, rnd, results, metric_prefix="Eval"):
        """记录指标到 TensorBoard"""
        # 记录客户端指标
        for client_id, (_, _, metrics) in enumerate(results):
            if "rmse" in metrics and "mse" in metrics:
                writer.add_scalar(f"Client_{client_id}/{metric_prefix}_MSE", 
                                metrics["mse"], rnd)
                writer.add_scalar(f"Client_{client_id}/{metric_prefix}_RMSE", 
                                metrics["rmse"], rnd)
        
        # 计算全局指标
        if results:
            total_rmse = 0.0
            total_mse = 0.0
            total_weight = 0
            
            for _, (num_examples, _, metrics) in enumerate(results):
                total_rmse += metrics["rmse"] * num_examples
                total_mse += metrics["mse"] * num_examples
                total_weight += num_examples
            
            if total_weight > 0:
                avg_rmse = total_rmse / total_weight
                avg_mse = total_mse / total_weight
                
                # 记录全局指标
                writer.add_scalar(f"Global/{metric_prefix}_MSE", avg_mse, rnd)
                writer.add_scalar(f"Global/{metric_prefix}_RMSE", avg_rmse, rnd)
    
    # 创建简单的日志记录器
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("flwr")
    
    # 创建标准的 FedAvg 策略
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=2,
        on_fit_config_fn=fit_config_fn,
        on_evaluate_config_fn=evaluate_config_fn,
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics,
    )
    


    # 修复点2：新版 Flower 用 ServerConfig 指定轮数
    config = ServerConfig(num_rounds=args.rounds)

    # 运行联邦学习模拟
    print("Starting federated learning simulation...")
    
    # 确保客户端数量不超过可用站点数量
    num_clients = min(2, len(dfs))
    print(f"Using {num_clients} clients out of {len(dfs)} available sites")
    
    # 创建初始全局模型
    _, _, in_dim = build_loaders(dfs[0], args.seq, args.horizon, args.batch)
    global_model = create_model(args.model, in_dim, args.num_experts)
    
    # 更新策略中的最小可用客户端数量
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=num_clients,
        on_fit_config_fn=fit_config_fn,
        on_evaluate_config_fn=evaluate_config_fn,
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics,
    )
    
    # 运行模拟
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=config,
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
        ray_init_args={"ignore_reinit_error": True, "num_cpus": num_clients}
    )
    
    # 在模拟结束后，将所有收集到的指标记录到 TensorBoard
    print("\nRecording metrics to TensorBoard...")
    for metric_entry in global_metrics:
        metric_type = metric_entry["type"]
        client_metrics = metric_entry["metrics"]
        
        if metric_type == "fit":
            for m in client_metrics:
                writer.add_scalar(f"Client_{m['client_id']}/Train_Loss", 
                                m["train_loss"], m["round"])
                # 记录AMD模型的moe_loss指标
                if "moe_loss" in m:
                    writer.add_scalar(f"Client_{m['client_id']}/MOE_Loss", 
                                    m["moe_loss"], m["round"])
        elif metric_type == "evaluate":
            for m in client_metrics:
                writer.add_scalar(f"Client_{m['client_id']}/Val_MSE", 
                                m["mse"], m["round"])
                writer.add_scalar(f"Client_{m['client_id']}/Val_RMSE", 
                                m["rmse"], m["round"])
                # 记录MAE和R2指标
                if "mae" in m:
                    writer.add_scalar(f"Client_{m['client_id']}/Val_MAE", 
                                    m["mae"], m["round"])
                if "r2" in m:
                    writer.add_scalar(f"Client_{m['client_id']}/Val_R2", 
                                    m["r2"], m["round"])
                # 记录门控权重信息
                for key, value in m.items():
                    if "gate_weight" in key:
                        writer.add_scalar(f"Client_{m['client_id']}/{key}", value, m["round"])
    
    # 记录全局平均指标
    for metric_entry in global_metrics:
        if metric_entry["type"] == "evaluate":
            client_metrics = metric_entry["metrics"]
            if client_metrics:
                round_num = client_metrics[0]["round"]
                
                # 计算全局平均指标
                total_mse = sum(m["mse"] * m["num_examples"] for m in client_metrics)
                total_rmse = sum(m["rmse"] * m["num_examples"] for m in client_metrics)
                total_mae = sum(m["mae"] * m["num_examples"] for m in client_metrics if "mae" in m)
                total_r2 = sum(m["r2"] * m["num_examples"] for m in client_metrics if "r2" in m)
                total_examples = sum(m["num_examples"] for m in client_metrics)
                
                avg_mse = total_mse / total_examples
                avg_rmse = total_rmse / total_examples
                avg_mae = total_mae / total_examples
                avg_r2 = total_r2 / total_examples
                
                writer.add_scalar("Global/Val_MSE", avg_mse, round_num)
                writer.add_scalar("Global/Val_RMSE", avg_rmse, round_num)
                writer.add_scalar("Global/Val_MAE", avg_mae, round_num)
                writer.add_scalar("Global/Val_R2", avg_r2, round_num)
                
                # 记录每个专家的权重统计
                # 确定有多少个专家
                num_experts = 0
                for m in client_metrics:
                    for key in m:
                        if "gate_weight_expert_" in key and "_mean" in key:
                            idx = int(key.split("_")[2])
                            num_experts = max(num_experts, idx + 1)
                
                for expert_idx in range(num_experts):
                    total_weight = 0
                    count = 0
                    for m in client_metrics:
                        key = f"gate_weight_expert_{expert_idx}_mean"
                        if key in m:
                            total_weight += m[key]
                            count += 1
                    if count > 0:
                        avg_weight = total_weight / count
                        writer.add_scalar(f"Experts/Avg_Gate_Weight_Expert_{expert_idx}", avg_weight, round_num)
    
    # 为了简化 TensorBoard 集成，我们可以使用一个更简单的方法：
    # 在每轮训练和评估时，通过客户端的 fit 和 evaluate 方法返回指标
    # 并在 TensorBoard 中记录这些指标
    
    # 注意：由于 Flower 1.8.0 的限制，我们无法直接从策略中获取这些指标
    # 但是我们可以在客户端的 fit 和 evaluate 方法中记录一些基本信息
    print("\nFederated learning simulation completed!")
    print(f"TensorBoard logs have been saved to: {tb_log_dir}")
    print(f"To view the results, run:")
    print(f"  tensorboard --logdir={args.tb_log_dir}")
    print(f"Then open http://localhost:6006 in your browser.")
    
    # 关闭 TensorBoard 写入器
    writer.close()
    
    # 打印完成信息
    print(f"\nFederated learning simulation completed!")
    print(f"TensorBoard logs are saved to: {args.tb_log_dir}")
    print(f"To view the results, run: tensorboard --logdir={args.tb_log_dir}")
    print(f"\nFederated learning simulation completed.")
    print(f"To view TensorBoard results, run:")
    print(f"tensorboard --logdir={args.tb_log_dir}")
    print(f"Then open http://localhost:6006 in your browser.")


if __name__ == "__main__":
    main()
