import argparse
import os

from algorithms.greedy_scheduler import GreedyScheduler
from algorithms.heuristic_schedulers import (
    EarliestDeadlineFirstScheduler,
    LeastLoadedScheduler,
    MinGridScheduler,
    RandomScheduler,
    SourceAffinityScheduler,
)
from algorithms.optimization_scheduler import OptimizationScheduler
from algorithms.robust_mpc_alns_scheduler import RobustMPCALNSScheduler
from model.problem import ProblemModel
from utils.alibaba_trace_loader import load_alibaba_v2018_batch_trace, load_alibaba_v2023_gpu_trace
from utils.forecast_loader import load_forecast_csv, write_forecast_csv
from utils.schedule_metrics import evaluate_schedule
from utils.test_data_generator import TestDataGenerator
from visualization.visualizer import Visualizer


def build_problem(args):
    params = build_params(args)
    if args.forecast_csv:
        forecast = load_forecast_csv(args.forecast_csv, D=params["D"], T=params["T"])
        params.update(
            {
                "R_hat": forecast["R_hat"],
                "R_mean": forecast["R_mean"],
                "R_lower": forecast["R_lower"],
                "R_upper": forecast["R_upper"],
                "site_metadata": forecast["site_metadata"],
            }
        )
    if args.export_forecast_csv:
        site_ids = [site["site_id"] for site in params["site_metadata"]]
        write_forecast_csv(
            args.export_forecast_csv,
            params["R_mean"],
            params["R_lower"],
            params["R_upper"],
            site_ids=site_ids,
        )

    return ProblemModel(
        D=params["D"],
        T=params["T"],
        R_hat=params["R_hat"],
        Cap=params["Cap"],
        Base=params["Base"],
        BW_tot=params["BW_tot"],
        tasks=params["tasks"],
        alpha=params["alpha"],
        E_base=params["E_base"],
        R_mean=params.get("R_mean"),
        R_lower=params.get("R_lower"),
        R_upper=params.get("R_upper"),
        site_metadata=params.get("site_metadata"),
        battery=params.get("battery"),
        BW_link=params.get("BW_link"),
        migration_cooldown=params.get("migration_cooldown", 0),
        weights=params.get("weights"),
        slot_length_hours=params.get("slot_length_hours", 1.0),
    )


def build_params(args):
    if args.workload == "synthetic":
        return TestDataGenerator(seed=args.seed).generate_problem(D=args.domains, T=args.slots, J=args.tasks)
    if args.workload == "alibaba-v2023":
        if not args.pod_csv:
            raise ValueError("--pod-csv is required for --workload alibaba-v2023")
        return load_alibaba_v2023_gpu_trace(
            pod_csv=args.pod_csv,
            node_csv=args.node_csv,
            domains=args.domains,
            slots=args.slots,
            slot_seconds=args.slot_seconds,
            task_limit=args.trace_task_limit,
            seed=args.seed,
            deadline_slack=args.deadline_slack,
            target_utilization=args.target_utilization,
            sample_policy=getattr(args, "trace_sample_policy", "random"),
            window_index=getattr(args, "trace_window_index", 0),
        )
    if args.workload == "alibaba-v2018":
        if not args.batch_task_csv:
            raise ValueError("--batch-task-csv is required for --workload alibaba-v2018")
        return load_alibaba_v2018_batch_trace(
            batch_task_csv=args.batch_task_csv,
            machine_meta_csv=args.machine_meta_csv,
            domains=args.domains,
            slots=args.slots,
            slot_seconds=args.slot_seconds,
            task_limit=args.trace_task_limit,
            seed=args.seed,
            deadline_slack=args.deadline_slack,
            target_utilization=args.target_utilization,
            sample_policy=getattr(args, "trace_sample_policy", "random"),
            window_index=getattr(args, "trace_window_index", 0),
        )
    raise ValueError(f"Unsupported workload: {args.workload}")


def main():
    parser = argparse.ArgumentParser(description="Forecast-driven robust task scheduling demo")
    parser.add_argument("--workload", choices=["synthetic", "alibaba-v2023", "alibaba-v2018"], default="synthetic")
    parser.add_argument("--forecast-csv", help="CSV with site_id, slot, r_mean_kwh, r_p10_kwh, r_p90_kwh")
    parser.add_argument("--export-forecast-csv", help="Write generated simulator forecasts to this CSV path")
    parser.add_argument("--pod-csv", help="Alibaba v2023 openb_pod_list_*.csv")
    parser.add_argument("--node-csv", help="Alibaba v2023 openb_node_list_*.csv")
    parser.add_argument("--batch-task-csv", help="Alibaba v2018 batch_task.csv")
    parser.add_argument("--machine-meta-csv", help="Alibaba v2018 machine_meta.csv")
    parser.add_argument("--domains", type=int, default=2)
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--tasks", type=int, default=3)
    parser.add_argument("--trace-task-limit", type=int, default=200)
    parser.add_argument("--slot-seconds", type=int, default=3600)
    parser.add_argument("--deadline-slack", type=float, default=1.5)
    parser.add_argument("--target-utilization", type=float, default=0.68)
    parser.add_argument("--trace-sample-policy", choices=["random", "earliest", "latest", "peak"], default="peak")
    parser.add_argument("--trace-window-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=os.path.join("results", "demo"))
    parser.add_argument("--alns-iterations", type=int, default=300)
    parser.add_argument("--algorithms", default="robust_mpc_alns,min_grid,edf,source_affinity,least_loaded,greedy")
    args = parser.parse_args()

    problem = build_problem(args)
    visualizer = Visualizer(problem)
    registry = {
        "robust_mpc_alns": lambda: RobustMPCALNSScheduler(problem, alns_iterations=args.alns_iterations, seed=args.seed),
        "min_grid": lambda: MinGridScheduler(problem, seed=args.seed),
        "edf": lambda: EarliestDeadlineFirstScheduler(problem, seed=args.seed),
        "source_affinity": lambda: SourceAffinityScheduler(problem, seed=args.seed),
        "least_loaded": lambda: LeastLoadedScheduler(problem, seed=args.seed),
        "greedy": lambda: GreedyScheduler(problem, seed=args.seed),
        "random": lambda: RandomScheduler(problem, seed=args.seed),
        "optimization": lambda: OptimizationScheduler(problem),
    }
    schedulers = [registry[name]() for name in _parse_algorithms(args.algorithms) if name in registry]

    results = []
    for scheduler in schedulers:
        print(f"Running algorithm: {scheduler.__class__.__name__}")
        result = scheduler.solve()
        if "x" in result:
            result["metrics"] = evaluate_schedule(
                problem,
                result["x"],
                risk_mode=result.get("risk_mode", "mean"),
            )["metrics"]
        results.append(result)
        print(f"Status: {result['status']}")
        print(f"Objective: {result['objective']:.4f}")
        visualizer.visualize_results(result, args.output_dir)

    visualizer.visualize_comparison(results, args.output_dir)


def _parse_algorithms(raw):
    return [item.strip() for item in raw.split(",") if item.strip()]


if __name__ == "__main__":
    main()
