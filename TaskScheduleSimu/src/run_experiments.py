import argparse
import copy
import csv
import os

import numpy as np

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
from main import build_params
from model.problem import ProblemModel
from utils.forecast_loader import load_forecast_csv, write_forecast_csv
from utils.schedule_metrics import evaluate_schedule
from utils.test_data_generator import TestDataGenerator
from visualization.visualizer import Visualizer


class ExperimentRunner:
    """Run synthetic and Alibaba-trace scheduling experiments."""

    def __init__(self, algorithms=None, seed=42, output_root="results/scenarios", alns_iterations=300):
        self.generator = TestDataGenerator(seed=seed)
        self.algorithms = algorithms or ["robust_mpc_alns", "min_grid", "edf", "source_affinity", "greedy"]
        self.output_root = output_root
        self.seed = int(seed)
        self.alns_iterations = int(alns_iterations)

    def run_scenario(self, scenario_name, scenario_params, output_root=None):
        output_root = output_root or self.output_root
        print(f"\n=== Running scenario: {scenario_name} ===")
        problem = self._build_problem(scenario_params)
        visualizer = Visualizer(problem)
        schedulers = self._build_schedulers(problem)
        scenario_output_dir = os.path.join(output_root, scenario_name)
        os.makedirs(scenario_output_dir, exist_ok=True)

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
            print(f"Status: {result.get('status')}")
            print(f"Objective: {result.get('objective')}")
            visualizer.visualize_results(result, scenario_output_dir)

        visualizer.visualize_comparison(results, scenario_output_dir)
        self._generate_scenario_report(scenario_name, results, scenario_output_dir, problem)
        self._write_metrics_csv(results, scenario_output_dir, scenario_name)
        return results

    def run_all_scenarios(self):
        os.makedirs(self.output_root, exist_ok=True)
        all_results = {}
        for scenario in self.generator.generate_scenarios():
            all_results[scenario["name"]] = self.run_scenario(scenario["name"], scenario["params"])
        self._generate_comprehensive_report(all_results)
        print("\n=== All scenarios completed ===")

    def run_trace_workload(self, args):
        all_results = {}
        for window_idx in range(max(int(getattr(args, "trace_windows", 1)), 1)):
            window_args = copy.copy(args)
            window_args.trace_window_index = int(getattr(args, "trace_window_index", 0)) + window_idx
            params = build_params(window_args)
            params = self._apply_forecast_options(params, window_args)
            scenario_name = f"{args.workload}_{args.trace_sample_policy}_window_{window_args.trace_window_index}"
            all_results[scenario_name] = self.run_scenario(scenario_name, params)
        self._generate_comprehensive_report(all_results)

    def _build_problem(self, params):
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

    def _build_schedulers(self, problem):
        registry = {
            "robust_mpc_alns": lambda: RobustMPCALNSScheduler(
                problem,
                alns_iterations=self.alns_iterations,
                seed=self.seed,
            ),
            "min_grid": lambda: MinGridScheduler(problem, seed=self.seed),
            "edf": lambda: EarliestDeadlineFirstScheduler(problem, seed=self.seed),
            "source_affinity": lambda: SourceAffinityScheduler(problem, seed=self.seed),
            "least_loaded": lambda: LeastLoadedScheduler(problem, seed=self.seed),
            "random": lambda: RandomScheduler(problem, seed=self.seed),
            "greedy": lambda: GreedyScheduler(problem, seed=self.seed),
            "optimization": lambda: OptimizationScheduler(problem),
        }
        return [registry[name]() for name in self.algorithms if name in registry]

    def _apply_forecast_options(self, params, args):
        if getattr(args, "forecast_csv", None):
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
        if getattr(args, "export_forecast_csv", None):
            site_ids = [site["site_id"] for site in params["site_metadata"]]
            write_forecast_csv(
                args.export_forecast_csv,
                params["R_mean"],
                params["R_lower"],
                params["R_upper"],
                site_ids=site_ids,
            )
        return params

    def _generate_scenario_report(self, scenario_name, results, output_dir, problem):
        profile = self._scenario_profile(problem)
        rows = []
        for result in results:
            if "x" not in result:
                continue
            metrics = result["metrics"]
            rows.append(
                (
                    "| {algo} | {obj:.4f} | {grid:.4f} | {robust:.4f} | {co2:.4f} | "
                    "{mig:.4f} | {events:.0f} | {comp:.4f} | {miss:.4f} | {clean:.4f} | "
                    "{cfe:.4f} | {curt:.4f} | {link:.4f} |"
                ).format(
                    algo=result["algorithm"],
                    obj=metrics["objective_common"],
                    grid=metrics["expected_grid_energy"],
                    robust=metrics["robust_grid_energy"],
                    co2=metrics["grid_emissions_kgco2e"],
                    mig=metrics["migration"],
                    events=metrics["migration_events"],
                    comp=metrics["completion_rate"],
                    miss=metrics["deadline_miss_rate"],
                    clean=metrics["clean_utilization"],
                    cfe=metrics["cfe_share"],
                    curt=metrics["curtailment"],
                    link=metrics["peak_link_utilization"],
                )
            )
        table = "\n".join(
            [
                (
                    "| Algorithm | Common objective | Expected grid(kWh) | P10 grid(kWh) | CO2e(kg) | "
                    "Migration(GB) | Mig events | Completion | Miss rate | Clean util | CFE | Curtail(kWh) | Peak link util |"
                ),
                "|------|----------|-----------|----------|----------|----------|----------|--------|--------|------------|------|-----------|-----------|",
                *rows,
            ]
        )
        profile_lines = "\n".join(f"- {key}: {value}" for key, value in profile.items())
        content = f"# {scenario_name} scenario report\n\n## Workload profile\n\n{profile_lines}\n\n## Metrics\n\n{table}\n"
        with open(os.path.join(output_dir, "scenario_report.md"), "w", encoding="utf-8") as f:
            f.write(content)

    def _generate_comprehensive_report(self, all_results):
        rows = []
        records = []
        for scenario, results in all_results.items():
            for result in results:
                if "x" not in result:
                    continue
                metrics = result.get("metrics", {})
                records.append((scenario, result["algorithm"], metrics))
                rows.append(
                    "| {scenario} | {algo} | {obj:.4f} | {grid:.4f} | {robust:.4f} | {mig:.4f} | {comp:.4f} | {clean:.4f} | {cfe:.4f} |".format(
                        scenario=scenario,
                        algo=result["algorithm"],
                        obj=metrics.get("objective_common", result["objective"]),
                        grid=metrics.get("expected_grid_energy", np.sum(result["g"])),
                        robust=metrics.get("robust_grid_energy", np.sum(result["g"])),
                        mig=metrics.get("migration", np.sum(result["m"])),
                        comp=metrics.get("completion_rate", 0.0),
                        clean=metrics.get("clean_utilization", 0.0),
                        cfe=metrics.get("cfe_share", 0.0),
                    )
                )
        table = "\n".join(
            [
                "| Scenario | Algorithm | Objective | Expected grid(kWh) | P10 grid(kWh) | Migration(GB) | Completion | Clean util | CFE |",
                "|------|------|----------|-----------|----------|----------|--------|------------|------|",
                *rows,
            ]
        )
        aggregate = self._aggregate_records(records)
        os.makedirs(self.output_root, exist_ok=True)
        with open(os.path.join(self.output_root, "comprehensive_report.md"), "w", encoding="utf-8") as f:
            f.write(f"# Comprehensive experiment report\n\n## Scenario Results\n\n{table}\n\n## Aggregate\n\n{aggregate}\n")
        self._write_comprehensive_csv(records)

    def _legacy_metrics(self, result, problem):
        return evaluate_schedule(problem, result["x"], risk_mode=result.get("risk_mode", "mean"))["metrics"]

    def _scenario_profile(self, problem):
        windows = [int(task["d_j"]) - int(task["a_j"]) + 1 for task in problem.tasks]
        gpu_tasks = [task for task in problem.tasks if task.get("workload_type") == "gpu"]
        return {
            "tasks": problem.J,
            "sites": problem.D,
            "slots": problem.T,
            "total_work": f"{sum(task['W_j'] for task in problem.tasks):.4f}",
            "total_input_size_gb": f"{sum(task['S_j'] for task in problem.tasks):.4f}",
            "avg_task_window_slots": f"{np.mean(windows) if windows else 0.0:.4f}",
            "gpu_task_share": f"{len(gpu_tasks) / problem.J if problem.J else 0.0:.4f}",
            "mean_forecast_uncertainty": f"{np.mean((problem.R_upper - problem.R_lower) / np.maximum(problem.R_mean, 1.0)):.4f}",
        }

    def _write_metrics_csv(self, results, output_dir, scenario_name):
        rows = []
        for result in results:
            if "x" not in result:
                continue
            row = {"scenario": scenario_name, "algorithm": result["algorithm"]}
            row.update(result["metrics"])
            rows.append(row)
        if not rows:
            return
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with open(os.path.join(output_dir, "metrics.csv"), "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _write_comprehensive_csv(self, records):
        rows = []
        for scenario, algorithm, metrics in records:
            row = {"scenario": scenario, "algorithm": algorithm}
            row.update(metrics)
            rows.append(row)
        if not rows:
            return
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with open(os.path.join(self.output_root, "comprehensive_metrics.csv"), "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _aggregate_records(self, records):
        if not records:
            return "No valid results.\n"
        metric_keys = ["objective_common", "expected_grid_energy", "robust_grid_energy", "migration", "completion_rate", "clean_utilization", "cfe_share"]
        algorithms = sorted({algorithm for _, algorithm, _ in records})
        rows = []
        for algorithm in algorithms:
            selected = [metrics for _, algo, metrics in records if algo == algorithm]
            values = {
                key: np.array([metrics.get(key, np.nan) for metrics in selected], dtype=float)
                for key in metric_keys
            }
            rows.append(
                "| {algo} | {n} | {obj:.4f} +/- {obj_std:.4f} | {grid:.4f} | {robust:.4f} | {mig:.4f} | {comp:.4f} | {clean:.4f} | {cfe:.4f} |".format(
                    algo=algorithm,
                    n=len(selected),
                    obj=np.nanmean(values["objective_common"]),
                    obj_std=np.nanstd(values["objective_common"]),
                    grid=np.nanmean(values["expected_grid_energy"]),
                    robust=np.nanmean(values["robust_grid_energy"]),
                    mig=np.nanmean(values["migration"]),
                    comp=np.nanmean(values["completion_rate"]),
                    clean=np.nanmean(values["clean_utilization"]),
                    cfe=np.nanmean(values["cfe_share"]),
                )
            )
        return "\n".join(
            [
                "| Algorithm | N | Objective mean +/- std | Expected grid | P10 grid | Migration | Completion | Clean util | CFE |",
                "|------|---:|----------|-----------|----------|----------|--------|------------|------|",
                *rows,
            ]
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Run TaskScheduleSimu experiments")
    parser.add_argument("--workload", choices=["synthetic", "alibaba-v2023", "alibaba-v2018"], default="synthetic")
    parser.add_argument("--pod-csv")
    parser.add_argument("--node-csv")
    parser.add_argument("--batch-task-csv")
    parser.add_argument("--machine-meta-csv")
    parser.add_argument("--forecast-csv")
    parser.add_argument("--export-forecast-csv")
    parser.add_argument("--domains", type=int, default=4)
    parser.add_argument("--slots", type=int, default=24)
    parser.add_argument("--tasks", type=int, default=5)
    parser.add_argument("--trace-task-limit", type=int, default=200)
    parser.add_argument("--slot-seconds", type=int, default=3600)
    parser.add_argument("--deadline-slack", type=float, default=1.5)
    parser.add_argument("--target-utilization", type=float, default=0.68)
    parser.add_argument("--trace-sample-policy", choices=["random", "earliest", "latest", "peak"], default="peak")
    parser.add_argument("--trace-window-index", type=int, default=0)
    parser.add_argument("--trace-windows", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--algorithms", default="robust_mpc_alns,min_grid,edf,source_affinity,least_loaded,greedy")
    parser.add_argument("--alns-iterations", type=int, default=300)
    parser.add_argument("--output-root", default="results/scenarios")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    algorithms = [item.strip() for item in cli_args.algorithms.split(",") if item.strip()]
    runner = ExperimentRunner(
        algorithms=algorithms,
        seed=cli_args.seed,
        output_root=cli_args.output_root,
        alns_iterations=cli_args.alns_iterations,
    )
    if cli_args.workload == "synthetic":
        runner.run_all_scenarios()
    else:
        runner.run_trace_workload(cli_args)
