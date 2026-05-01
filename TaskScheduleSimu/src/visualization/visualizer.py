import os

import matplotlib.pyplot as plt
import numpy as np

from utils.schedule_metrics import evaluate_schedule


class Visualizer:
    """Visualization and Markdown reporting for scheduling experiments."""

    def __init__(self, problem):
        self.problem = problem

    def visualize_results(self, result, output_dir="results"):
        if "x" not in result:
            return
        algo_dir = os.path.join(output_dir, result.get("algorithm", "unknown"))
        os.makedirs(algo_dir, exist_ok=True)
        self._visualize_task_allocation(result, algo_dir)
        self._visualize_grid_energy(result, algo_dir)
        self._visualize_migration(result, algo_dir)
        self._visualize_clean_energy_usage(result, algo_dir)
        if "battery_soc" in result:
            self._visualize_battery_soc(result, algo_dir)
        self._generate_algorithm_report(result, algo_dir)
        print(f"Saved result artifacts to {algo_dir}")

    def _visualize_task_allocation(self, result, output_dir):
        D, T, J = self.problem.D, self.problem.T, self.problem.J
        plt.figure(figsize=(12, 7))
        width = min(0.8 / max(D, 1), 0.28)
        for j in range(J):
            for d in range(D):
                values = result["x"][j, d, :]
                if np.max(values) <= 1e-9:
                    continue
                plt.bar(
                    np.arange(T) + d * width,
                    values,
                    width=width,
                    label=f"Task {j + 1} Site {d + 1}",
                )
        plt.xlabel("Time Slot")
        plt.ylabel("Execution Work")
        plt.title("Task Allocation")
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "task_allocation.png"), dpi=180)
        plt.close()

    def _visualize_grid_energy(self, result, output_dir):
        D, T = self.problem.D, self.problem.T
        plt.figure(figsize=(10, 5))
        for d in range(D):
            plt.plot(np.arange(T), result["g"][d, :], marker="o", label=f"Site {d + 1}")
        plt.xlabel("Time Slot")
        plt.ylabel("Grid Energy (kWh)")
        plt.title("Grid Energy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "grid_energy.png"), dpi=180)
        plt.close()

    def _visualize_migration(self, result, output_dir):
        J, T = self.problem.J, self.problem.T
        plt.figure(figsize=(10, 5))
        for j in range(J):
            plt.plot(np.arange(T), result["m"][j, :], marker="o", label=f"Task {j + 1}")
        plt.xlabel("Time Slot")
        plt.ylabel("Migration Data (GB)")
        plt.title("Task Migration")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "migration.png"), dpi=180)
        plt.close()

    def _visualize_clean_energy_usage(self, result, output_dir):
        D, T = self.problem.D, self.problem.T
        load = self.problem.E_base + self.problem.alpha[:, None] * np.sum(result["x"], axis=0)
        plt.figure(figsize=(11, 5))
        for d in range(D):
            plt.plot(np.arange(T), load[d, :], marker="o", label=f"Site {d + 1} Load")
            plt.plot(np.arange(T), self.problem.R_lower[d, :], linestyle="--", label=f"Site {d + 1} P10")
            plt.plot(np.arange(T), self.problem.R_mean[d, :], linestyle=":", label=f"Site {d + 1} Mean")
        plt.xlabel("Time Slot")
        plt.ylabel("Energy (kWh)")
        plt.title("Clean Forecast vs Scheduled Load")
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "clean_energy_usage.png"), dpi=180)
        plt.close()

    def _visualize_battery_soc(self, result, output_dir):
        soc = result["battery_soc"]
        plt.figure(figsize=(10, 5))
        for d in range(self.problem.D):
            plt.step(np.arange(soc.shape[1]), soc[d], where="post", label=f"Site {d + 1}")
        plt.xlabel("Time Slot")
        plt.ylabel("Battery SOC (kWh)")
        plt.title("Battery State of Charge")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "battery_soc.png"), dpi=180)
        plt.close()

    def _generate_algorithm_report(self, result, output_dir):
        metrics = self._metrics(result)
        lines = [
            f"# {result.get('algorithm', 'unknown')} scheduling report",
            "",
            "## Summary",
            f"- Status: {result.get('status', 'unknown')}",
            f"- Objective: {result.get('objective', float('nan')):.4f}",
            f"- Robustness mode: {metrics.get('risk_mode', 'mean')}",
            "",
            "## Metrics",
            f"- Grid energy: {metrics['grid_energy']:.4f} kWh",
            f"- Expected grid energy: {metrics.get('expected_grid_energy', metrics['grid_energy']):.4f} kWh",
            f"- P10 robust grid energy: {metrics.get('robust_grid_energy', metrics['grid_energy']):.4f} kWh",
            f"- Grid emissions: {metrics.get('grid_emissions_kgco2e', 0.0):.4f} kgCO2e",
            f"- Migration data: {metrics['migration']:.4f} GB",
            f"- Migration events: {metrics.get('migration_events', 0.0):.0f}",
            f"- Completion rate: {metrics['completion_rate']:.4f}",
            f"- Deadline miss rate: {metrics.get('deadline_miss_rate', 0.0):.4f}",
            f"- Clean energy utilization: {metrics['clean_utilization']:.4f}",
            f"- CFE share: {metrics['cfe_share']:.4f}",
            f"- Curtailment: {metrics['curtailment']:.4f} kWh",
            f"- Curtailment rate: {metrics['curtailment_rate']:.4f}",
            f"- Peak capacity utilization: {metrics.get('peak_capacity_utilization', 0.0):.4f}",
            f"- Peak link utilization: {metrics.get('peak_link_utilization', 0.0):.4f}",
            f"- Forecast risk exposure: {metrics.get('forecast_risk_exposure', 0.0):.4f}",
            f"- Capacity violation: {metrics.get('capacity_violation', 0.0):.6f}",
            f"- Link bandwidth violation: {metrics.get('link_violation', 0.0):.6f}",
            "",
            "## Figures",
            "- task_allocation.png",
            "- grid_energy.png",
            "- migration.png",
            "- clean_energy_usage.png",
        ]
        if "battery_soc" in result:
            lines.append("- battery_soc.png")
        with open(os.path.join(output_dir, "report.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def visualize_comparison(self, results, output_dir="results"):
        valid = [result for result in results if "x" in result]
        if not valid:
            return
        comparison_dir = os.path.join(output_dir, "comparison")
        os.makedirs(comparison_dir, exist_ok=True)
        algorithms = [result["algorithm"] for result in valid]
        metrics = [self._metrics(result) for result in valid]

        self._bar(algorithms, [result["objective"] for result in valid], "Objective", comparison_dir, "algorithm_comparison.png")
        self._bar(algorithms, [m["grid_energy"] for m in metrics], "Grid Energy (kWh)", comparison_dir, "grid_energy_comparison.png")
        self._bar(algorithms, [m["migration"] for m in metrics], "Migration Data (GB)", comparison_dir, "migration_comparison.png")
        self._bar(algorithms, [m["completion_rate"] for m in metrics], "Completion Rate", comparison_dir, "completion_rate_comparison.png")
        self._bar(algorithms, [m["clean_utilization"] for m in metrics], "Clean Utilization", comparison_dir, "clean_utilization_comparison.png")
        self._generate_comparison_report(valid, comparison_dir)
        print(f"Saved comparison artifacts to {comparison_dir}")

    def _bar(self, labels, values, ylabel, output_dir, filename):
        plt.figure(figsize=(10, 5))
        plt.bar(labels, values)
        plt.ylabel(ylabel)
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), dpi=180)
        plt.close()

    def _generate_comparison_report(self, results, output_dir):
        rows = []
        for result in results:
            m = self._metrics(result)
            rows.append(
                "| {algo} | {obj:.4f} | {grid:.4f} | {mig:.4f} | {comp:.4f} | {clean:.4f} | {curt:.4f} |".format(
                    algo=result["algorithm"],
                    obj=result["objective"],
                    grid=m["grid_energy"],
                    mig=m["migration"],
                    comp=m["completion_rate"],
                    clean=m["clean_utilization"],
                    curt=m["curtailment"],
                )
            )
        table = "\n".join(
            [
                "| Algorithm | Objective | Grid(kWh) | Migration(GB) | Completion | Clean utilization | Curtailment(kWh) |",
                "|------|----------|-----------|----------|--------|------------|-----------|",
                *rows,
            ]
        )
        with open(os.path.join(output_dir, "comparison_report.md"), "w", encoding="utf-8") as f:
            f.write(f"# Algorithm comparison report\n\n{table}\n")

    def _metrics(self, result):
        if "metrics" in result:
            metrics = dict(result["metrics"])
            defaults = evaluate_schedule(self.problem, result["x"], risk_mode=result.get("risk_mode", "mean"))["metrics"]
            defaults.update(metrics)
            metrics = defaults
            return metrics
        return evaluate_schedule(self.problem, result["x"], risk_mode=result.get("risk_mode", "mean"))["metrics"]
