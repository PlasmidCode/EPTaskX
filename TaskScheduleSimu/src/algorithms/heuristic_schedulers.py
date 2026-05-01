import numpy as np

from .base import BaseScheduler
from utils.schedule_metrics import EPS, evaluate_schedule


class ListHeuristicScheduler(BaseScheduler):
    """Shared list-scheduling implementation for transparent baselines."""

    def __init__(self, problem, policy, algorithm, seed=42):
        super().__init__(problem)
        self.policy = policy
        self.algorithm = algorithm
        self.rng = np.random.default_rng(seed)

    def solve(self):
        p = self.problem
        x = np.zeros((p.J, p.D, p.T), dtype=float)
        used = np.zeros((p.D, p.T), dtype=float)
        link_usage = np.zeros((p.D, p.D, p.T), dtype=float)

        for j in self._task_order():
            task = p.tasks[j]
            remaining = float(task["W_j"])
            current_domain = int(task["src_j"]) - 1
            last_migration = -1
            for t in self._slot_order(task):
                if remaining <= EPS:
                    break
                candidate = self._best_candidate(
                    x=x,
                    used=used,
                    link_usage=link_usage,
                    j=j,
                    t=t,
                    remaining=remaining,
                    current_domain=current_domain,
                    last_migration=last_migration,
                )
                if candidate is None:
                    continue
                d, work, migration_size = candidate
                x[j, d, t] = work
                used[d, t] += work
                remaining -= work
                if migration_size > EPS:
                    link_usage[current_domain, d, t] += migration_size
                    current_domain = d
                    last_migration = t

        eval_data = evaluate_schedule(p, x, risk_mode="mean")
        status = "optimal"
        if eval_data["metrics"]["capacity_violation"] > EPS or eval_data["metrics"]["link_violation"] > EPS:
            status = "feasible_with_penalty"
        return {
            "status": status,
            "x": x,
            "g": eval_data["g"],
            "u": eval_data["u"],
            "m": eval_data["m"],
            "objective": eval_data["objective"],
            "algorithm": self.algorithm,
            "battery_soc": eval_data["battery_soc"],
            "clean_used": eval_data["clean_used"],
            "curtailment": eval_data["curtailment"],
            "link_migration": eval_data["link_migration"],
            "metrics": eval_data["metrics"],
        }

    def _task_order(self):
        tasks = self.problem.tasks
        if self.policy in ("edf", "min_grid"):
            return sorted(range(self.problem.J), key=lambda j: (tasks[j]["d_j"], -tasks[j]["pi_j"], tasks[j]["a_j"]))
        if self.policy == "random":
            order = np.arange(self.problem.J)
            self.rng.shuffle(order)
            return list(order)
        return sorted(range(self.problem.J), key=lambda j: (tasks[j]["a_j"], tasks[j]["d_j"], -tasks[j]["pi_j"]))

    def _slot_order(self, task):
        slots = list(range(int(task["a_j"]) - 1, int(task["d_j"])))
        if self.policy == "random":
            self.rng.shuffle(slots)
        return slots

    def _best_candidate(self, x, used, link_usage, j, t, remaining, current_domain, last_migration):
        p = self.problem
        task = p.tasks[j]
        src = int(task["src_j"]) - 1
        if self.policy == "source_affinity":
            domains = [src]
        else:
            domains = list(range(p.D))
            if self.policy == "random":
                self.rng.shuffle(domains)

        best = None
        for d in domains:
            if not self._task_can_use_domain(task, d, src):
                continue
            available = p.Cap[d, t] - p.Base[d, t] - used[d, t]
            if available <= EPS:
                continue
            migration_size = float(task["S_j"]) if d != current_domain else 0.0
            if migration_size > EPS:
                if p.migration_cooldown and last_migration >= 0 and t - last_migration <= p.migration_cooldown:
                    continue
                if link_usage[current_domain, d, t] + migration_size > p.BW_link[current_domain, d, t] + 1e-7:
                    continue
            work = min(float(remaining), float(available))
            score = self._score(x, used, j, d, t, work, current_domain, migration_size)
            if best is None or score < best[0]:
                best = (score, d, work, migration_size)
        if best is None:
            return None
        _, d, work, migration_size = best
        return d, work, migration_size

    def _score(self, x, used, j, d, t, work, current_domain, migration_size):
        p = self.problem
        task = p.tasks[j]
        load_before = p.E_base[d, t] + p.alpha[d] * np.sum(x[:, d, t])
        load_after = load_before + p.alpha[d] * work
        expected_clean = p.R_mean[d, t]
        robust_clean = p.R_lower[d, t]
        grid_delta = max(load_after - expected_clean, 0.0) - max(load_before - expected_clean, 0.0)
        robust_grid_delta = max(load_after - robust_clean, 0.0) - max(load_before - robust_clean, 0.0)
        clean_headroom = max(expected_clean - load_before, 0.0)
        utilization = (used[d, t] + work) / max(p.Cap[d, t] - p.Base[d, t], EPS)
        deadline_pressure = 1.0 / max(int(task["d_j"]) - t, 1)

        if self.policy == "least_loaded":
            return utilization + 0.01 * migration_size
        if self.policy == "edf":
            return grid_delta + 0.08 * migration_size - 0.03 * task["pi_j"] * deadline_pressure
        if self.policy == "min_grid":
            uncertainty = max(p.R_mean[d, t] - p.R_lower[d, t], 0.0) / max(p.R_mean[d, t], 1.0)
            return (
                grid_delta
                + 0.35 * robust_grid_delta
                + 0.12 * migration_size
                + 0.05 * uncertainty * work
                - 0.02 * task["pi_j"] * deadline_pressure
            )
        if self.policy == "random":
            return self.rng.random()
        return -clean_headroom + grid_delta + 0.05 * migration_size

    def _task_can_use_domain(self, task, d, src):
        task_type = str(task.get("type", "migratable")).lower()
        if task_type in ("fixed", "non_migratable", "non-migratable"):
            return d == src
        return True


class CleanGreedyScheduler(ListHeuristicScheduler):
    def __init__(self, problem, seed=42):
        super().__init__(problem, policy="clean_greedy", algorithm="greedy", seed=seed)


class SourceAffinityScheduler(ListHeuristicScheduler):
    def __init__(self, problem, seed=42):
        super().__init__(problem, policy="source_affinity", algorithm="source_affinity", seed=seed)


class EarliestDeadlineFirstScheduler(ListHeuristicScheduler):
    def __init__(self, problem, seed=42):
        super().__init__(problem, policy="edf", algorithm="edf", seed=seed)


class MinGridScheduler(ListHeuristicScheduler):
    def __init__(self, problem, seed=42):
        super().__init__(problem, policy="min_grid", algorithm="min_grid", seed=seed)


class LeastLoadedScheduler(ListHeuristicScheduler):
    def __init__(self, problem, seed=42):
        super().__init__(problem, policy="least_loaded", algorithm="least_loaded", seed=seed)


class RandomScheduler(ListHeuristicScheduler):
    def __init__(self, problem, seed=42):
        super().__init__(problem, policy="random", algorithm="random", seed=seed)
