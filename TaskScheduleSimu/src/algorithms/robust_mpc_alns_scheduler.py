import math

import numpy as np
from scipy.optimize import linprog

from .base import BaseScheduler


class RobustMPCALNSScheduler(BaseScheduler):
    """
    Robust forecast-aware scheduler using an LP relaxation plus ALNS repair.

    The LP stage provides a clean-energy-aware guide under forecast lower
    bounds, storage dynamics and link budgets. ALNS then repairs the relaxed
    allocation into a practical schedule where a task uses at most one site per
    time slot and migration is charged on concrete source-destination links.
    """

    def __init__(
        self,
        problem,
        rolling_horizon=None,
        alns_iterations=300,
        destroy_ratio=0.25,
        seed=42,
        risk_mode="p10",
    ):
        super().__init__(problem)
        self.rolling_horizon = rolling_horizon or min(6, problem.T)
        self.alns_iterations = int(alns_iterations)
        self.destroy_ratio = float(destroy_ratio)
        self.seed = int(seed)
        self.risk_mode = risk_mode
        self.rng = np.random.default_rng(seed)
        self.eps = 1e-8

    def solve(self):
        relaxed = self._solve_relaxed_mpc()
        current = {"x": self._initial_solution(relaxed)}
        self._repair_all(current)
        self._make_feasible(current)
        current_eval = self._evaluate(current["x"])

        best = {"x": current["x"].copy()}
        best_eval = current_eval
        temperature = max(1.0, current_eval["objective"] * 0.03)
        destroy_ops = [
            self._destroy_deadline_critical,
            self._destroy_high_grid,
            self._destroy_high_migration,
            self._destroy_random_window,
        ]
        op_weights = np.ones(len(destroy_ops), dtype=float)

        for iteration in range(self.alns_iterations):
            probs = op_weights / np.sum(op_weights)
            op_idx = int(self.rng.choice(len(destroy_ops), p=probs))
            candidate = {"x": current["x"].copy()}
            removed = destroy_ops[op_idx](candidate)
            if removed == 0:
                continue

            self._repair_all(candidate)
            self._make_feasible(candidate)
            candidate_eval = self._evaluate(candidate["x"])
            delta = candidate_eval["objective"] - current_eval["objective"]
            accepted = delta <= 0 or self.rng.random() < math.exp(-delta / max(temperature, 1e-9))
            if accepted:
                current = candidate
                current_eval = candidate_eval
                op_weights[op_idx] += 0.5
            if candidate_eval["objective"] + self.eps < best_eval["objective"]:
                best = {"x": candidate["x"].copy()}
                best_eval = candidate_eval
                op_weights[op_idx] += 1.0
            temperature *= 0.992
            if iteration and iteration % 80 == 0:
                op_weights = 0.8 * op_weights + 0.2

        return self._build_result(best["x"], best_eval)

    def _solve_relaxed_mpc(self):
        p = self.problem
        D, T, J = p.D, p.T, p.J
        n_x = J * D * T
        n_g = D * T
        n_u = J
        n_charge = D * T
        n_discharge = D * T
        n_soc = D * (T + 1)
        n_curtail = D * T
        off_g = n_x
        off_u = off_g + n_g
        off_charge = off_u + n_u
        off_discharge = off_charge + n_charge
        off_soc = off_discharge + n_discharge
        off_curtail = off_soc + n_soc
        n_vars = off_curtail + n_curtail
        weights = p.weights
        R_eff = p.renewable(self.risk_mode)

        def xi(j, d, t):
            return (j * D + d) * T + t

        def gi(d, t):
            return off_g + d * T + t

        def ui(j):
            return off_u + j

        def ci(d, t):
            return off_charge + d * T + t

        def di(d, t):
            return off_discharge + d * T + t

        def si(d, t):
            return off_soc + d * (T + 1) + t

        def ki(d, t):
            return off_curtail + d * T + t

        c = np.zeros(n_vars, dtype=float)
        bounds = [(0.0, None) for _ in range(n_vars)]
        for j, task in enumerate(p.tasks):
            a_j = int(task["a_j"]) - 1
            d_j = int(task["d_j"]) - 1
            src = int(task["src_j"]) - 1
            for d in range(D):
                for t in range(T):
                    idx = xi(j, d, t)
                    if t < a_j or t > d_j or (not self._task_can_use_domain(task, d, src)):
                        bounds[idx] = (0.0, 0.0)
                    else:
                        uncertainty = max(p.R_mean[d, t] - p.R_lower[d, t], 0.0) / max(p.R_mean[d, t], 1.0)
                        migration_proxy = 0.0 if d == src else task["S_j"] / max(task["W_j"], 1e-9)
                        c[idx] = (
                            weights["risk"] * uncertainty * p.alpha[d]
                            + weights["migration"] * migration_proxy
                        )
            c[ui(j)] = weights["sla"] * task["pi_j"]
            bounds[ui(j)] = (0.0, float(task["W_j"]))

        for d in range(D):
            for t in range(T):
                c[gi(d, t)] = weights["grid"]
                c[ki(d, t)] = weights["curtailment"]
                bounds[ci(d, t)] = (0.0, float(p.battery["max_charge"][d]))
                bounds[di(d, t)] = (0.0, float(p.battery["max_discharge"][d]))
                bounds[si(d, t)] = (0.0, float(p.battery["capacity"][d]))
            bounds[si(d, T)] = (0.0, float(p.battery["capacity"][d]))

        A_eq = []
        b_eq = []
        for j, task in enumerate(p.tasks):
            row = np.zeros(n_vars, dtype=float)
            a_j = int(task["a_j"]) - 1
            d_j = int(task["d_j"]) - 1
            for d in range(D):
                for t in range(a_j, d_j + 1):
                    row[xi(j, d, t)] = 1.0
            row[ui(j)] = 1.0
            A_eq.append(row)
            b_eq.append(float(task["W_j"]))

        for d in range(D):
            row = np.zeros(n_vars, dtype=float)
            row[si(d, 0)] = 1.0
            A_eq.append(row)
            b_eq.append(float(p.battery["initial_soc"][d]))
            for t in range(T):
                row = np.zeros(n_vars, dtype=float)
                row[si(d, t + 1)] = 1.0
                row[si(d, t)] = -1.0
                row[ci(d, t)] = -float(p.battery["charge_efficiency"][d])
                row[di(d, t)] = 1.0 / float(p.battery["discharge_efficiency"][d])
                A_eq.append(row)
                b_eq.append(0.0)

        # Renewable balance: load + charge + curtail - discharge - grid = forecast.
        for d in range(D):
            for t in range(T):
                row = np.zeros(n_vars, dtype=float)
                for j in range(J):
                    row[xi(j, d, t)] = p.alpha[d]
                row[ci(d, t)] = 1.0
                row[ki(d, t)] = 1.0
                row[di(d, t)] = -1.0
                row[gi(d, t)] = -1.0
                A_eq.append(row)
                b_eq.append(float(R_eff[d, t] - p.E_base[d, t]))

        A_ub = []
        b_ub = []
        for d in range(D):
            for t in range(T):
                row = np.zeros(n_vars, dtype=float)
                for j in range(J):
                    row[xi(j, d, t)] = 1.0
                A_ub.append(row)
                b_ub.append(float(max(p.Cap[d, t] - p.Base[d, t], 0.0)))

        for src in range(D):
            for dst in range(D):
                if src == dst:
                    continue
                for t in range(T):
                    row = np.zeros(n_vars, dtype=float)
                    has_coeff = False
                    for j, task in enumerate(p.tasks):
                        if int(task["src_j"]) - 1 != src:
                            continue
                        row[xi(j, dst, t)] = task["S_j"] / max(task["W_j"], 1e-9)
                        has_coeff = True
                    if has_coeff:
                        A_ub.append(row)
                        b_ub.append(float(p.BW_link[src, dst, t]))

        try:
            result = linprog(
                c,
                A_ub=np.array(A_ub) if A_ub else None,
                b_ub=np.array(b_ub) if b_ub else None,
                A_eq=np.array(A_eq) if A_eq else None,
                b_eq=np.array(b_eq) if b_eq else None,
                bounds=bounds,
                method="highs",
            )
        except TypeError:
            result = linprog(
                c,
                A_ub=np.array(A_ub) if A_ub else None,
                b_ub=np.array(b_ub) if b_ub else None,
                A_eq=np.array(A_eq) if A_eq else None,
                b_eq=np.array(b_eq) if b_eq else None,
                bounds=bounds,
            )

        if not result.success:
            return np.zeros((J, D, T), dtype=float)

        relaxed = np.zeros((J, D, T), dtype=float)
        for j in range(J):
            for d in range(D):
                for t in range(T):
                    relaxed[j, d, t] = max(result.x[xi(j, d, t)], 0.0)
        return relaxed

    def _initial_solution(self, relaxed):
        p = self.problem
        x = np.zeros((p.J, p.D, p.T), dtype=float)
        used = np.zeros((p.D, p.T), dtype=float)
        task_order = sorted(range(p.J), key=lambda j: (p.tasks[j]["d_j"], -p.tasks[j]["pi_j"]))

        for j in task_order:
            task = p.tasks[j]
            remaining = float(task["W_j"])
            assigned_slot = set()
            candidates = []
            for d in range(p.D):
                for t in range(int(task["a_j"]) - 1, int(task["d_j"])):
                    if relaxed[j, d, t] > self.eps:
                        candidates.append((relaxed[j, d, t], d, t))
            candidates.sort(reverse=True)
            for amount, d, t in candidates:
                if remaining <= self.eps:
                    break
                if t in assigned_slot:
                    continue
                available = max(p.Cap[d, t] - p.Base[d, t] - used[d, t], 0.0)
                assign = min(remaining, available, amount)
                if assign <= self.eps:
                    continue
                x[j, d, t] = assign
                used[d, t] += assign
                assigned_slot.add(t)
                remaining -= assign
        return x

    def _repair_all(self, state):
        p = self.problem
        for _ in range(max(p.J * p.T * 2, 1)):
            best = None
            for j, task in enumerate(p.tasks):
                remaining = self._remaining_work(state["x"], j)
                if remaining <= self.eps:
                    continue
                candidates = self._top_candidates_for_task(state["x"], j, remaining, limit=2)
                if not candidates:
                    continue
                primary = candidates[0]
                secondary_cost = candidates[1]["cost"] if len(candidates) > 1 else primary["cost"] + 100.0
                urgency = task["pi_j"] / max(int(task["d_j"]) - int(task["a_j"]) + 1, 1)
                regret = secondary_cost - primary["cost"] + urgency
                if best is None or regret > best["regret"]:
                    best = {"regret": regret, "candidate": primary}
            if best is None:
                break
            self._apply_candidate(state["x"], best["candidate"])

    def _top_candidates_for_task(self, x, j, remaining, limit=2):
        p = self.problem
        task = p.tasks[j]
        src = int(task["src_j"]) - 1
        link_usage = self._compute_migration(x)["link_migration"]
        candidates = []
        for t in range(int(task["a_j"]) - 1, int(task["d_j"])):
            assigned_domain = self._assigned_domain(x, j, t)
            domain_candidates = [assigned_domain] if assigned_domain >= 0 else range(p.D)
            prev_domain = assigned_domain if assigned_domain >= 0 else self._previous_domain(x, j, t)
            last_migration = self._last_migration_slot(x, j, t)
            for d in domain_candidates:
                if not self._task_can_use_domain(task, d, src):
                    continue
                available = p.Cap[d, t] - p.Base[d, t] - np.sum(x[:, d, t])
                if available <= self.eps:
                    continue
                migration_size = task["S_j"] if d != prev_domain else 0.0
                if migration_size > 0:
                    if p.migration_cooldown and last_migration >= 0 and t - last_migration <= p.migration_cooldown:
                        continue
                    if link_usage[prev_domain, d, t] + migration_size > p.BW_link[prev_domain, d, t] + 1e-7:
                        continue
                work = min(remaining, available)
                cost = self._candidate_cost(x, j, d, t, work, prev_domain)
                candidates.append(
                    {
                        "j": j,
                        "d": d,
                        "t": t,
                        "work": work,
                        "cost": cost,
                        "add": assigned_domain >= 0,
                    }
                )
        candidates.sort(key=lambda item: item["cost"])
        return candidates[:limit]

    def _apply_candidate(self, x, candidate):
        if candidate.get("add"):
            x[candidate["j"], candidate["d"], candidate["t"]] += candidate["work"]
        else:
            x[candidate["j"], candidate["d"], candidate["t"]] = candidate["work"]

    def _candidate_cost(self, x, j, d, t, work, prev_domain):
        p = self.problem
        weights = p.weights
        task = p.tasks[j]
        load_before = p.E_base[d, t] + p.alpha[d] * np.sum(x[:, d, t])
        load_after = load_before + p.alpha[d] * work
        R_eff = p.renewable(self.risk_mode)[d, t]
        grid_delta = max(load_after - R_eff, 0.0) - max(load_before - R_eff, 0.0)
        curtail_before = max(R_eff - load_before, 0.0)
        curtail_after = max(R_eff - load_after, 0.0)
        curtail_delta = curtail_after - curtail_before
        uncertainty = max(p.R_mean[d, t] - p.R_lower[d, t], 0.0) / max(p.R_mean[d, t], 1.0)
        migration = task["S_j"] if d != prev_domain else 0.0
        deadline_pressure = 1.0 / max(int(task["d_j"]) - t, 1)
        return (
            weights["grid"] * grid_delta
            + weights["migration"] * migration
            + weights["curtailment"] * curtail_delta
            + weights["risk"] * uncertainty * p.alpha[d] * work
            - 0.05 * task["pi_j"] * deadline_pressure * work
        ) / max(work, 1e-9)

    def _make_feasible(self, state):
        for _ in range(4):
            changed = self._trim_capacity_violations(state["x"])
            changed |= self._trim_link_violations(state["x"])
            self._repair_all(state)
            if not changed:
                break

    def _trim_capacity_violations(self, x):
        p = self.problem
        changed = False
        for d in range(p.D):
            for t in range(p.T):
                limit = p.Cap[d, t] - p.Base[d, t]
                overload = np.sum(x[:, d, t]) - limit
                if overload <= 1e-7:
                    continue
                tasks_here = [
                    (p.tasks[j]["pi_j"], j)
                    for j in range(p.J)
                    if x[j, d, t] > self.eps
                ]
                tasks_here.sort()
                for _, j in tasks_here:
                    cut = min(overload, x[j, d, t])
                    x[j, d, t] -= cut
                    overload -= cut
                    changed = True
                    if overload <= 1e-7:
                        break
        return changed

    def _trim_link_violations(self, x):
        p = self.problem
        changed = False
        for _ in range(p.J * p.T + 1):
            migration = self._compute_migration(x)
            excess = migration["link_migration"] - p.BW_link
            if np.max(excess) <= 1e-7:
                break
            src, dst, t = np.unravel_index(np.argmax(excess), excess.shape)
            offenders = migration["events"].get((src, dst, t), [])
            if not offenders:
                break
            offenders.sort(key=lambda j: p.tasks[j]["pi_j"])
            j = offenders[0]
            x[j, :, t] = 0.0
            changed = True
        return changed

    def _destroy_deadline_critical(self, state):
        p = self.problem
        cells = []
        for j, task in enumerate(p.tasks):
            slack = int(task["d_j"]) - int(task["a_j"]) + 1
            for t in range(int(task["a_j"]) - 1, int(task["d_j"])):
                d = self._assigned_domain(state["x"], j, t)
                if d >= 0:
                    cells.append((slack, -task["pi_j"], j, t))
        cells.sort()
        return self._remove_cells(state["x"], cells)

    def _destroy_high_grid(self, state):
        eval_data = self._evaluate(state["x"])
        cells = []
        for d in range(self.problem.D):
            for t in range(self.problem.T):
                if eval_data["g"][d, t] <= self.eps:
                    continue
                for j in range(self.problem.J):
                    if state["x"][j, d, t] > self.eps:
                        cells.append((eval_data["g"][d, t], j, t))
        cells.sort(reverse=True)
        return self._remove_cells(state["x"], cells)

    def _destroy_high_migration(self, state):
        migration = self._compute_migration(state["x"])
        cells = []
        for j in range(self.problem.J):
            for t in range(self.problem.T):
                if migration["m"][j, t] > self.eps:
                    cells.append((migration["m"][j, t], j, t))
        cells.sort(reverse=True)
        return self._remove_cells(state["x"], cells)

    def _destroy_random_window(self, state):
        p = self.problem
        cells = []
        if p.J == 0:
            return 0
        width = max(1, int(math.ceil(self.rolling_horizon / 2)))
        start = int(self.rng.integers(0, max(p.T - width + 1, 1)))
        for j in range(p.J):
            for t in range(start, min(p.T, start + width)):
                if np.sum(state["x"][j, :, t]) > self.eps:
                    cells.append((self.rng.random(), j, t))
        cells.sort()
        return self._remove_cells(state["x"], cells)

    def _remove_cells(self, x, ranked_cells):
        target = max(1, int(math.ceil(self.destroy_ratio * max(self.problem.J, 1) * max(self.problem.T, 1))))
        removed = 0
        seen = set()
        for cell in ranked_cells:
            j = int(cell[-2])
            t = int(cell[-1])
            if (j, t) in seen:
                continue
            seen.add((j, t))
            if np.sum(x[j, :, t]) <= self.eps:
                continue
            x[j, :, t] = 0.0
            removed += 1
            if removed >= target:
                break
        return removed

    def _remaining_work(self, x, j):
        task = self.problem.tasks[j]
        a_j = int(task["a_j"]) - 1
        d_j = int(task["d_j"]) - 1
        completed = np.sum(x[j, :, a_j : d_j + 1])
        return max(float(task["W_j"]) - completed, 0.0)

    def _assigned_domain(self, x, j, t):
        row = x[j, :, t]
        if np.sum(row) <= self.eps:
            return -1
        return int(np.argmax(row))

    def _previous_domain(self, x, j, t):
        for prev_t in range(t - 1, -1, -1):
            d = self._assigned_domain(x, j, prev_t)
            if d >= 0:
                return d
        return int(self.problem.tasks[j]["src_j"]) - 1

    def _last_migration_slot(self, x, j, t):
        prev_domain = int(self.problem.tasks[j]["src_j"]) - 1
        last = -1
        for slot in range(t):
            d = self._assigned_domain(x, j, slot)
            if d < 0:
                continue
            if d != prev_domain:
                last = slot
            prev_domain = d
        return last

    def _task_can_use_domain(self, task, d, src):
        task_type = str(task.get("type", "migratable")).lower()
        if task_type in ("fixed", "non_migratable", "non-migratable"):
            return d == src
        return True

    def _compute_migration(self, x):
        p = self.problem
        m = np.zeros((p.J, p.T), dtype=float)
        link_migration = np.zeros((p.D, p.D, p.T), dtype=float)
        events = {}
        for j, task in enumerate(p.tasks):
            current = int(task["src_j"]) - 1
            for t in range(p.T):
                d = self._assigned_domain(x, j, t)
                if d < 0:
                    continue
                if d != current:
                    size = float(task["S_j"])
                    m[j, t] += size
                    link_migration[current, d, t] += size
                    events.setdefault((current, d, t), []).append(j)
                    current = d
        return {"m": m, "link_migration": link_migration, "events": events}

    def _evaluate(self, x):
        p = self.problem
        weights = p.weights
        migration = self._compute_migration(x)
        m = migration["m"]
        link_migration = migration["link_migration"]
        R_eff = p.renewable(self.risk_mode)

        g = np.zeros((p.D, p.T), dtype=float)
        clean_used = np.zeros((p.D, p.T), dtype=float)
        curtailment = np.zeros((p.D, p.T), dtype=float)
        charge = np.zeros((p.D, p.T), dtype=float)
        discharge = np.zeros((p.D, p.T), dtype=float)
        battery_soc = np.zeros((p.D, p.T + 1), dtype=float)
        battery_soc[:, 0] = p.battery["initial_soc"]

        load = p.E_base + p.alpha[:, None] * np.sum(x, axis=0)
        for d in range(p.D):
            for t in range(p.T):
                soc = battery_soc[d, t]
                demand = load[d, t]
                renewable = R_eff[d, t]
                if demand > renewable:
                    deficit = demand - renewable
                    deliverable = min(
                        p.battery["max_discharge"][d],
                        soc * p.battery["discharge_efficiency"][d],
                    )
                    discharge[d, t] = min(deficit, deliverable)
                    g[d, t] = max(deficit - discharge[d, t], 0.0)
                    clean_used[d, t] = max(demand - g[d, t], 0.0)
                else:
                    surplus = renewable - demand
                    charge[d, t] = min(
                        p.battery["max_charge"][d],
                        surplus,
                        max(p.battery["capacity"][d] - soc, 0.0)
                        / p.battery["charge_efficiency"][d],
                    )
                    curtailment[d, t] = max(surplus - charge[d, t], 0.0)
                    clean_used[d, t] = demand + charge[d, t]
                battery_soc[d, t + 1] = np.clip(
                    soc
                    + charge[d, t] * p.battery["charge_efficiency"][d]
                    - discharge[d, t] / p.battery["discharge_efficiency"][d],
                    0.0,
                    p.battery["capacity"][d],
                )

        u = np.zeros(p.J, dtype=float)
        for j in range(p.J):
            u[j] = self._remaining_work(x, j)

        capacity_excess = np.maximum(np.sum(x, axis=0) - (p.Cap - p.Base), 0.0)
        link_excess = np.maximum(link_migration - p.BW_link, 0.0)
        risk = np.maximum(p.R_mean - p.R_lower, 0.0) / np.maximum(p.R_mean, 1.0) * load
        objective = (
            weights["grid"] * np.sum(g)
            + weights["migration"] * np.sum(m)
            + weights["sla"] * np.sum([p.tasks[j]["pi_j"] * u[j] for j in range(p.J)])
            + weights["curtailment"] * np.sum(curtailment)
            + weights["risk"] * np.sum(risk)
            + 1e5 * np.sum(capacity_excess)
            + 1e5 * np.sum(link_excess)
        )

        total_work = np.sum([task["W_j"] for task in p.tasks])
        completed = max(total_work - np.sum(u), 0.0)
        total_load = float(np.sum(load))
        renewable_absorbed = float(np.sum(clean_used))
        total_available_clean = renewable_absorbed + float(np.sum(curtailment))
        metrics = {
            "grid_energy": float(np.sum(g)),
            "migration": float(np.sum(m)),
            "completion_rate": float(completed / total_work) if total_work > 0 else 1.0,
            "clean_energy_absorbed": renewable_absorbed,
            "clean_utilization": float(renewable_absorbed / total_available_clean)
            if total_available_clean > 0
            else 1.0,
            "cfe_share": float((total_load - np.sum(g)) / total_load) if total_load > 0 else 1.0,
            "curtailment": float(np.sum(curtailment)),
            "curtailment_rate": float(np.sum(curtailment) / total_available_clean)
            if total_available_clean > 0
            else 0.0,
            "capacity_violation": float(np.sum(capacity_excess)),
            "link_violation": float(np.sum(link_excess)),
            "risk_mode": self.risk_mode,
        }
        return {
            "objective": float(objective),
            "g": g,
            "u": u,
            "m": m,
            "link_migration": link_migration,
            "battery_soc": battery_soc,
            "clean_used": clean_used,
            "curtailment": curtailment,
            "charge": charge,
            "discharge": discharge,
            "metrics": metrics,
        }

    def _build_result(self, x, eval_data):
        status = "optimal"
        if eval_data["metrics"]["capacity_violation"] > 1e-6 or eval_data["metrics"]["link_violation"] > 1e-6:
            status = "feasible_with_penalty"
        return {
            "status": status,
            "x": x,
            "g": eval_data["g"],
            "u": eval_data["u"],
            "m": eval_data["m"],
            "objective": eval_data["objective"],
            "algorithm": "robust_mpc_alns",
            "battery_soc": eval_data["battery_soc"],
            "clean_used": eval_data["clean_used"],
            "curtailment": eval_data["curtailment"],
            "link_migration": eval_data["link_migration"],
            "metrics": eval_data["metrics"],
            "risk_mode": self.risk_mode,
        }
