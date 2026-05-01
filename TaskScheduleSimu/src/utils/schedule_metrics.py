import numpy as np


GRID_EMISSION_FACTOR_KG_PER_KWH = 0.57
EPS = 1e-8


def evaluate_schedule(problem, x, risk_mode="mean"):
    """Evaluate a concrete schedule with common scientific metrics."""
    x = np.asarray(x, dtype=float)
    migration = compute_migration(problem, x)
    mean_energy = simulate_energy(problem, x, problem.R_mean)
    robust_energy = simulate_energy(problem, x, problem.R_lower)
    selected_energy = robust_energy if risk_mode in ("p10", "lower", "robust") else mean_energy

    u = unfinished_work(problem, x)
    capacity_limit = np.maximum(problem.Cap - problem.Base, 0.0)
    capacity_excess = np.maximum(np.sum(x, axis=0) - capacity_limit, 0.0)
    link_excess = np.maximum(migration["link_migration"] - problem.BW_link, 0.0)
    total_work = float(np.sum([task["W_j"] for task in problem.tasks]))
    completed_work = max(total_work - float(np.sum(u)), 0.0)
    weighted_unfinished = float(np.sum([problem.tasks[j]["pi_j"] * u[j] for j in range(problem.J)]))
    load = selected_energy["load"]

    total_available_clean = float(np.sum(problem.R_mean))
    expected_clean_absorbed = float(np.sum(mean_energy["renewable_absorbed"]))
    capacity_utilization = np.divide(
        np.sum(x, axis=0),
        np.maximum(capacity_limit, EPS),
        out=np.zeros_like(capacity_limit),
        where=capacity_limit > EPS,
    )
    link_mask = problem.BW_link > EPS
    link_utilization = np.divide(
        migration["link_migration"],
        np.maximum(problem.BW_link, EPS),
        out=np.zeros_like(problem.BW_link),
        where=link_mask,
    )
    forecast_width = problem.R_upper - problem.R_lower
    forecast_uncertainty = np.divide(
        forecast_width,
        np.maximum(problem.R_mean, 1.0),
        out=np.zeros_like(problem.R_mean),
        where=problem.R_mean > EPS,
    )
    risk_exposure = float(np.sum(forecast_uncertainty * load))
    objective = (
        problem.weights["grid"] * float(np.sum(selected_energy["g"]))
        + problem.weights["migration"] * float(np.sum(migration["m"]))
        + problem.weights["sla"] * weighted_unfinished
        + problem.weights["curtailment"] * float(np.sum(selected_energy["curtailment"]))
        + problem.weights["risk"] * risk_exposure
        + 1e5 * float(np.sum(capacity_excess))
        + 1e5 * float(np.sum(link_excess))
    )

    metrics = {
        "objective_common": float(objective),
        "grid_energy": float(np.sum(selected_energy["g"])),
        "expected_grid_energy": float(np.sum(mean_energy["g"])),
        "robust_grid_energy": float(np.sum(robust_energy["g"])),
        "robustness_gap": float(np.sum(robust_energy["g"]) - np.sum(mean_energy["g"])),
        "grid_emissions_kgco2e": float(np.sum(selected_energy["g"]) * GRID_EMISSION_FACTOR_KG_PER_KWH),
        "peak_grid_energy": float(np.max(selected_energy["g"])) if selected_energy["g"].size else 0.0,
        "migration": float(np.sum(migration["m"])),
        "migration_events": float(np.sum(migration["m"] > EPS)),
        "completed_work": completed_work,
        "total_work": total_work,
        "total_unfinished_work": float(np.sum(u)),
        "weighted_unfinished_work": weighted_unfinished,
        "completion_rate": completed_work / total_work if total_work > EPS else 1.0,
        "deadline_miss_rate": float(np.mean(u > EPS)) if problem.J else 0.0,
        "finished_task_rate": float(np.mean(u <= EPS)) if problem.J else 1.0,
        "clean_energy_absorbed": expected_clean_absorbed,
        "clean_utilization": expected_clean_absorbed / total_available_clean if total_available_clean > EPS else 1.0,
        "cfe_share": _cfe_share(mean_energy["load"], mean_energy["g"]),
        "curtailment": float(np.sum(mean_energy["curtailment"])),
        "curtailment_rate": float(np.sum(mean_energy["curtailment"]) / total_available_clean)
        if total_available_clean > EPS
        else 0.0,
        "capacity_violation": float(np.sum(capacity_excess)),
        "avg_capacity_utilization": float(np.mean(capacity_utilization)) if capacity_utilization.size else 0.0,
        "peak_capacity_utilization": float(np.max(capacity_utilization)) if capacity_utilization.size else 0.0,
        "link_violation": float(np.sum(link_excess)),
        "mean_link_utilization": float(np.mean(link_utilization[link_mask])) if np.any(link_mask) else 0.0,
        "peak_link_utilization": float(np.max(link_utilization[link_mask])) if np.any(link_mask) else 0.0,
        "forecast_risk_exposure": risk_exposure,
        "forecast_interval_width_mean": float(np.mean(forecast_width)) if forecast_width.size else 0.0,
        "renewable_deficit_rate_p10": float(np.mean(mean_energy["load"] > problem.R_lower)),
        "energy_per_work": float(np.sum(selected_energy["g"]) / completed_work) if completed_work > EPS else 0.0,
        "migration_per_work": float(np.sum(migration["m"]) / completed_work) if completed_work > EPS else 0.0,
        "risk_mode": risk_mode,
    }
    return {
        "objective": float(objective),
        "g": selected_energy["g"],
        "u": u,
        "m": migration["m"],
        "link_migration": migration["link_migration"],
        "battery_soc": selected_energy["battery_soc"],
        "clean_used": selected_energy["clean_used"],
        "curtailment": selected_energy["curtailment"],
        "charge": selected_energy["charge"],
        "discharge": selected_energy["discharge"],
        "metrics": metrics,
    }


def simulate_energy(problem, x, renewable):
    """Greedy battery dispatch for a fixed schedule and renewable profile."""
    load = problem.E_base + problem.alpha[:, None] * np.sum(x, axis=0)
    g = np.zeros((problem.D, problem.T), dtype=float)
    clean_used = np.zeros((problem.D, problem.T), dtype=float)
    renewable_absorbed = np.zeros((problem.D, problem.T), dtype=float)
    curtailment = np.zeros((problem.D, problem.T), dtype=float)
    charge = np.zeros((problem.D, problem.T), dtype=float)
    discharge = np.zeros((problem.D, problem.T), dtype=float)
    battery_soc = np.zeros((problem.D, problem.T + 1), dtype=float)
    battery_soc[:, 0] = problem.battery["initial_soc"]

    for d in range(problem.D):
        for t in range(problem.T):
            soc = battery_soc[d, t]
            demand = load[d, t]
            available_clean = renewable[d, t]
            if demand > available_clean:
                deficit = demand - available_clean
                deliverable = min(
                    problem.battery["max_discharge"][d],
                    soc * problem.battery["discharge_efficiency"][d],
                )
                discharge[d, t] = min(deficit, deliverable)
                g[d, t] = max(deficit - discharge[d, t], 0.0)
                clean_used[d, t] = max(demand - g[d, t], 0.0)
                renewable_absorbed[d, t] = available_clean
            else:
                surplus = available_clean - demand
                charge[d, t] = min(
                    problem.battery["max_charge"][d],
                    surplus,
                    max(problem.battery["capacity"][d] - soc, 0.0)
                    / problem.battery["charge_efficiency"][d],
                )
                curtailment[d, t] = max(surplus - charge[d, t], 0.0)
                clean_used[d, t] = demand + charge[d, t]
                renewable_absorbed[d, t] = demand + charge[d, t]
            battery_soc[d, t + 1] = np.clip(
                soc
                + charge[d, t] * problem.battery["charge_efficiency"][d]
                - discharge[d, t] / problem.battery["discharge_efficiency"][d],
                0.0,
                problem.battery["capacity"][d],
            )

    return {
        "load": load,
        "g": g,
        "clean_used": clean_used,
        "renewable_absorbed": renewable_absorbed,
        "curtailment": curtailment,
        "charge": charge,
        "discharge": discharge,
        "battery_soc": battery_soc,
    }


def compute_migration(problem, x):
    m = np.zeros((problem.J, problem.T), dtype=float)
    link_migration = np.zeros((problem.D, problem.D, problem.T), dtype=float)
    events = {}
    for j, task in enumerate(problem.tasks):
        current = int(task["src_j"]) - 1
        for t in range(problem.T):
            d = assigned_domain(x, j, t)
            if d < 0:
                continue
            if d != current:
                size = float(task["S_j"])
                m[j, t] += size
                link_migration[current, d, t] += size
                events.setdefault((current, d, t), []).append(j)
                current = d
    return {"m": m, "link_migration": link_migration, "events": events}


def unfinished_work(problem, x):
    u = np.zeros(problem.J, dtype=float)
    for j, task in enumerate(problem.tasks):
        a_j = int(task["a_j"]) - 1
        d_j = int(task["d_j"]) - 1
        completed = float(np.sum(x[j, :, a_j : d_j + 1]))
        u[j] = max(float(task["W_j"]) - completed, 0.0)
    return u


def assigned_domain(x, j, t):
    row = x[j, :, t]
    if np.sum(row) <= EPS:
        return -1
    return int(np.argmax(row))


def _cfe_share(load, grid):
    total_load = float(np.sum(load))
    if total_load <= EPS:
        return 1.0
    return float((total_load - np.sum(grid)) / total_load)
