import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd


QOS_PRIORITY = {
    "LS": 2.2,
    "LatencySensitive": 2.2,
    "Burstable": 1.4,
    "BE": 0.8,
    "BestEffort": 0.8,
}


def load_alibaba_v2023_gpu_trace(
    pod_csv,
    node_csv=None,
    domains=4,
    slots=24,
    slot_seconds=3600,
    task_limit=200,
    seed=42,
    deadline_slack=1.5,
    target_utilization=0.68,
    gpu_weight=8.0,
    sample_policy="random",
    window_index=0,
):
    """
    Convert Alibaba cluster-trace-gpu-v2023 CSV files into TaskScheduleSimu params.

    Required pod columns follow ``openb_pod_list_default.csv``:
    name, cpu_milli, memory_mib, num_gpu, gpu_milli, qos, pod_phase,
    creation_time, deletion_time, scheduled_time.

    The node CSV is optional. When present, CPU/GPU capacities are aggregated
    into ``domains`` virtual sites; otherwise capacity is inferred from sampled
    workload demand.
    """
    rng = np.random.default_rng(seed)
    pod_df = pd.read_csv(pod_csv)
    required = {"name", "cpu_milli", "memory_mib", "creation_time"}
    _require_columns(pod_df, required, pod_csv)
    pod_df = _clean_v2023_pods(pod_df)
    if pod_df.empty:
        raise ValueError("No usable v2023 pod rows after filtering")
    pod_df["_work_hint"] = (
        pod_df["cpu_milli"].fillna(0.0) / 1000.0
        + gpu_weight * pod_df["num_gpu"].fillna(0.0) * np.maximum(pod_df["gpu_milli"].fillna(0.0), 0.0) / 1000.0
    )

    pod_df = _sample_window(
        pod_df,
        "creation_time",
        slots,
        slot_seconds,
        task_limit,
        rng,
        sample_policy=sample_policy,
        window_index=window_index,
        demand_col="_work_hint",
    )
    base_time = float(pod_df["creation_time"].min())
    tasks = []
    demand_hint = np.zeros(slots, dtype=float)

    for row in pod_df.itertuples(index=False):
        creation = float(getattr(row, "creation_time"))
        deletion = float(getattr(row, "deletion_time", creation + slot_seconds))
        if not np.isfinite(deletion) or deletion <= creation:
            deletion = creation + slot_seconds
        a_slot = _to_slot(creation, base_time, slot_seconds, slots)
        duration_slots = max(1, int(math.ceil((deletion - creation) / slot_seconds)))
        d_slot = min(slots, a_slot + max(1, int(math.ceil(duration_slots * deadline_slack))) - 1)

        cpu_cores = _value(row, "cpu_milli", 0.0) / 1000.0
        num_gpu = _value(row, "num_gpu", 0.0)
        gpu_milli = _value(row, "gpu_milli", 1000.0 if num_gpu > 0 else 0.0)
        gpu_units = num_gpu * max(gpu_milli, 0.0) / 1000.0
        work_per_slot = max(cpu_cores + gpu_weight * gpu_units, 0.05)
        W_j = work_per_slot * duration_slots
        mem_gb = max(_value(row, "memory_mib", 512.0) / 1024.0, 0.1)
        S_j = max(mem_gb * 0.25 + gpu_units * 2.0, 0.1)
        src_j = _stable_domain(getattr(row, "name"), domains) + 1
        pi_j = QOS_PRIORITY.get(str(getattr(row, "qos", "Burstable")), 1.0)
        task_type = "gpu" if gpu_units > 0 else "cpu"

        tasks.append(
            {
                "a_j": a_slot,
                "d_j": d_slot,
                "W_j": float(W_j),
                "S_j": float(S_j),
                "src_j": int(src_j),
                "pi_j": float(pi_j),
                "type": "migratable",
                "trace_id": str(getattr(row, "name")),
                "workload_type": task_type,
                "cpu_cores": float(cpu_cores),
                "gpu_units": float(gpu_units),
                "memory_gb": float(mem_gb),
            }
        )
        for t in range(a_slot - 1, min(slots, a_slot - 1 + duration_slots)):
            demand_hint[t] += work_per_slot

    node_capacity = None
    if node_csv:
        node_df = pd.read_csv(node_csv)
        _require_columns(node_df, {"sn", "cpu_milli", "memory_mib", "gpu"}, node_csv)
        raw = np.zeros(domains, dtype=float)
        for row in node_df.itertuples(index=False):
            d = _stable_domain(getattr(row, "sn"), domains)
            raw[d] += _value(row, "cpu_milli", 0.0) / 1000.0
            raw[d] += gpu_weight * _value(row, "gpu", 0.0)
        node_capacity = raw

    return _assemble_params(
        D=domains,
        T=slots,
        tasks=tasks,
        node_capacity=node_capacity,
        demand_hint=demand_hint,
        seed=seed,
        target_utilization=target_utilization,
        trace_name="alibaba_gpu_v2023",
        slot_length_hours=slot_seconds / 3600.0,
    )


def load_alibaba_v2018_batch_trace(
    batch_task_csv,
    machine_meta_csv=None,
    domains=4,
    slots=24,
    slot_seconds=3600,
    task_limit=200,
    seed=42,
    deadline_slack=1.5,
    target_utilization=0.68,
    sample_policy="random",
    window_index=0,
):
    """
    Convert Alibaba cluster-trace-v2018 batch task data into scheduler params.

    The official v2018 files are headerless; this loader also accepts headered
    CSVs with the same field names for small curated samples.
    """
    rng = np.random.default_rng(seed)
    task_df = _read_v2018_batch_task(batch_task_csv)
    task_df = _clean_v2018_tasks(task_df)
    if task_df.empty:
        raise ValueError("No usable v2018 batch task rows after filtering")
    task_df["_work_hint"] = (
        task_df["plan_cpu"].fillna(0.0) / 100.0 * np.maximum(task_df["instance_num"].fillna(1.0), 1.0)
    )

    task_df = _sample_window(
        task_df,
        "start_time",
        slots,
        slot_seconds,
        task_limit,
        rng,
        sample_policy=sample_policy,
        window_index=window_index,
        demand_col="_work_hint",
    )
    base_time = float(task_df["start_time"].min())
    tasks = []
    demand_hint = np.zeros(slots, dtype=float)

    for row in task_df.itertuples(index=False):
        start = float(getattr(row, "start_time"))
        end = float(getattr(row, "end_time"))
        a_slot = _to_slot(start, base_time, slot_seconds, slots)
        duration_slots = max(1, int(math.ceil((end - start) / slot_seconds)))
        d_slot = min(slots, a_slot + max(1, int(math.ceil(duration_slots * deadline_slack))) - 1)
        inst_num = max(_value(row, "instance_num", 1.0), 1.0)
        cpu_cores = max(_value(row, "plan_cpu", 10.0) / 100.0, 0.05)
        W_j = cpu_cores * inst_num * duration_slots
        mem_norm = max(_value(row, "plan_mem", 1.0), 0.1)
        S_j = max(mem_norm * inst_num * 0.05, 0.1)
        src_key = f"{getattr(row, 'job_name')}-{getattr(row, 'task_name')}"
        task_name = str(getattr(row, "task_name"))
        tasks.append(
            {
                "a_j": a_slot,
                "d_j": d_slot,
                "W_j": float(W_j),
                "S_j": float(S_j),
                "src_j": _stable_domain(src_key, domains) + 1,
                "pi_j": 1.2 if str(getattr(row, "task_type", "")).startswith("M") else 1.0,
                "type": "migratable",
                "trace_id": src_key,
                "task_name": task_name,
                "dependencies": _parse_v2018_dependencies(task_name),
                "cpu_cores": float(cpu_cores),
                "memory_norm": float(mem_norm),
                "instance_num": float(inst_num),
            }
        )
        for t in range(a_slot - 1, min(slots, a_slot - 1 + duration_slots)):
            demand_hint[t] += cpu_cores * inst_num

    node_capacity = None
    if machine_meta_csv:
        machine_df = _read_v2018_machine_meta(machine_meta_csv)
        machine_df = machine_df[machine_df["status"].astype(str).str.lower().isin(["using", "running", "available", "add", "normal"])]
        if machine_df.empty:
            machine_df = _read_v2018_machine_meta(machine_meta_csv)
        raw = np.zeros(domains, dtype=float)
        for row in machine_df.itertuples(index=False):
            key = getattr(row, "failure_domain_1", getattr(row, "machine_id"))
            d = _stable_domain(key, domains)
            raw[d] += max(_value(row, "cpu_num", 0.0), 0.0)
        node_capacity = raw

    return _assemble_params(
        D=domains,
        T=slots,
        tasks=tasks,
        node_capacity=node_capacity,
        demand_hint=demand_hint,
        seed=seed,
        target_utilization=target_utilization,
        trace_name="alibaba_batch_v2018",
        slot_length_hours=slot_seconds / 3600.0,
    )


def _assemble_params(D, T, tasks, node_capacity, demand_hint, seed, target_utilization, trace_name, slot_length_hours):
    rng = np.random.default_rng(seed)
    if node_capacity is None or np.sum(node_capacity) <= 0:
        base_cap = np.full(D, max(np.max(demand_hint), np.sum([task["W_j"] for task in tasks]) / max(T, 1), 1.0))
    else:
        base_cap = np.asarray(node_capacity, dtype=float)

    desired_total = max(float(np.max(demand_hint)) / max(target_utilization, 1e-6), D * 4.0)
    scale = desired_total / max(np.sum(base_cap), 1e-9)
    site_capacity = np.maximum(base_cap * scale, desired_total / D * 0.25)
    Cap = np.repeat(site_capacity[:, None], T, axis=1)
    Cap *= rng.uniform(0.92, 1.08, size=(D, T))
    Base = Cap * rng.uniform(0.10, 0.24, size=(D, T))

    alpha = rng.uniform(0.48, 0.76, size=D)
    E_base = np.maximum(Cap * rng.uniform(0.05, 0.10, size=(D, T)), 0.5)
    site_types = ["solar" if d % 2 == 0 else "wind" for d in range(D)]
    R_mean, R_lower, R_upper = _renewable_forecast(D, T, site_types, rng, site_capacity, alpha, E_base)
    BW_link = rng.uniform(15, 55, size=(D, D, T))
    for d in range(D):
        BW_link[d, d, :] = 0.0
    BW_tot = np.sum(BW_link, axis=(0, 1))
    battery = {
        "capacity": np.maximum(site_capacity * rng.uniform(0.25, 0.45, size=D), 5.0),
        "initial_soc": np.maximum(site_capacity * rng.uniform(0.08, 0.18, size=D), 1.0),
        "max_charge": np.maximum(site_capacity * rng.uniform(0.08, 0.16, size=D), 1.0),
        "max_discharge": np.maximum(site_capacity * rng.uniform(0.08, 0.16, size=D), 1.0),
        "charge_efficiency": np.full(D, 0.94),
        "discharge_efficiency": np.full(D, 0.94),
    }
    battery["initial_soc"] = np.minimum(battery["initial_soc"], battery["capacity"])
    site_metadata = [
        {
            "site_id": f"{trace_name}_{site_types[d]}_{d + 1}",
            "name": f"{trace_name} {site_types[d]} domain {d + 1}",
            "type": site_types[d],
            "trace": trace_name,
        }
        for d in range(D)
    ]
    return {
        "D": D,
        "T": T,
        "R_hat": R_mean,
        "R_mean": R_mean,
        "R_lower": R_lower,
        "R_upper": R_upper,
        "Cap": Cap,
        "Base": Base,
        "BW_tot": BW_tot,
        "BW_link": BW_link,
        "tasks": tasks,
        "alpha": alpha,
        "E_base": E_base,
        "battery": battery,
        "site_metadata": site_metadata,
        "migration_cooldown": 1,
        "slot_length_hours": float(slot_length_hours),
        "weights": {
            "grid": 1.0,
            "migration": 0.12,
            "sla": 18.0,
            "curtailment": 0.02,
            "risk": 0.06,
        },
        "trace_source": trace_name,
    }


def _renewable_forecast(D, T, site_types, rng, site_capacity, alpha, E_base):
    time = np.arange(T, dtype=float)
    R_mean = np.zeros((D, T), dtype=float)
    for d, site_type in enumerate(site_types):
        typical_dynamic_energy = site_capacity[d] * alpha[d]
        base_energy = float(np.mean(E_base[d]))
        renewable_scale = max(base_energy + typical_dynamic_energy * rng.uniform(0.28, 0.72), 1.0)
        if site_type == "solar":
            daylight = np.sin(np.pi * (time + 0.5) / max(T, 1))
            profile = np.maximum(daylight, 0.0) ** 1.6
            curve = renewable_scale * (0.08 + rng.uniform(0.85, 1.35) * profile)
        else:
            phase = rng.uniform(0, 2 * np.pi)
            profile = 0.58 + 0.32 * np.sin(2 * np.pi * time / max(T, 1) + phase)
            profile += 0.12 * np.sin(4 * np.pi * time / max(T, 1) + phase / 2)
            curve = renewable_scale * np.maximum(profile, 0.18)
        R_mean[d] = np.maximum(curve * (1 + rng.normal(0.0, 0.18, size=T)), 2.0)
    spread = np.maximum(R_mean * 0.22, 2.0)
    return R_mean, np.maximum(R_mean - spread, 0.0), R_mean + spread


def _clean_v2023_pods(df):
    out = df.copy()
    for col in ["deletion_time", "scheduled_time", "cpu_milli", "memory_mib", "num_gpu", "gpu_milli"]:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["creation_time"] = pd.to_numeric(out["creation_time"], errors="coerce")
    if "pod_phase" in out.columns:
        out = out[out["pod_phase"].astype(str).isin(["Succeeded", "Running", "Pending"])]
    out = out[np.isfinite(out["creation_time"])]
    out = out[out["cpu_milli"].fillna(0) > 0]
    out["deletion_time"] = out["deletion_time"].fillna(out["creation_time"] + 3600)
    out["memory_mib"] = out["memory_mib"].fillna(512)
    out["num_gpu"] = out["num_gpu"].fillna(0)
    out["gpu_milli"] = out["gpu_milli"].fillna(0)
    return out.sort_values("creation_time").reset_index(drop=True)


def _clean_v2018_tasks(df):
    out = df.copy()
    for col in ["start_time", "end_time", "plan_cpu", "plan_mem", "instance_num"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out[np.isfinite(out["start_time"]) & np.isfinite(out["end_time"])]
    out = out[out["end_time"] > out["start_time"]]
    out = out[out["plan_cpu"].fillna(0) > 0]
    if "status" in out.columns:
        out = out[out["status"].astype(str).str.lower().isin(["terminated", "running"])]
    out["instance_num"] = out["instance_num"].fillna(1)
    out["plan_mem"] = out["plan_mem"].fillna(1)
    return out.sort_values("start_time").reset_index(drop=True)


def _sample_window(
    df,
    time_col,
    slots,
    slot_seconds,
    task_limit,
    rng,
    sample_policy="random",
    window_index=0,
    demand_col=None,
):
    df = df.sort_values(time_col).reset_index(drop=True)
    if df.empty:
        return df
    min_time = float(df[time_col].min())
    max_time = float(df[time_col].max())
    window_seconds = max(slots * slot_seconds, slot_seconds)
    if max_time <= min_time + window_seconds:
        start_time = min_time
    else:
        max_start = max_time - window_seconds
        if sample_policy == "earliest":
            start_time = min_time
        elif sample_policy == "latest":
            start_time = max_start
        elif sample_policy == "peak":
            start_time = _peak_window_start(
                df,
                time_col,
                min_time,
                max_start,
                window_seconds,
                slot_seconds,
                demand_col,
                window_index,
            )
        else:
            candidate_count = max(int((max_start - min_time) // slot_seconds) + 1, 1)
            idx = int(rng.integers(0, candidate_count))
            start_time = min_time + idx * slot_seconds
    window = df[(df[time_col] >= start_time) & (df[time_col] < start_time + window_seconds)].copy()
    if window.empty:
        window = df.head(min(task_limit, len(df))).copy()
    if len(window) > task_limit:
        if sample_policy == "random":
            window = window.sample(n=task_limit, random_state=int(rng.integers(0, 2**31 - 1))).sort_values(time_col)
        else:
            indices = np.linspace(0, len(window) - 1, num=task_limit, dtype=int)
            window = window.iloc[indices]
    return window.reset_index(drop=True)


def _peak_window_start(df, time_col, min_time, max_start, window_seconds, slot_seconds, demand_col, window_index):
    starts = np.arange(min_time, max_start + 1e-9, slot_seconds, dtype=float)
    if len(starts) == 0:
        return min_time
    if demand_col and demand_col in df.columns:
        demand = df[demand_col].to_numpy(dtype=float)
    else:
        demand = np.ones(len(df), dtype=float)
    times = df[time_col].to_numpy(dtype=float)
    values = []
    for start in starts:
        mask = (times >= start) & (times < start + window_seconds)
        values.append(float(np.sum(demand[mask])))
    order = np.argsort(values)[::-1]
    chosen = order[min(max(int(window_index), 0), len(order) - 1)]
    return float(starts[chosen])


def _read_v2018_batch_task(path):
    columns = [
        "task_name",
        "instance_num",
        "job_name",
        "task_type",
        "status",
        "start_time",
        "end_time",
        "plan_cpu",
        "plan_mem",
    ]
    return _read_headered_or_headerless(path, columns)


def _read_v2018_machine_meta(path):
    columns = [
        "machine_id",
        "time_stamp",
        "failure_domain_1",
        "failure_domain_2",
        "cpu_num",
        "mem_size",
        "status",
    ]
    return _read_headered_or_headerless(path, columns)


def _read_headered_or_headerless(path, columns):
    path = Path(path)
    preview = path.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
    first = [part.strip() for part in preview.split(",")]
    if set(columns[: min(3, len(columns))]).intersection(first):
        return pd.read_csv(path)
    return pd.read_csv(path, names=columns, header=None)


def _parse_v2018_dependencies(task_name):
    parts = str(task_name).split("_")
    deps = []
    for item in parts[1:]:
        try:
            deps.append(int(item))
        except ValueError:
            continue
    return deps


def _require_columns(df, columns, path):
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")


def _to_slot(ts, base_time, slot_seconds, slots):
    return min(max(int(math.floor((ts - base_time) / slot_seconds)) + 1, 1), slots)


def _stable_domain(value, domains):
    digest = hashlib.md5(str(value).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % domains


def _value(row, name, default):
    value = getattr(row, name, default)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(parsed):
        return default
    return parsed
