import csv
from pathlib import Path

import numpy as np


REQUIRED_COLUMNS = {"site_id", "slot", "r_mean_kwh"}
OPTIONAL_COLUMNS = {"r_p10_kwh", "r_p90_kwh"}


def load_forecast_csv(path, D=None, T=None, site_ids=None, lower_fallback_ratio=0.85):
    """
    Load renewable generation forecasts for the scheduler.

    Expected columns:
        site_id, slot, r_mean_kwh, r_p10_kwh, r_p90_kwh

    ``r_p10_kwh`` and ``r_p90_kwh`` are optional. Missing lower forecasts are
    filled with ``r_mean_kwh * lower_fallback_ratio``; missing upper forecasts
    are filled with ``max(r_mean_kwh, r_p10_kwh)``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"forecast CSV not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"forecast CSV missing required columns: {sorted(missing)}")
        rows = list(reader)

    if not rows:
        raise ValueError("forecast CSV is empty")

    discovered_sites = []
    for row in rows:
        site = row["site_id"]
        if site not in discovered_sites:
            discovered_sites.append(site)

    if site_ids is None:
        site_ids = discovered_sites
    else:
        site_ids = list(site_ids)
    if D is None:
        D = len(site_ids)
    if len(site_ids) != D:
        raise ValueError(f"site_ids length {len(site_ids)} does not match D={D}")

    raw_slots = [int(float(row["slot"])) for row in rows]
    if T is None:
        if min(raw_slots) == 0:
            T = max(raw_slots) + 1
        else:
            T = max(raw_slots)

    slot_offset = 0 if min(raw_slots) == 0 else 1
    site_to_idx = {site: idx for idx, site in enumerate(site_ids)}
    R_mean = np.full((D, T), np.nan, dtype=float)
    R_lower = np.full((D, T), np.nan, dtype=float)
    R_upper = np.full((D, T), np.nan, dtype=float)

    for row in rows:
        site = row["site_id"]
        if site not in site_to_idx:
            continue
        slot = int(float(row["slot"])) - slot_offset
        if not (0 <= slot < T):
            raise ValueError(f"slot {row['slot']} normalizes to {slot}, outside [0, {T - 1}]")
        d = site_to_idx[site]
        mean = _nonnegative_float(row["r_mean_kwh"], "r_mean_kwh")
        lower = _optional_nonnegative_float(row.get("r_p10_kwh"))
        upper = _optional_nonnegative_float(row.get("r_p90_kwh"))
        R_mean[d, slot] = mean
        R_lower[d, slot] = lower if lower is not None else mean * lower_fallback_ratio
        R_upper[d, slot] = upper if upper is not None else max(mean, R_lower[d, slot])

    for name, arr in [("R_mean", R_mean), ("R_lower", R_lower), ("R_upper", R_upper)]:
        missing = np.argwhere(np.isnan(arr))
        if missing.size:
            d, t = missing[0]
            raise ValueError(f"forecast CSV missing {name} value for site={site_ids[d]}, slot={t}")

    metadata = [{"site_id": site, "name": site} for site in site_ids]
    return {
        "D": D,
        "T": T,
        "R_hat": R_mean,
        "R_mean": R_mean,
        "R_lower": np.minimum(R_lower, R_mean),
        "R_upper": np.maximum(R_upper, R_mean),
        "site_metadata": metadata,
    }


def write_forecast_csv(path, R_mean, R_lower=None, R_upper=None, site_ids=None, one_based_slots=False):
    """Write simulator forecasts in the standard CSV shape."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    R_mean = np.asarray(R_mean, dtype=float)
    D, T = R_mean.shape
    R_lower = np.asarray(R_lower if R_lower is not None else R_mean * 0.85, dtype=float)
    R_upper = np.asarray(R_upper if R_upper is not None else np.maximum(R_mean, R_lower), dtype=float)
    if site_ids is None:
        site_ids = [f"site_{d + 1}" for d in range(D)]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["site_id", "slot", "r_mean_kwh", "r_p10_kwh", "r_p90_kwh"],
        )
        writer.writeheader()
        for d, site in enumerate(site_ids):
            for t in range(T):
                writer.writerow(
                    {
                        "site_id": site,
                        "slot": t + 1 if one_based_slots else t,
                        "r_mean_kwh": f"{R_mean[d, t]:.6f}",
                        "r_p10_kwh": f"{R_lower[d, t]:.6f}",
                        "r_p90_kwh": f"{R_upper[d, t]:.6f}",
                    }
                )


def _nonnegative_float(value, name):
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric, got {value!r}") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative, got {parsed}")
    return parsed


def _optional_nonnegative_float(value):
    if value is None or value == "":
        return None
    return _nonnegative_float(value, "optional forecast quantile")
