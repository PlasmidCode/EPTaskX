import numpy as np


class ProblemModel:
    """
    Forecast-driven cross-domain scheduling problem definition.

    The original simulator only required ``R_hat``. New fields are optional so
    older algorithms continue to run while robust schedulers can consume
    forecast intervals, storage parameters and link-level migration budgets.
    """

    DEFAULT_WEIGHTS = {
        "grid": 1.0,
        "migration": 0.1,
        "sla": 10.0,
        "curtailment": 0.02,
        "risk": 0.05,
    }

    def __init__(
        self,
        D,
        T,
        R_hat,
        Cap,
        Base,
        BW_tot,
        tasks,
        alpha,
        E_base,
        R_mean=None,
        R_lower=None,
        R_upper=None,
        site_metadata=None,
        battery=None,
        BW_link=None,
        migration_cooldown=0,
        weights=None,
        slot_length_hours=1.0,
    ):
        self.D = int(D)
        self.T = int(T)
        self.R_mean = self._matrix(R_mean if R_mean is not None else R_hat, "R_mean")
        self.R_lower = self._matrix(
            R_lower if R_lower is not None else self.R_mean * 0.85,
            "R_lower",
        )
        self.R_upper = self._matrix(
            R_upper if R_upper is not None else np.maximum(self.R_mean, self.R_lower),
            "R_upper",
        )
        self.R_hat = self.R_mean
        self.Cap = self._matrix(Cap, "Cap")
        self.Base = self._matrix(Base, "Base")
        self.BW_tot = np.asarray(BW_tot, dtype=float).reshape(self.T)
        self.tasks = [dict(task) for task in tasks]
        self.alpha = np.asarray(alpha, dtype=float).reshape(self.D)
        self.E_base = self._matrix(E_base, "E_base")
        self.J = len(self.tasks)
        self.site_metadata = site_metadata or [
            {"site_id": f"site_{d + 1}", "name": f"Site {d + 1}"}
            for d in range(self.D)
        ]
        self.battery = self._normalize_battery(battery)
        self.BW_link = self._normalize_bw_link(BW_link)
        self.migration_cooldown = int(migration_cooldown or 0)
        merged_weights = dict(self.DEFAULT_WEIGHTS)
        if weights:
            merged_weights.update(weights)
        self.weights = merged_weights
        self.slot_length_hours = float(slot_length_hours)
        self.validate()

    def _matrix(self, value, name):
        arr = np.asarray(value, dtype=float)
        if arr.shape != (self.D, self.T):
            raise ValueError(f"{name} must have shape ({self.D}, {self.T}), got {arr.shape}")
        return arr

    def _normalize_battery(self, battery):
        battery = battery or {}

        def vector(name, default):
            value = battery.get(name, default)
            arr = np.asarray(value, dtype=float)
            if arr.ndim == 0:
                arr = np.full(self.D, float(arr))
            if arr.shape != (self.D,):
                raise ValueError(f"battery['{name}'] must be scalar or shape ({self.D},)")
            return arr

        capacity = vector("capacity", 0.0)
        initial_soc = np.minimum(vector("initial_soc", 0.0), capacity)
        return {
            "capacity": capacity,
            "initial_soc": initial_soc,
            "max_charge": vector("max_charge", 0.0),
            "max_discharge": vector("max_discharge", 0.0),
            "charge_efficiency": np.clip(vector("charge_efficiency", 0.95), 1e-6, 1.0),
            "discharge_efficiency": np.clip(vector("discharge_efficiency", 0.95), 1e-6, 1.0),
        }

    def _normalize_bw_link(self, BW_link):
        if BW_link is None:
            link = np.zeros((self.D, self.D, self.T), dtype=float)
            for t in range(self.T):
                per_link = self.BW_tot[t] if self.D <= 1 else self.BW_tot[t] / max(self.D - 1, 1)
                for d in range(self.D):
                    for k in range(self.D):
                        if d != k:
                            link[d, k, t] = per_link
            return link
        arr = np.asarray(BW_link, dtype=float).copy()
        if arr.shape != (self.D, self.D, self.T):
            raise ValueError(f"BW_link must have shape ({self.D}, {self.D}, {self.T}), got {arr.shape}")
        for d in range(self.D):
            arr[d, d, :] = 0.0
        return arr

    def validate(self):
        for name, arr in [
            ("R_mean", self.R_mean),
            ("R_lower", self.R_lower),
            ("R_upper", self.R_upper),
            ("Cap", self.Cap),
            ("Base", self.Base),
            ("E_base", self.E_base),
        ]:
            if np.any(~np.isfinite(arr)):
                raise ValueError(f"{name} contains non-finite values")
            if np.any(arr < 0):
                raise ValueError(f"{name} contains negative values")
        if np.any(self.Cap - self.Base < -1e-9):
            raise ValueError("Cap must be greater than or equal to Base for every site and slot")
        if len(self.site_metadata) != self.D:
            raise ValueError("site_metadata length must match D")
        for idx, task in enumerate(self.tasks):
            for key in ["a_j", "d_j", "W_j", "S_j", "src_j", "pi_j"]:
                if key not in task:
                    raise ValueError(f"task {idx} missing required key '{key}'")
            if not (1 <= int(task["a_j"]) <= int(task["d_j"]) <= self.T):
                raise ValueError(f"task {idx} has invalid window [{task['a_j']}, {task['d_j']}]")
            if not (1 <= int(task["src_j"]) <= self.D):
                raise ValueError(f"task {idx} has invalid src_j {task['src_j']}")
            if task["W_j"] < 0 or task["S_j"] < 0 or task["pi_j"] < 0:
                raise ValueError(f"task {idx} has negative W_j, S_j or pi_j")

    def renewable(self, risk_mode="p10"):
        if risk_mode in ("p10", "lower", "robust"):
            return self.R_lower
        if risk_mode in ("p90", "upper"):
            return self.R_upper
        return self.R_mean

    def get_task_info(self, j):
        return self.tasks[j]

    def get_domain_info(self, d):
        return {
            "Cap": self.Cap[d],
            "Base": self.Base[d],
            "alpha": self.alpha[d],
            "E_base": self.E_base[d],
            "R_hat": self.R_hat[d],
            "R_mean": self.R_mean[d],
            "R_lower": self.R_lower[d],
            "R_upper": self.R_upper[d],
            "battery": {key: value[d] for key, value in self.battery.items()},
            "site_metadata": self.site_metadata[d],
        }
