import numpy as np


class TestDataGenerator:
    """Generate reproducible forecast-driven scheduling scenarios."""

    def __init__(self, seed=42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate_problem(self, D=3, T=6, J=5, volatility=0.18, density=1.0):
        site_types = ["solar" if d % 2 == 0 else "wind" for d in range(D)]
        R_mean, R_lower, R_upper = self._renewable_forecast(D, T, site_types, volatility)

        Cap = self.rng.uniform(55, 95, size=(D, T))
        Base = self.rng.uniform(8, 20, size=(D, T))
        alpha = self.rng.uniform(0.45, 0.72, size=D)
        E_base = self.rng.uniform(8, 16, size=(D, T))

        BW_link = self.rng.uniform(25, 70, size=(D, D, T))
        for d in range(D):
            BW_link[d, d, :] = 0.0
        BW_tot = np.sum(BW_link, axis=(0, 1))
        BW_tot[0] = max(BW_tot[0], 30.0)

        tasks = []
        for j in range(J):
            a_j = int(self.rng.integers(1, max(2, T)))
            d_j = int(self.rng.integers(a_j, T + 1))
            if d_j == a_j and d_j < T:
                d_j += 1
            window = max(d_j - a_j + 1, 1)
            W_j = float(self.rng.uniform(35, 90) * density * (1 + 0.08 * window))
            S_j = float(self.rng.uniform(6, 28) * (1 + 0.08 * density))
            src_j = int(self.rng.integers(1, D + 1))
            pi_j = float(self.rng.uniform(0.8, 2.0))
            tasks.append(
                {
                    "a_j": a_j,
                    "d_j": d_j,
                    "W_j": W_j,
                    "S_j": S_j,
                    "src_j": src_j,
                    "pi_j": pi_j,
                    "type": "migratable",
                }
            )

        battery = {
            "capacity": self.rng.uniform(25, 55, size=D),
            "initial_soc": self.rng.uniform(8, 25, size=D),
            "max_charge": self.rng.uniform(10, 22, size=D),
            "max_discharge": self.rng.uniform(10, 22, size=D),
            "charge_efficiency": np.full(D, 0.94),
            "discharge_efficiency": np.full(D, 0.94),
        }
        battery["initial_soc"] = np.minimum(battery["initial_soc"], battery["capacity"])

        site_metadata = [
            {
                "site_id": f"{site_types[d]}_site_{d + 1}",
                "name": f"{site_types[d].capitalize()} Site {d + 1}",
                "type": site_types[d],
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
            "weights": {
                "grid": 1.0,
                "migration": 0.12,
                "sla": 15.0,
                "curtailment": 0.02,
                "risk": 0.05,
            },
        }

    def generate_scenarios(self):
        return [
            {"name": "small_scale", "params": self.generate_problem(D=2, T=4, J=3)},
            {"name": "medium_scale", "params": self.generate_problem(D=3, T=6, J=5)},
            {"name": "large_scale", "params": self.generate_problem(D=4, T=8, J=8)},
            {
                "name": "high_volatility",
                "params": self.generate_problem(D=3, T=6, J=5, volatility=0.34),
            },
            {
                "name": "high_density",
                "params": self.generate_problem(D=3, T=6, J=8, density=1.55),
            },
        ]

    def _renewable_forecast(self, D, T, site_types, volatility):
        time = np.arange(T, dtype=float)
        R_mean = np.zeros((D, T), dtype=float)
        for d, site_type in enumerate(site_types):
            if site_type == "solar":
                daylight = np.sin(np.pi * (time + 0.5) / max(T, 1))
                profile = np.maximum(daylight, 0.0) ** 1.7
                base = self.rng.uniform(35, 70)
                amplitude = self.rng.uniform(70, 130)
                curve = base + amplitude * profile
            else:
                phase = self.rng.uniform(0, 2 * np.pi)
                profile = 0.55 + 0.25 * np.sin(2 * np.pi * time / max(T, 1) + phase)
                profile += 0.18 * np.sin(4 * np.pi * time / max(T, 1) + phase / 2)
                base = self.rng.uniform(75, 120)
                curve = base * np.maximum(profile, 0.2)

            noise = self.rng.normal(0.0, volatility, size=T)
            R_mean[d] = np.maximum(curve * (1 + noise), 5.0)

        spread = np.maximum(R_mean * (0.12 + volatility), 3.0)
        R_lower = np.maximum(R_mean - spread, 0.0)
        R_upper = R_mean + spread
        return R_mean, R_lower, R_upper
