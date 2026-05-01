import numpy as np

from .base import BaseScheduler

try:
    import cvxpy as cp
except ImportError:
    cp = None


class OptimizationScheduler(BaseScheduler):
    """Convex optimization baseline. Falls back gracefully when cvxpy is absent."""

    def __init__(self, problem, beta=1.0, lambda1=0.1, lambda2=1.0):
        super().__init__(problem)
        self.beta = beta
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def solve(self):
        if cp is None:
            from .greedy_scheduler import GreedyScheduler

            result = GreedyScheduler(self.problem).solve()
            result["status"] = "fallback_no_cvxpy"
            result["algorithm"] = "optimization_fallback_greedy"
            return result

        D = self.problem.D
        T = self.problem.T
        J = self.problem.J
        R_hat = self.problem.R_hat
        Cap = self.problem.Cap
        Base = self.problem.Base
        BW_tot = self.problem.BW_tot
        tasks = self.problem.tasks
        alpha = self.problem.alpha
        E_base = self.problem.E_base

        x = cp.Variable((J, D, T), nonneg=True)
        g = cp.Variable((D, T), nonneg=True)
        u = cp.Variable(J, nonneg=True)
        m = cp.Variable((J, T), nonneg=True)

        objective = cp.Minimize(
            cp.sum(g)
            + self.lambda1 * self.beta * cp.sum(m[:, 1:])
            + self.lambda2 * cp.sum(cp.multiply([task["pi_j"] for task in tasks], u))
        )

        constraints = []
        for j, task in enumerate(tasks):
            a_j = int(task["a_j"]) - 1
            d_j = int(task["d_j"]) - 1
            constraints.append(cp.sum(x[j, :, a_j : d_j + 1]) + u[j] == task["W_j"])
            if a_j > 0:
                constraints.append(x[j, :, :a_j] == 0)
            if d_j + 1 < T:
                constraints.append(x[j, :, d_j + 1 :] == 0)

        for d in range(D):
            for t in range(T):
                constraints.append(cp.sum(x[:, d, t]) <= Cap[d, t] - Base[d, t])
                energy = E_base[d, t] + alpha[d] * cp.sum(x[:, d, t])
                constraints.append(energy <= R_hat[d, t] + g[d, t])

        for j, task in enumerate(tasks):
            a_j = int(task["a_j"]) - 1
            S_j = task["S_j"]
            W_j = max(task["W_j"], 1e-9)
            cum_x = cp.cumsum(x[j, :, a_j:], axis=1)
            for t in range(1, T - a_j):
                delta = cum_x[:, t] - cum_x[:, t - 1]
                constraints.append(m[j, a_j + t] >= (S_j / W_j) * 0.5 * cp.sum(cp.abs(delta)))

        for t in range(1, T):
            constraints.append(cp.sum(m[:, t]) <= BW_tot[t])

        problem = cp.Problem(objective, constraints)
        problem.solve()

        if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return {
                "status": problem.status,
                "x": np.maximum(x.value, 0.0),
                "g": np.maximum(g.value, 0.0),
                "u": np.maximum(u.value, 0.0),
                "m": np.maximum(m.value, 0.0),
                "objective": float(problem.value),
                "algorithm": "optimization",
            }
        return {"status": problem.status, "objective": None, "algorithm": "optimization"}
