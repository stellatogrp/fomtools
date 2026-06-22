"""Small wrappers for PEPit comparison experiments."""

from __future__ import annotations

import math


class PEPitRunner:
    """Run the PEPit bounds used by the comparison examples."""

    def __init__(self, solver: str = "SCS", verbose: bool = False):
        self.solver = solver
        self.verbose = verbose

    def proximal_gradient_residuals(
        self,
        *,
        K: int,
        L: float,
        eta: float,
        R: float | None = None,
        calibrate_to: float | None = None,
        nonsmooth: str = "indicator",
    ) -> tuple[list[float], float]:
        """Return PEPit residual bounds for proximal gradient.

        ``nonsmooth`` is ``"indicator"`` for projected gradient and ``"convex"``
        for generic composite objectives such as lasso.
        """
        if calibrate_to is not None:
            unit_k1 = self._proximal_gradient_residual(
                k=1, L=L, eta=eta, R=1.0, nonsmooth=nonsmooth)
            R = math.sqrt(calibrate_to / unit_k1)
        elif R is None:
            raise ValueError("Either R or calibrate_to must be provided.")

        residuals = [
            self._proximal_gradient_residual(
                k=k, L=L, eta=eta, R=R, nonsmooth=nonsmooth)
            for k in range(1, K + 1)
        ]
        return residuals, R

    def accelerated_proximal_gradient_residuals(
        self,
        *,
        K: int,
        L: float,
        eta: float,
        R: float | None = None,
        calibrate_to: float | None = None,
        nonsmooth: str = "indicator",
    ) -> tuple[list[float], float]:
        """Return PEPit residual bounds for FISTA-style proximal gradient."""
        if calibrate_to is not None:
            unit_k1 = self._accelerated_proximal_gradient_residual(
                k=1, L=L, eta=eta, R=1.0, nonsmooth=nonsmooth)
            R = math.sqrt(calibrate_to / unit_k1)
        elif R is None:
            raise ValueError("Either R or calibrate_to must be provided.")

        residuals = [
            self._accelerated_proximal_gradient_residual(
                k=k, L=L, eta=eta, R=R, nonsmooth=nonsmooth)
            for k in range(1, K + 1)
        ]
        return residuals, R

    def admm_residuals(
        self,
        *,
        K: int,
        mu: float,
        L: float,
        R: float | None = None,
        calibrate_to: float | None = None,
    ) -> tuple[list[float], float]:
        """Return PEPit residual bounds for the DRS/ADMM template."""
        if calibrate_to is not None:
            unit_k1 = self._admm_residual(k=1, mu=mu, L=L, R=1.0)
            R = math.sqrt(calibrate_to / unit_k1)
        elif R is None:
            raise ValueError("Either R or calibrate_to must be provided.")

        residuals = [
            self._admm_residual(k=k, mu=mu, L=L, R=R)
            for k in range(1, K + 1)
        ]
        return residuals, R

    def pdhg_residuals(
        self,
        *,
        K: int,
        L: float,
        eta: float,
        R: float | None = None,
        calibrate_to: float | None = None,
    ) -> tuple[list[float], float]:
        """Return PEPit residual bounds for the PDHG template."""
        if calibrate_to is not None:
            unit_k1 = self._pdhg_residual(k=1, L=L, eta=eta, R=1.0)
            R = math.sqrt(calibrate_to / unit_k1)
        elif R is None:
            raise ValueError("Either R or calibrate_to must be provided.")

        residuals = [
            self._pdhg_residual(k=k, L=L, eta=eta, R=R)
            for k in range(1, K + 1)
        ]
        return residuals, R

    def _proximal_gradient_residual(
        self, *, k: int, L: float, eta: float, R: float, nonsmooth: str
    ) -> float:
        from PEPit import PEP
        from PEPit.functions import (
            ConvexFunction,
            ConvexIndicatorFunction,
            SmoothConvexFunction,
        )
        from PEPit.primitive_steps import proximal_step

        problem = PEP()
        f = problem.declare_function(SmoothConvexFunction, L=L)
        if nonsmooth == "indicator":
            g = problem.declare_function(ConvexIndicatorFunction)
        elif nonsmooth == "convex":
            g = problem.declare_function(ConvexFunction)
        else:
            raise ValueError("nonsmooth must be 'indicator' or 'convex'.")

        xs = (f + g).stationary_point()
        x0 = problem.set_initial_point()
        problem.set_initial_condition((x0 - xs) ** 2 <= R ** 2)

        x = [x0]
        for j in range(1, k + 1):
            x_next, _, _ = proximal_step(
                x[j - 1] - eta * f.gradient(x[j - 1]), g, eta)
            x.append(x_next)

        problem.set_performance_metric((x[k] - x[k - 1]) ** 2)
        return problem.solve(solver=self.solver, verbose=self.verbose)

    def _accelerated_proximal_gradient_residual(
        self, *, k: int, L: float, eta: float, R: float, nonsmooth: str
    ) -> float:
        from PEPit import PEP
        from PEPit.functions import (
            ConvexFunction,
            ConvexIndicatorFunction,
            SmoothConvexFunction,
        )
        from PEPit.primitive_steps import proximal_step

        problem = PEP()
        f = problem.declare_function(SmoothConvexFunction, L=L)
        if nonsmooth == "indicator":
            g = problem.declare_function(ConvexIndicatorFunction)
        elif nonsmooth == "convex":
            g = problem.declare_function(ConvexFunction)
        else:
            raise ValueError("nonsmooth must be 'indicator' or 'convex'.")

        xs = (f + g).stationary_point()
        z0 = problem.set_initial_point()
        problem.set_initial_condition((z0 - xs) ** 2 <= R ** 2)

        z = [z0]
        w = [z0]
        for j in range(1, k + 1):
            w_next, _, _ = proximal_step(
                z[j - 1] - eta * f.gradient(z[j - 1]), g, eta)
            beta_j = (j - 1) / (j + 2)
            z_next = w_next + beta_j * (w_next - w[j - 1])
            w.append(w_next)
            z.append(z_next)

        problem.set_performance_metric((z[k] - z[k - 1]) ** 2)
        return problem.solve(solver=self.solver, verbose=self.verbose)

    def _admm_residual(self, *, k: int, mu: float, L: float, R: float) -> float:
        from PEPit import PEP
        from PEPit.functions import ConvexIndicatorFunction, SmoothStronglyConvexFunction
        from PEPit.primitive_steps import proximal_step

        problem = PEP()
        f = problem.declare_function(SmoothStronglyConvexFunction, mu=mu, L=L)
        g = problem.declare_function(ConvexIndicatorFunction)
        xs = (f + g).stationary_point()
        us = xs + f.gradient(xs)
        u0 = problem.set_initial_point()

        x = [None]
        y = [None]
        u = [u0]
        x[0], _, _ = proximal_step(u[0], f, 1.0)
        y[0], _, _ = proximal_step(2 * x[0] - u[0], g, 1.0)

        problem.set_initial_condition((x[0] - xs) ** 2 + (u[0] - us) ** 2 <= R ** 2)

        for j in range(k):
            u.append(u[j] + (y[j] - x[j]))
            x_new, _, _ = proximal_step(u[j + 1], f, 1.0)
            y_new, _, _ = proximal_step(2 * x_new - u[j + 1], g, 1.0)
            x.append(x_new)
            y.append(y_new)

        problem.set_performance_metric(
            (x[k] - x[k - 1]) ** 2 + (u[k] - u[k - 1]) ** 2)
        return problem.solve(solver=self.solver, verbose=self.verbose)

    def _pdhg_residual(self, *, k: int, L: float, eta: float, R: float) -> float:
        from PEPit import PEP, Point
        from PEPit.functions import ConvexFunction
        from PEPit.operators import LinearOperator
        from PEPit.primitive_steps import proximal_step

        problem = PEP()
        f = problem.declare_function(ConvexFunction)
        g = problem.declare_function(ConvexFunction)
        linear = problem.declare_function(LinearOperator, L=L)

        xs = Point()
        us = Point()
        f.add_point((xs, -linear.T.gradient(us), f(xs)))
        g.add_point((us, linear.gradient(xs), g(us)))

        x0 = problem.set_initial_point()
        u0 = problem.set_initial_point()
        problem.set_initial_condition((x0 - xs) ** 2 + (u0 - us) ** 2 <= R ** 2)

        x = [x0]
        u = [u0]
        for _ in range(k):
            x_next, _, _ = proximal_step(
                x[-1] - eta * linear.T.gradient(u[-1]), f, eta)
            u_next, _, _ = proximal_step(
                u[-1] + eta * linear.gradient(2 * x_next - x[-1]), g, eta)
            x.append(x_next)
            u.append(u_next)

        problem.set_performance_metric(
            (x[k] - x[k - 1]) ** 2 + (u[k] - u[k - 1]) ** 2)
        return problem.solve(solver=self.solver, verbose=self.verbose)
