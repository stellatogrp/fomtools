"""Small wrappers for PEPit comparison experiments."""

from __future__ import annotations

import math

import numpy as np


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

    def osqp_admm_residuals(
        self,
        *,
        K: int,
        mu: float,
        L: float,
        sigma: float,
        rho: float,
        R: float | None = None,
        calibrate_to: float | None = None,
    ) -> tuple[list[float], float]:
        """Return PEPit bounds for the OSQP-style ADMM recurrence with ``M = I``.

        The performance metric is the squared Euclidean fixed-point residual
        ``||z^k-z^{k-1}||^2 + ||v^k-v^{k-1}||^2``. The initial condition uses
        the corresponding OSQP state:
        ``||z^0-z^*||^2 + ||v^0-v^*||^2 <= R^2``.
        """
        if sigma < 0:
            raise ValueError("sigma must be nonnegative.")
        if rho <= 0:
            raise ValueError("rho must be positive.")

        if calibrate_to is None and R is None:
            raise ValueError("Either R or calibrate_to must be provided.")

        # The PEP is homogeneous in the initial radius. Solving at unit radius
        # and scaling by R^2 is both cheaper and substantially better
        # conditioned when the OSQP dual state makes R large.
        unit_residuals = [
            self._osqp_admm_residual(
                k=k,
                mu=mu,
                L=L,
                sigma=sigma,
                rho=rho,
                R=1.0,
            )
            for k in range(1, K + 1)
        ]
        if calibrate_to is not None:
            R = math.sqrt(calibrate_to / unit_residuals[0])
        residuals = [value * R ** 2 for value in unit_residuals]
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

    def pdhg_momentum_residuals(
        self,
        *,
        K: int,
        L: float,
        eta: float,
        R: float | None = None,
        calibrate_to: float | None = None,
    ) -> tuple[list[float], float]:
        """Return PEPit residual bounds for the momentum PDHG template."""
        if calibrate_to is not None:
            unit_k1 = self._pdhg_momentum_residual(k=1, L=L, eta=eta, R=1.0)
            R = math.sqrt(calibrate_to / unit_k1)
        elif R is None:
            raise ValueError("Either R or calibrate_to must be provided.")

        residuals = [
            self._pdhg_momentum_residual(k=k, L=L, eta=eta, R=R)
            for k in range(1, K + 1)
        ]
        return residuals, R

    def pdhg_radius(
        self,
        *,
        A_s: np.ndarray,
        A_d: np.ndarray,
        b_s: np.ndarray,
        mu: np.ndarray,
        c: np.ndarray,
        eta: float,
        z0_samples: np.ndarray,
        v0_samples: np.ndarray,
        w0_samples: np.ndarray,
        x_samples: np.ndarray,
        max_iter: int = 2000,
        tol: float = 1e-10,
    ) -> float:
        """Estimate the PEPit initial-condition radius ``R`` for network PDHG.

        Runs the exact (NumPy) PDHG recursion from each sampled initial point
        to its own fixed point, then returns the largest distance between an
        initial sample and its fixed point. This gives an empirical ``R`` for
        calibrating :meth:`pdhg_residuals`, sized to the actual sampled
        initial-iterate and demand distributions rather than a generic bound.
        """
        z_star = z0_samples.copy()
        v_star = v0_samples.copy()
        w_star = w0_samples.copy()
        for _ in range(max_iter):
            z_rhs = z_star - eta * (
                mu[np.newaxis, :] + v_star @ A_s - w_star @ A_d
            )
            z_next = np.clip(z_rhs, 0.0, c)
            v_rhs = v_star + eta * (
                -b_s[np.newaxis, :] + (2 * z_next - z_star) @ A_s.T
            )
            v_next = np.maximum(v_rhs, 0.0)
            w_next = w_star + eta * (
                x_samples - (2 * z_next - z_star) @ A_d.T
            )
            step_norm = np.max(np.linalg.norm(
                np.hstack([z_next - z_star, v_next - v_star, w_next - w_star]),
                axis=1,
            ))
            z_star, v_star, w_star = z_next, v_next, w_next
            if step_norm <= tol:
                break

        return np.max(np.linalg.norm(
            np.hstack([
                z0_samples - z_star,
                v0_samples - v_star,
                w0_samples - w_star,
            ]),
            axis=1,
        ))

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

    def _osqp_admm_residual(
        self,
        *,
        k: int,
        mu: float,
        L: float,
        sigma: float,
        rho: float,
        R: float,
    ) -> float:
        from PEPit import PEP, Point
        from PEPit.expression import Expression
        from PEPit.functions import ConvexIndicatorFunction, SmoothStronglyConvexFunction
        from PEPit.primitive_steps import proximal_step

        problem = PEP()
        f = problem.declare_function(SmoothStronglyConvexFunction, mu=mu, L=L)
        g = problem.declare_function(ConvexIndicatorFunction)

        # Construct an OSQP fixed point. Stationarity gives
        # grad f(z*) + grad g(z*) = 0, and the scaled dual state is
        # v* = z* - grad f(z*) / rho.
        zs = Point()
        grad_fs, _ = f.oracle(zs)
        g.add_point((zs, -grad_fs, Expression()))
        g.add_point((zs, -grad_fs / rho, Expression()))
        vs = zs - grad_fs / rho

        z0 = problem.set_initial_point()
        v0 = problem.set_initial_point()
        problem.set_initial_condition(
            (z0 - zs) ** 2 + (v0 - vs) ** 2 <= R ** 2
        )

        z = [z0]
        v = [v0]
        gamma = 1.0 / (sigma + rho)
        for j in range(k):
            # Projection is prox_g; for an indicator its step size is immaterial.
            w_next, _, _ = proximal_step(v[j], g, 1.0)

            # This proximal step is exactly
            # (H + (sigma+rho)I) z^{k+1}
            #   = sigma z^k + rho(2w^{k+1}-v^k) - q
            # for a quadratic f(z) = 0.5 z^T H z + q^T z.
            prox_center = gamma * (
                sigma * z[j] + rho * (2 * w_next - v[j])
            )
            z_next, _, _ = proximal_step(prox_center, f, gamma)
            v_next = v[j] + z_next - w_next
            z.append(z_next)
            v.append(v_next)

        problem.set_performance_metric(
            (z[k] - z[k - 1]) ** 2 + (v[k] - v[k - 1]) ** 2
        )
        return problem.solve(solver=self.solver, verbose=self.verbose)

    def _pdhg_momentum_residual(self, *, k: int, L: float, eta: float, R: float) -> float:
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
        for j in range(1, k + 1):
            x_next, _, _ = proximal_step(
                x[-1] - eta * linear.T.gradient(u[-1]), f, eta)
            beta = (j - 1) / (j + 2)
            x_bar = x_next + beta * (x_next - x[-1])
            u_next, _, _ = proximal_step(
                u[-1] + eta * linear.gradient(2 * x_bar - x[-1]), g, eta)
            x.append(x_next)
            u.append(u_next)

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
