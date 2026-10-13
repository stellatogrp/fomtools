"""Small wrappers for PEPit comparison experiments."""

from __future__ import annotations

import math


class PEPitRunner:
    """Run the PEPit bounds used by the comparison examples."""

    def __init__(self, solver: str = "CLARABEL", verbose: bool = False):
        self.solver = solver
        self.verbose = verbose

    def proximal_gradient_residuals(
        self,
        *,
        K: int,
        L: float,
        eta: float,
        R: float,
        nonsmooth: str = "indicator",
    ) -> tuple[list[float], float]:
        """Return PEPit residual bounds for proximal gradient.

        ``nonsmooth`` is ``"indicator"`` for projected gradient and ``"convex"``
        for generic composite objectives such as lasso.
        """
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
        R: float,
        nonsmooth: str = "indicator",
        variant: str = "simple",
    ) -> tuple[list[float], float]:
        """Return PEPit residual bounds for FISTA-style proximal gradient.

        ``variant`` selects which accelerated recurrence the PEP models, so that
        the bound applies to the method the corresponding tutorial actually runs:

        ``"simple"``
            ``beta_j = (j - 1) / (j + 2)``, with the residual measured on the
            *extrapolated* iterates ``z^j - z^(j-1)``. This is the momentum
            variant of ``pgd_step`` used by the NNLS tutorial.
        ``"fista"``
            the classical sequence ``t_j = (1 + sqrt(1 + 4 t_(j-1)^2)) / 2`` with
            ``t_0 = 1`` and ``beta_j = (t_(j-1) - 1) / t_j``, with the residual
            measured on the *proximal* iterates ``y^j - y^(j-1)``. This is the
            momentum variant of ``ista_step`` used by the Lasso tutorial.
        """
        residuals = [
            self._accelerated_proximal_gradient_residual(
                k=k, L=L, eta=eta, R=R, nonsmooth=nonsmooth, variant=variant)
            for k in range(1, K + 1)
        ]
        return residuals, R

    def admm_residuals(
        self,
        *,
        K: int,
        mu: float,
        L: float,
        R: float,
    ) -> tuple[list[float], float]:
        """Return PEPit residual bounds for the DRS/ADMM template."""
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
        R: float,
    ) -> tuple[list[float], float]:
        """Return PEPit residual bounds for the PDHG template."""
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
        R: float,
    ) -> tuple[list[float], float]:
        """Return PEPit residual bounds for the momentum PDHG template."""
        residuals = [
            self._pdhg_momentum_residual(k=k, L=L, eta=eta, R=R)
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
        self,
        *,
        k: int,
        L: float,
        eta: float,
        R: float,
        nonsmooth: str,
        variant: str = "simple",
    ) -> float:
        from PEPit import PEP
        from PEPit.functions import (
            ConvexFunction,
            ConvexIndicatorFunction,
            SmoothConvexFunction,
        )
        from PEPit.primitive_steps import proximal_step

        if variant not in ("simple", "fista"):
            raise ValueError("variant must be 'simple' or 'fista'.")

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

        # Both tutorials start from y^0 = z^0 = s^0, so the single initial point
        # above covers the extrapolated and the proximal sequence alike.
        z = [z0]
        w = [z0]
        t_prev = 1.0
        for j in range(1, k + 1):
            w_next, _, _ = proximal_step(
                z[j - 1] - eta * f.gradient(z[j - 1]), g, eta)
            if variant == "simple":
                beta_j = (j - 1) / (j + 2)
            else:
                t_next = (1.0 + math.sqrt(1.0 + 4.0 * t_prev ** 2)) / 2.0
                beta_j = (t_prev - 1.0) / t_next
                t_prev = t_next
            z_next = w_next + beta_j * (w_next - w[j - 1])
            w.append(w_next)
            z.append(z_next)

        residual = z[k] - z[k - 1] if variant == "simple" else w[k] - w[k - 1]
        problem.set_performance_metric(residual ** 2)
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
