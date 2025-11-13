from typing import Tuple

import cvxpy as cp
import numpy as np
import scipy.sparse as spa


def douglas_rachford(
    n: int, A: np.ndarray, b: np.ndarray, m: int, P: np.ndarray, c: np.ndarray
) -> None:
    """
    Solve the Douglas–Rachford algorithm using CVXPY.
    """
    x = cp.Variable(n)
    s = cp.Variable(m)
    obj = 0.5 * cp.quad_form(x, P) + c.T @ x
    constraints = [A @ x + s == b, s >= 0]
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver=cp.CLARABEL)
    print("optimal x from cp:", x.value)
    print("optimal s from cp:", s.value)
    print("optimal y from cp:", constraints[1].dual_value)


def fixed_point(
    n: int, A: np.ndarray, b: np.ndarray, l: float = -np.inf, u: float = np.inf
) -> np.ndarray:
    """
    Solve a bounded least-squares optimization problem using CVXPY.
    """
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(A @ x - b)), constraints=[x >= l, x <= u])
    prob.solve(solver=cp.CLARABEL)
    return x.value


def ista(n: int, A: np.ndarray, b: np.ndarray, lambd: int) -> np.ndarray:
    """
    Solve the LASSO regression problem using CVXPY.
    """
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(A @ x - b) + lambd * cp.norm(x, 1)))
    prob.solve(solver=cp.CLARABEL)
    return x.value


def pdhg(n: int, A: np.ndarray, b: np.ndarray, c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve the Primal-Dual hybrid gradient method using CVXPY.
    """
    x = cp.Variable(n)
    constraints = [A @ x == b, x >= 0]
    prob = cp.Problem(cp.Minimize(c @ x), constraints)
    prob.solve(solver=cp.CLARABEL)
    return x.value, constraints[0].dual_value


def mincost_flow(
    n: int,
    c: np.ndarray,
    A_block: spa.csc_array,
    b_tilde: np.ndarray,
    A_supply: spa.csc_array = None,
    b_supply: spa.csc_array = None,
    u: float = np.inf,
) -> Tuple[np.ndarray, float]:
    """Solve the minimum cost-flow problem using CVXPY."""
    x = cp.Variable(n)
    obj = cp.Minimize(c.T @ x)

    A_supply = np.zeros(n) if A_supply is None else A_supply
    b_supply = np.zeros(n) if b_supply is None else b_supply

    constraints = [A_supply @ x <= b_supply, A_block @ x == b_tilde, x >= 0, x <= u]
    prob = cp.Problem(obj, constraints)
    res = prob.solve(solver=cp.CLARABEL)
    return x.value, res


def partial_relu(n: int, proj_ranges: Tuple[int, int], P: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Solve the partial ReLU problem using CVXPY."""
    x = cp.Variable(n)
    obj = 0.5 * cp.quad_form(x, P) + q.T @ x
    constraints = [x[proj_ranges[0] : proj_ranges[1]] >= 0]
    prob = cp.Problem(cp.Minimize(obj), constraints)

    prob.solve(solver=cp.CLARABEL)
    return x.value


def satlin(
    n: int, P: np.ndarray, q: np.ndarray, l: np.ndarray, u: np.ndarray
) -> Tuple[np.ndarray, float]:
    """Solve the Saturated Linear Unit problem using CVXPY."""
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, P) + q.T @ x), [x >= l, x <= u])
    res = prob.solve(solver=cp.CLARABEL)

    return x.value, res


def verifier(n: int, c: np.ndarray, A: np.ndarray, b: np.ndarray, x_test: np.ndarray):
    """Solve the standard form linear programming problem using CVXPY."""
    x = cp.Variable(n)
    problem = cp.Problem(cp.Minimize(c.T @ x), [A @ x == b, x >= 0])
    res = problem.solve(solver=cp.CLARABEL)
    print(f"x_test: {x_test}")
    print(f"optimal obj: {res}")
    print(f"optimal x: {x.value}")


def portfolio_1(
    n: int,
    mu: np.ndarray,
    gamma: int,
    Sigma: np.ndarray,
    lambd: float,
    x_prev: np.ndarray,
    d: int,
    D: np.ndarray,
    F: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    w1 = cp.Variable(n)
    obj = mu.T @ w1 - gamma * cp.quad_form(w1, Sigma) - lambd * cp.sum_squares(w1 - x_prev)
    constraints = [cp.sum(w1) == 1, w1 >= 0]
    prob = cp.Problem(cp.Maximize(obj), constraints)
    prob.solve(solver=cp.CLARABEL)

    # reformed
    w2 = cp.Variable(n)
    y = cp.Variable(d)
    obj = (
        cp.quad_form(w2, gamma * D + lambd * np.eye(n))
        + gamma * cp.sum_squares(y)
        - (mu + 2 * lambd * x_prev) @ w2
    )
    constraints = [cp.sum(w2) == 1, w2 >= 0, y == F.T @ w2]
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver=cp.CLARABEL)

    return w1.value, w2.value


def portfolio_2(
    n: int,
    mu: np.ndarray,
    gamma: float,
    P: np.ndarray,
    lambd: float,
    x_prev: np.ndarray,
    d: int,
    F: np.ndarray,
    b: np.ndarray,
    A: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x1 = cp.Variable(n + d)

    c1 = np.concatenate([mu, np.zeros(d)])  # linear term for returns
    c2 = np.concatenate([2 * lambd * x_prev, np.zeros(d)])
    obj = -c1.T @ x1 + gamma * cp.quad_form(x1, P) - c2.T @ x1

    A1 = np.block([[F.T, -np.eye(d)]])
    b1 = np.zeros(d)

    A2 = np.block([[np.ones((1, n)), np.zeros((1, d))]])
    b2 = 1

    A3 = np.block([[-np.eye(n), np.zeros((n, d))]])
    b3 = np.zeros(n)

    constraints_stage1 = [
        A1 @ x1 == b1,
        A2 @ x1 == b2,
        A3 @ x1 <= b3,
    ]
    prob1 = cp.Problem(cp.Minimize(obj), constraints_stage1)
    prob1.solve(solver=cp.CLARABEL)
    z_triple_block = x1.value[:n]

    s = cp.Variable(n + d + 1)
    constraints_stage2 = [
        A @ x1 + s == b,
        s[d + 1 :] >= 0,
        s[: d + 1] == 0,
    ]

    prob2 = cp.Problem(cp.Minimize(obj), constraints_stage2)
    prob2.solve(solver=cp.SCS)

    return x1.value[:n], x1.value, constraints_stage2[0].dual_value, z_triple_block


def portfolio_l1(
    n: int,
    mu: np.ndarray,
    gamma: int,
    P: np.ndarray,
    lambd: int,
    x_prev: np.ndarray,
    d: int,
    D: np.ndarray,
    F: np.ndarray,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = cp.Variable(n)
    obj = mu.T @ x - gamma * cp.quad_form(x, P) - lambd * cp.norm(x - x_prev, 1)
    constraints = [cp.sum(x) == 1, x >= 0, x <= 1.5 / n]
    prob = cp.Problem(cp.Maximize(obj), constraints)
    prob.solve(solver=cp.CLARABEL)
    x_orig = x.value

    # reformulated problem
    x2 = cp.Variable(2 * n + d)

    # block matrices
    A_top = [F.T, -np.eye(d), np.zeros((d, n))]
    A_budget = [np.ones((1, n)), np.zeros((1, d)), np.zeros((1, n))]
    A_abs_plus = [np.eye(n), np.zeros((n, d)), -np.eye(n)]
    A_abs_minus = [-np.eye(n), np.zeros((n, d)), -np.eye(n)]
    A_lower_bound = [-np.eye(n), np.zeros((n, d)), np.zeros((n, n))]
    A_upper_bound = [np.eye(n), np.zeros((n, d)), np.zeros((n, n))]

    A = np.block([A_top, A_budget, A_abs_plus, A_abs_minus, A_lower_bound, A_upper_bound])

    rhs = np.hstack([np.zeros(d), 1, x_prev, -x_prev, np.zeros(n), 1.5 / n * np.ones(n)])
    slack = cp.Variable(4 * n + d + 1)

    q = np.hstack([-alpha * mu, np.zeros(d), alpha * lambd * np.ones(n)])
    P = 2 * np.block(
        [
            [D, np.zeros((n, d)), np.zeros((n, n))],
            [np.zeros((d, n)), np.eye(d), np.zeros((d, n))],
            [np.zeros((n, n)), np.zeros((n, d)), np.zeros((n, n))],
        ]
    )

    obj = 0.5 * cp.quad_form(x2, P) + q @ x2
    constraints = [
        A @ x2 + slack == rhs,
        slack[: d + 1] == 0,
        slack[d + 1 :] >= 0,
    ]

    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver=cp.CLARABEL)
    return A, P, q, rhs, x_orig, x2.value
