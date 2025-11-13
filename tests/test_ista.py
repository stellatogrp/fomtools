import numpy as np

from tests.cvxpy import ista as ista_cvxpy

np.set_printoptions(precision=5)  # Print few decimal places
np.set_printoptions(suppress=True)  # Suppress scientific notation


def soft_threshold(x, gamma):
    return np.sign(x) * np.maximum(np.abs(x) - gamma, 0)


def test_convergence():
    m = 5
    n = 10
    K = 10000

    np.random.seed(0)
    A = np.random.normal(size=(m, n))
    b = np.random.uniform(size=(m,))

    t = 1 / np.max(np.linalg.eigvals(A.T @ A))
    lambd = 1e-2

    x_lstsq, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    print(x_lstsq)

    cvxpy = ista_cvxpy(n, A, b, lambd)
    print(cvxpy)

    At = np.eye(n) - t * A.T @ A
    Bt = t * A.T

    xk = np.zeros(n)
    lambda_t = lambd * t

    for _ in range(K):
        xk = soft_threshold(At @ xk + Bt @ b, lambda_t)

    print(xk)
    assert np.linalg.norm(xk - cvxpy) <= 1e-7

    print("--testing fista--")

    betak = 1.0
    xk = np.zeros(n)
    wk = xk
    for _ in range(K):
        xnew = soft_threshold(At @ wk + Bt @ b, lambda_t)
        beta_new = 0.5 * (1 + np.sqrt(1 + 4 * betak**2))
        wk = xnew + (betak - 1) / beta_new * (xnew - xk)

        xk = xnew
        betak = beta_new

    print(xk)
    assert np.linalg.norm(xk - cvxpy) <= 1e-7
