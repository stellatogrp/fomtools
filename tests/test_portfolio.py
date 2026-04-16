import numpy as np

from tests.conftest import TEST_EPS
from tests.cvxpy import portfolio_1 as portfolio_1_cvxpy
from tests.cvxpy import portfolio_2 as portfolio_2_cvxpy
from tests.cvxpy import portfolio_l1 as portfolio_l1_cvxpy

np.set_printoptions(precision=5)  # Print few decimal places
np.set_printoptions(suppress=True)  # Suppress scientific notation


def test_portfolio():
    n = 10
    d = 3
    gamma = 3
    lambd = 1e-4

    np.random.seed(4)
    F = np.random.normal(size=(n, d))
    Fmask = np.random.randint(0, high=2, size=(n, d))
    print(Fmask)
    F = np.multiply(F, Fmask)
    Ddiag = np.random.uniform(high=np.sqrt(d), size=(n,))
    D = np.diag(Ddiag)

    mu = np.random.normal(size=(n,))
    Sigma = F @ F.T + D
    x_prev = 1 / n * np.ones(n)

    x_orig, x_reformed = portfolio_1_cvxpy(n, mu, gamma, Sigma, lambd, x_prev, d, D, F)

    P = 2 * np.block(
        [[gamma * D + lambd * np.eye(n), np.zeros((n, d))], [np.zeros((d, n)), gamma * np.eye(d)]]
    )

    A = np.block([[F.T, -np.eye(d)], [np.ones(n), np.zeros(d)], [-np.eye(n), np.zeros((n, d))]])
    b = np.hstack([np.zeros(d), 1, np.zeros(n)])
    q = np.concatenate([-(mu + 2 * lambd * x_prev), np.zeros(d)])
    z_single_block, z_full, y_full, z_triple_block = portfolio_2_cvxpy(
        n, mu, 0.5, P, lambd, x_prev, d, F, b, A
    )

    assert np.linalg.norm(x_orig - x_reformed) <= TEST_EPS
    assert np.linalg.norm(x_reformed - z_triple_block) <= TEST_EPS
    assert np.linalg.norm(z_triple_block - z_single_block) <= TEST_EPS

    # test with DR splitting
    Am, An = A.shape
    M = np.block([[P, A.T], [-A, np.zeros((Am, Am))]])
    lhs = np.eye(Am + An) + M
    c = np.hstack([q, b])
    sk = np.hstack([z_full, y_full])

    def proj(v):
        l = An + d + 1
        u = Am + An
        v[l:u] = np.maximum(v[l:u], 0)
        return v

    print("---testing DR---")
    K = 1000
    print(c)
    for _ in range(K):
        utilde = np.linalg.solve(lhs, sk - c)

        u = proj(2 * utilde - sk)
        sk = sk + u - utilde
    print("from DR:")
    z_DR = sk[:An]
    y_DR = sk[An:]
    print(z_DR)
    print(y_DR)
    print("w:", b - A @ z_DR)
    print("dual feas:", P @ z_DR + A.T @ y_full + q)


def test_portfolio_l1():
    print("----testing l1----")
    n = 10
    d = 3
    gamma = 5
    alpha = 1 / gamma
    lambd = 1

    np.random.seed(3)
    F = np.random.normal(size=(n, d))
    Fmask = np.random.randint(0, high=2, size=(n, d))
    F = np.multiply(F, Fmask)
    Ddiag = np.random.uniform(high=np.sqrt(d), size=(n,))
    D = np.diag(Ddiag)

    mu = np.random.normal(size=(n,))
    Sigma = F @ F.T + D
    x_prev = 1 / n * np.ones(n)

    A, P, q, b, x_orig, z = portfolio_l1_cvxpy(n, mu, gamma, Sigma, lambd, x_prev, d, D, F, alpha)
    assert np.linalg.norm(x_orig - z[:n]) <= TEST_EPS

    print("solving with DR")

    Am, An = A.shape
    M = np.block([[P, A.T], [-A, np.zeros((Am, Am))]])
    lhs = np.eye(Am + An) + M
    c = np.hstack([q, b])
    sk = np.zeros(Am + An)

    def proj(v):
        l = An + d + 1
        u = Am + An
        v[l:u] = np.maximum(v[l:u], 0)
        return v

    K = 1000
    for _ in range(K):
        utilde = np.linalg.solve(lhs, sk - c)
        u = proj(2 * utilde - sk)
        sk = sk + u - utilde

    print(sk)
    assert np.linalg.norm(sk[:n] - x_orig) <= TEST_EPS


def test_sampling():
    n = 10
    C = 2 / n

    N = 10000
    samples = np.random.dirichlet(np.ones(n), size=N)
    print(samples)
    filtered_samples = samples[np.all(samples <= C, axis=1)]
    # mask = samples[]
    print(filtered_samples.shape)
    # print(filtered_samples)

    assert np.all(filtered_samples <= C)
