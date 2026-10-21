import numpy as np

import fomtools as ft
from tests.conftest import TEST_EPS
from tests.cvxpy import fixed_point

np.set_printoptions(precision=5)  # Print few decimal places
np.set_printoptions(suppress=True)  # Suppress scientific notation


def test_nnls_convergence(vp_param_1):
    """
    Test non-negative least squares using projected gradient descent.
    The problem is:
        minimize (1/2) ||Az - x||^2
        subject to z >= 0

    We use projected gradient descent with step size eta = 0.5/L,
    where L is the maximum eigenvalue of A^T A.
    """
    m = 5  # Number of constraints
    n = 3  # Number of variables
    K = 10  # Number of iterations for verification

    np.random.seed(0)
    A = np.random.normal(size=(m, n))
    x_lb, x_ub = 10, 11  # Parameter bounds

    ATA = A.T @ A
    # eigvalsh (symmetric routine) returns real eigenvalues; eigvals (general) can return
    # complex128 with a zero imaginary part on some BLAS backends, which propagates a
    # complex step size into bound_propagation and fails its float accumulator.
    L = np.max(np.linalg.eigvalsh(ATA))
    eta = 0.5 / L

    print(f"Matrix A shape: {A.shape}")
    print(f"Parameter bounds: [{x_lb}, {x_ub}]")
    print(f"Lipschitz constant L: {L:.6f}")
    print(f"Step size eta: {eta:.6f}")

    # Add parameter x as a box constraint
    x_param = vp_param_1.parameter(m, lb=x_lb, ub=x_ub)

    # Get fixed point z* using Verifier's fixed_point method
    z_star = vp_param_1.fixed_point(
        n_dim=n,
        operator=lambda z: vp_param_1.relu(z - eta * A.T @ (A @ z - x_param)),
        lb=0,
        ub=100,
    )

    # Set up initial iterate
    z0 = vp_param_1.iterate(n, lb=1, ub=1)
    z = [None for _ in range(K + 1)]
    z[0] = z0

    all_residuals = []

    for k in range(1, K + 1):
        print(k)
        # Perform ReLU step: z[k] = ReLU(z[k-1] - eta * A^T(Az[k-1] - x))
        z[k] = vp_param_1.relu(z[k - 1] - eta * A.T @ (A @ z[k - 1] - x_param))
        # Set objective to measure ||z[k] - z*||_inf
        vp_param_1.maximize(ft.norm([z[k] - z_star], np.inf))
        res = vp_param_1.solve()
        all_residuals.append(res)

    print(f"opt x_param at last K: {x_param.value}")
    print(f"VP resids: {all_residuals}")

    # Extract the actual values from the verification
    z_star_value = z_star.value
    x_worst_case = x_param.value  # Extract the worst-case x that was used

    print(f"\nExtracted z_star from verification: {z_star_value}")
    print("extracting values from VP:")
    for k in range(K + 1):
        print(f"-K={k}-")
        print(z[k].value)

    # Solve the same problem with CVXPY using the exact same x that verification used
    z_optimal_value = fixed_point(n, A, x_worst_case, l=0)

    # Compare objective function values using the same x  |
    # Wasn't sure exactly how to compare , right now I just compare the z* from cvxpy using the x extracted from VP  (exactly the same)
    # Also did it with just comparing the z* from the worst case in sampling and found to still be about the same
    f_z_star = 0.5 * np.sum((A @ z_star_value - x_worst_case) ** 2)
    f_optimal = 0.5 * np.sum((A @ z_optimal_value - x_worst_case) ** 2)

    print(f"CVXPY optimal z*: {z_optimal_value}")
    print(f"objective diff: {abs(f_z_star - f_optimal):.2e}")

    # Assert that verification solution is close to optimal
    assert abs(f_z_star - f_optimal) <= TEST_EPS, "Verification solution not close to optimal"

    # return all_residuals, z_star, z_star_value, z_optimal_value

    residuals = all_residuals

    m, n = 5, 3
    K = 10
    N = 100  # number of samples

    np.random.seed(0)
    A = np.random.normal(size=(m, n))
    x_lb, x_ub = 10, 11

    ATA = A.T @ A
    # eigvalsh (symmetric routine) guarantees real eigenvalues — see note above.
    L = np.max(np.linalg.eigvalsh(ATA))
    eta = 0.5 / L

    sample_residuals = np.zeros((N, K))

    for i in range(N):
        # Generate random x_param
        x_param_sample = np.random.uniform(x_lb, x_ub, m)
        z_star_sample = fixed_point(n, A, x_param_sample, l=0)

        # Initialize z
        z_k = np.ones(n)  # Start from z0 = 1

        for k in range(1, K + 1):
            # Perform ReLU step
            z_k_new = np.maximum(z_k - eta * A.T @ (A @ z_k - x_param_sample), 0)

            # Record residual
            sample_residuals[i, k - 1] = np.max(np.abs(z_k_new - z_star_sample))

            # Update z for next iteration
            z_k = z_k_new

    # Compare results
    verification_bounds = np.array(residuals)
    sample_max = np.max(sample_residuals, axis=0)

    print(f"sample max resids: {sample_max}")

    # Assert that verification bounds are at least as good as sample max
    assert np.all(verification_bounds >= sample_max - TEST_EPS), "Verification bounds not valid"
