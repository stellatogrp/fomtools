import cvxpy as cp
import numpy as np

from algoverify.verifier import Verifier

np.set_printoptions(precision=5)  # Print few decimal places
np.set_printoptions(suppress=True)  # Suppress scientific notation


def test_nnls_convergence():
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
    L = np.max(np.linalg.eigvals(ATA))
    eta = 0.5 / L
    
    print(f"Matrix A shape: {A.shape}")
    print(f"Parameter bounds: [{x_lb}, {x_ub}]")
    print(f"Lipschitz constant L: {L:.6f}")
    print(f"Step size eta: {eta:.6f}")
    
    
    solver_params = {
        'OutputFlag': 0,
    }
    VP = Verifier(solver_params=solver_params)
    
    # Add parameter x as a box constraint
    x_param = VP.add_param(m, lb=x_lb, ub=x_ub)
    
    # Get fixed point z* using Verifier's fixed_point method
    z_star = VP.fixed_point(
        n_dim=n,
        operator=lambda z: VP.relu_step(z - eta * A.T @ (A @ z - x_param)),
        lb=0, ub=100
    )

    # Set up initial iterate
    z0 = VP.add_initial_iterate(n, lb=1, ub=1)
    z = [None for _ in range(K + 1)]
    z[0] = z0
    
    all_residuals = []
    
    for k in range(1, K+1):
        print(k)
        # Perform ReLU step: z[k] = ReLU(z[k-1] - eta * A^T(Az[k-1] - x))
        z[k] = VP.relu_step(z[k-1] - eta * A.T @ (A @ z[k-1] - x_param))
        
        # Set objective to measure ||z[k] - z*||_inf
        VP.set_infinity_norm_objective([z[k] - z_star])
        res = VP.solve()
        all_residuals.append(res)
    
    print(f'opt x_param at last K: {VP.extract_sol(x_param)}')
    print(f'VP resids: {all_residuals}')
    
    # Extract the actual values from the verification
    z_star_value = VP.extract_sol(z_star)
    x_worst_case = VP.extract_sol(x_param)  # Extract the worst-case x that was used
    
    print(f"\nExtracted z_star from verification: {z_star_value}")
    print('extracting values from VP:')
    for k in range(K+1):
        print(f'-K={k}-')
        print(VP.extract_sol(z[k]))
    
    # Solve the same problem with CVXPY using the exact same x that verification used
    z_optimal = cp.Variable(n)
    obj = 0.5 * cp.sum_squares(A @ z_optimal - x_worst_case)
    constraints = [z_optimal >= 0]
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve()
    z_optimal_value = z_optimal.value
    
    # Compare objective function values using the same x  |
    # Wasn't sure exactly how to compare , right now I just compare the z* from cvxpy using the x extracted from VP  (exactly the same)
    # Also did it with just comparing the z* from the worst case in sampling and found to still be about the same
    f_z_star = 0.5 * np.sum((A @ z_star_value - x_worst_case)**2)
    f_optimal = 0.5 * np.sum((A @ z_optimal_value - x_worst_case)**2)
    
    print(f'CVXPY optimal z*: {z_optimal_value}')
    print(f'objective diff: {abs(f_z_star - f_optimal):.2e}')
    
    # Assert that verification solution is close to optimal
    assert abs(f_z_star - f_optimal) <= 1e-6, "Verification solution not close to optimal"
    
    return all_residuals, z_star, z_star_value, z_optimal_value


def test_nnls_with_random_sampling():
    """
    Test NNLS by comparing verification results with random sampling.
    """
    residuals, z_star, z_star_value, z_optimal_value = test_nnls_convergence()
    
    m, n = 5, 3
    K = 10
    N = 100  # number of samples 
    
    np.random.seed(0)
    A = np.random.normal(size=(m, n))
    x_lb, x_ub = 10, 11
    
    ATA = A.T @ A
    L = np.max(np.linalg.eigvals(ATA))
    eta = 0.5 / L
    
    sample_residuals = np.zeros((N, K))
    
    for i in range(N):
        # Generate random x_param
        x_param_sample = np.random.uniform(x_lb, x_ub, m)
        
        # Find z* by solving the QP directly using CVXPY
        z_star_sample = cp.Variable(n)
        obj = 0.5 * cp.sum_squares(A @ z_star_sample - x_param_sample)
        constraints = [z_star_sample >= 0]
        prob = cp.Problem(cp.Minimize(obj), constraints)
        prob.solve()
        z_star_sample = z_star_sample.value
        
        # Initialize z
        z_k = np.ones(n)  # Start from z0 = 1
        
        for k in range(1, K+1):
            # Perform ReLU step
            z_k_new = np.maximum(z_k - eta * A.T @ (A @ z_k - x_param_sample), 0)
            
            # Record residual
            sample_residuals[i, k-1] = np.max(np.abs(z_k_new - z_star_sample))
            
            # Update z for next iteration
            z_k = z_k_new
    
    # Compare results
    verification_bounds = np.array(residuals)
    sample_max = np.max(sample_residuals, axis=0)
    
    print(f'sample max resids: {sample_max}')
    
    # Assert that verification bounds are at least as good as sample max
    assert np.all(verification_bounds >= sample_max - 1e-6), "Verification bounds not valid"
    
    return residuals, sample_residuals


# if __name__ == "__main__":
#     # Run the verification test
#     test_nnls_convergence()
    
#     # Run the comparison with random sampling
#     test_nnls_with_random_sampling()