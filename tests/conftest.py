
import numpy as np
import pytest

from algoverify.verifier import Verifier

# available solvers used as parameters to initialize verifier objects in each fixture
SOLVERS = ["gurobi", "highs"]


@pytest.fixture(params=SOLVERS, scope="function")
def vp(request: pytest.FixtureRequest):
    """
    Pytest fixture that provides a `Verifier` instance. This fixture
    is parameterized to run tests using different solvers.
    """
    return Verifier(solver=request.param, obbt_handler=request.param)


@pytest.fixture(params=SOLVERS, scope="function")
def vp_param_1(request: pytest.FixtureRequest):
    """
    Pytest fixture that provides a `Verifier` instance configured with
    a solver whose output flag is set to 0 (silent mode).
    """
    return Verifier(solver=request.param, obbt_handler=request.param, params={"OutputFlag": 0})


@pytest.fixture(params=SOLVERS, scope="function")
def vp_param_2(request: pytest.FixtureRequest):
    """
    Pytest fixture that provides a `Verifier` instance whose solver is
    configured with a 10 seconds time limit, a MIP gap set of 1e-4, and the
    output flag set to 0 (silent mode).
    """
    return Verifier(
        solver=request.param,
        obbt_handler=request.param,
        params={"TimeLimit": 10, "MIPGap": 1e-4, "OutputFlag": 0},
    )


@pytest.fixture(params=SOLVERS, scope="function")
def vp_theory(request: pytest.FixtureRequest):
    """
    Pytest fixture that provides a `Verifier` object initialized with a
    custom theoretical function, `theory_func`
    """
    def theory_func(k):
        init_C = 0.5
        if k == 1:
            return np.inf
        return 2 * init_C / np.sqrt((k - 1) * (k + 2))

    return Verifier(solver=request.param, obbt_handler=request.param, theory_func=theory_func)
