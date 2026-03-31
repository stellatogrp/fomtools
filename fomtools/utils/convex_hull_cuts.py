import copy

import numpy as np


def add_conv_cuts(wi, ai, y, lambd, y_l, y_u):
    """
    Generate convex hull cuts for MIP bound tightening.

    Implements the Huchette-Vielma algorithm to find violated convex hull
    inequalities for piecewise linear functions. Returns cut data if a
    violated inequality is found.

    Parameters
    ----------
    wi : np.ndarray
        Current objective coefficient values.
    ai : np.ndarray
        Constraint coefficients.
    y : np.ndarray
        Current variable values.
    lambd : float
        Threshold parameter for the cut.
    y_l : np.ndarray
        Lower bounds on variables.
    y_u : np.ndarray
        Upper bounds on variables.

    Returns
    -------
    tuple
        A tuple (Iint, lI, h, L_hat, U_hat) where:
        - Iint : np.ndarray or None - Indices for the cut
        - lI : float or None - Cut coefficient
        - h : int or None - Pivot index
        - L_hat : np.ndarray or None - Adjusted lower bounds
        - U_hat : np.ndarray or None - Adjusted upper bounds

        Returns (None, None, None, None, None) if no violated cut is found.
    """
    L_hat = np.zeros(y_l.shape)
    U_hat = np.zeros(y_u.shape)

    for j in range(y.shape[0]):
        if ai[j] >= 0:
            L_hat[j] = y_l[j]
            U_hat[j] = y_u[j]
        else:
            L_hat[j] = y_u[j]
            U_hat[j] = y_l[j]

    Iint, rhs, lI, h = compute_v(ai, y, lambd, L_hat, U_hat)

    if Iint is not None and wi > rhs + 1e-6:
        return Iint, lI, h, L_hat, U_hat

    return None, None, None, None, None


def compute_v(wi, xi, lambd, Lhat, Uhat):
    """
    Compute a violated inequality using the Huchette-Vielma algorithm.

    Searches for a violated convex hull inequality by iteratively building
    an index set that separates the current point from the convex hull.

    Parameters
    ----------
    wi : np.ndarray
        Objective coefficient values.
    xi : np.ndarray
        Current variable values.
    lambd : float
        Threshold parameter.
    Lhat : np.ndarray
        Adjusted lower bounds (accounting for coefficient signs).
    Uhat : np.ndarray
        Adjusted upper bounds (accounting for coefficient signs).

    Returns
    -------
    tuple
        A tuple (Iint, rhs, lI, h) where:
        - Iint : np.ndarray or None - Index set for the cut
        - rhs : float or None - Right-hand side value
        - lI : float or None - Cut function value
        - h : int or None - Pivot index

        Returns (None, None, None, None) if no violated inequality exists.
    """
    idx = np.arange(len(wi))

    filtered_idx = np.array([j for j in idx if wi[j] != 0 and np.abs(Uhat[j] - Lhat[j]) > 1e-7])

    def key_func(j):
        return (xi[j] - Lhat[j]) / (Uhat[j] - Lhat[j])

    keys = np.array([key_func(j) for j in filtered_idx])
    sorted_idx = np.argsort(
        keys
    )  # this is nondecreasing, should be the corrct one according to paper
    # sorted_idx = np.argsort(keys)[::-1]
    filtered_idx = filtered_idx[sorted_idx]

    I = np.array([])
    Icomp = set(range(len(wi)))

    lI = compute_I(wi, lambd, Lhat, Uhat, I, np.array(list(Icomp)))
    if lI < 0:
        return None, None, None, None

    for h in filtered_idx:
        Itest = np.append(I, h)
        Icomp_test = copy.copy(Icomp)
        Icomp_test.remove(int(h))

        lI_new = compute_I(
            wi, lambd, Lhat, Uhat, Itest.astype(np.int32), np.array(list(Icomp_test))
        )
        if lI_new < 0:
            Iint = I.astype(np.int32)
            rhs = np.sum(np.multiply(wi[Iint], xi[Iint])) + lI / (Uhat[int(h)] - Lhat[int(h)]) * (
                xi[int(h)] - Lhat[int(h)]
            )
            return Iint, rhs, lI, int(h)

        I = Itest
        Icomp = Icomp_test
        lI = lI_new
    else:
        return None, None, None, None


def compute_I(w, lambd, Lhat, Uhat, I, Icomp):
    """
    Evaluate the cut function value for a given index partition.

    Computes the value of the piecewise linear function at the boundary
    defined by the index set I (using lower bounds) and its complement
    Icomp (using upper bounds), minus the threshold lambda.

    Parameters
    ----------
    w : np.ndarray
        Coefficient vector.
    lambd : float
        Threshold parameter to subtract.
    Lhat : np.ndarray
        Adjusted lower bounds.
    Uhat : np.ndarray
        Adjusted upper bounds.
    I : np.ndarray
        Indices using lower bounds.
    Icomp : np.ndarray
        Indices using upper bounds (complement of I).

    Returns
    -------
    float
        The cut function value: sum(w[I] * Lhat[I]) + sum(w[Icomp] * Uhat[Icomp]) - lambd.
    """
    if I.shape[0] == 0:
        return np.sum(np.multiply(w, Uhat)) - lambd
    if Icomp.shape[0] == 0:
        return np.sum(np.multiply(w, Lhat)) - lambd

    return np.sum(np.multiply(w[I], Lhat[I])) + np.sum(np.multiply(w[Icomp], Uhat[Icomp])) - lambd


def create_new_constr(w, a, y, Iint, lI, h, Lhat, Uhat):
    """
    Build a constraint from the cut data.

    Constructs the actual constraint expression from the cut information
    returned by ``compute_v``.

    Parameters
    ----------
    w : variable
        The variable to constrain (left-hand side).
    a : np.ndarray
        Coefficient vector.
    y : np.ndarray
        Variable vector for the right-hand side.
    Iint : np.ndarray
        Index set for the cut.
    lI : float
        Cut function value at the pivot.
    h : int
        Pivot index.
    Lhat : np.ndarray
        Adjusted lower bounds.
    Uhat : np.ndarray
        Adjusted upper bounds.

    Returns
    -------
    Constraint
        The constraint w <= (cut expression).
    """
    new_constr = 0
    for idx in Iint:
        new_constr += a[idx] * (y[idx] - Lhat[idx])
    # this lI func already factors in lambd so we dont subtract it again from the top
    new_constr += lI / (Uhat[h] - Lhat[h]) * (y[h] - Lhat[h])

    return w <= new_constr


def get_cuts(step, i, _w, _A, _y, l, u):
    """
    Dispatch cut generation based on step type.

    Determines which cut generation strategy to use based on the step
    type (ReLU or soft threshold) and the index configuration.

    Parameters
    ----------
    step : Step
        The optimization step (ReLU or SoftThreshold).
    i : int
        The component index to generate cuts for.
    _w : np.ndarray
        Objective coefficient values.
    _A : np.ndarray
        Constraint coefficient matrix row.
    _y : np.ndarray
        Current variable values.
    l : np.ndarray
        Lower bounds.
    u : np.ndarray
        Upper bounds.

    Returns
    -------
    tuple
        Cut data (Iint, lI, h, L_hat, U_hat) or (None, None, None, None, None)
        if no cut applies.
    """
    try:
        if i in step.proj_indices and i in step.idx_with_binary_vars:
            return add_conv_cuts(_w, _A, _y, 0, l, u)
    except AttributeError:
        _r, _l = i in step.idx_with_right_binary_vars, i in step.idx_with_left_binary_vars
        if _r and (not _l or _w > 0):
            return add_conv_cuts(_w, _A, _y, step.lambd, l, u)
        if _l and (not _r or _w < 0):
            Iint, uI, h, l_Lhat, l_Uhat = add_conv_cuts(_w, _A, _y, -step.lambd, u, l)
            return Iint, -uI if uI is not None else uI, h, l_Lhat, l_Uhat

        return None, None, None, None, None
    return None, None, None, None, None
