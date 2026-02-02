import copy

import numpy as np


def add_conv_cuts(wi, ai, y, lambd, y_l, y_u):
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
    if I.shape[0] == 0:
        return np.sum(np.multiply(w, Uhat)) - lambd
    if Icomp.shape[0] == 0:
        return np.sum(np.multiply(w, Lhat)) - lambd

    return np.sum(np.multiply(w[I], Lhat[I])) + np.sum(np.multiply(w[Icomp], Uhat[Icomp])) - lambd


def create_new_constr(w, a, y, Iint, lI, h, Lhat, Uhat):
    new_constr = 0
    for idx in Iint:
        new_constr += a[idx] * (y[idx] - Lhat[idx])
    # this lI func already factors in lambd so we dont subtract it again from the top
    new_constr += lI / (Uhat[h] - Lhat[h]) * (y[h] - Lhat[h])

    return w <= new_constr


def get_cuts(step, i, _w, _A, _y, l, u):
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
