import numpy as np
import scipy.sparse as spa

from algoverify.linearexpression import LinearExpression


def test_LinearExpression_dense():
    np.random.seed(0)
    m = 15
    n = 10
    x = LinearExpression.from_int(n)
    assert np.linalg.norm(x.decomposition_dict[x] - np.eye(n)) <= 1e-10
    y = LinearExpression.from_int(n)
    z = x + y
    assert x in z.decomposition_dict
    assert y in z.decomposition_dict

    A = np.random.normal(size=(n, n))

    Ax = A @ x
    assert np.linalg.norm(Ax.decomposition_dict[x] - A) <= 1e-8

    negAx = -Ax
    assert np.linalg.norm(negAx.decomposition_dict[x] + A) <= 1e-8

    Axhalf = Ax / 2
    assert np.linalg.norm(Axhalf.decomposition_dict[x] - A / 2) <= 1e-8

    Axhalfmult = Ax * 0.5
    assert np.linalg.norm(Axhalfmult.decomposition_dict[x] - Axhalf.decomposition_dict[x]) <= 1e-8

    z2 = x - y
    assert x in z2.decomposition_dict
    assert y in z2.decomposition_dict

    null = x - x
    assert x not in null.decomposition_dict

    B = np.random.normal(size=(m, n))
    y = B @ x

    assert y.ndim == m
    assert not y.is_leaf


def test_LinearExpression_sparse():
    m, n = 2, 3
    x = LinearExpression.from_int(n)
    A = spa.random(m, n, density=0.4, random_state=10)
    y = LinearExpression.from_int(m)

    test_expr = 5 * y + 2 * A @ x
    assert spa.linalg.norm(test_expr.decomposition_dict[y] - 5 * spa.eye(m)) <= 1e-10
    assert spa.linalg.norm(test_expr.decomposition_dict[x] - 2 * A) <= 1e-10

    test_expr2 = 3 * x - 4 * A.T @ y
    assert spa.linalg.norm(test_expr2.decomposition_dict[x] - 3 * spa.eye(n)) <= 1e-10
    assert spa.linalg.norm(test_expr2.decomposition_dict[y] + 4 * A.T) <= 1e-10


def test_composite_LinearExpressions():
    m, n = 2, 3
    t = 0.1
    np.random.seed(0)

    # x_test = np.random.uniform(size=(n,))
    A = np.random.normal(size=(m, n))
    # c = np.random.uniform(size=(n,))
    # b = A @ x_test

    x = LinearExpression.from_int(n)
    y = LinearExpression.from_int(m)

    c_param = LinearExpression.from_int(n)
    # b_param = Vector(n)

    test_expr = x - t * (c_param - A.T @ y)
    # print(test_expr.decomposition_dict)

    assert spa.linalg.norm(test_expr.decomposition_dict[x] - spa.eye(n)) <= 1e-10
    assert np.linalg.norm(test_expr.decomposition_dict[y] - t * A.T) <= 1e-10
    assert spa.linalg.norm(test_expr.decomposition_dict[c_param] + t * spa.eye(n)) <= 1e-10


def test_LinearExpression_const():
    np.random.seed(0)
    m = 15
    n = 10
    x = LinearExpression.from_int(n)

    y = x + 1
    assert np.allclose(y.const, np.full(n, 1.0))

    x1 = LinearExpression.from_int(m) + np.ones(m)
    x2 = LinearExpression.from_int(m) + np.ones(m)

    y1 = x1 + x2
    assert np.allclose(y1.const, np.full(m, 2.0))

    y2 = x1 * 2
    assert np.allclose(y2.const, np.full(m, 2.0))

    y3 = -x1
    assert np.allclose(y3.const, np.full(m, -1.0))

    A = np.random.normal(size=(m, m))
    y4 = A @ x1
    assert np.allclose(
        y4.const,
        np.array(
            [
                10.305124759797446,
                2.980568658097792,
                -3.5177123686392218,
                -5.167943315980211,
                -4.653576369372361,
                0.6419465154339059,
                4.453573614583073,
                10.871428039550418,
                1.0422981175013304,
                1.8518359099179722,
                -0.9835712059772468,
                -1.0339206780812593,
                -7.531465321271639,
                4.512467241345987,
                -0.05284513124944179,
            ]
        ),
    )

    y5 = np.full(m, 2.0) + x2
    assert np.allclose(y5.const, np.full(m, 3.0))

    y6 = np.full(m, 2.0) - x2
    assert np.allclose(y6.const, np.full(m, -1.0))

