import numpy as np
import scipy.sparse as spa

from fomtools.linearexpression import LinearExpression
from tests.conftest import TEST_EPS


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
    assert np.linalg.norm(Ax.decomposition_dict[x] - A) <= TEST_EPS

    negAx = -Ax
    assert np.linalg.norm(negAx.decomposition_dict[x] + A) <= TEST_EPS

    Axhalf = Ax / 2
    assert np.linalg.norm(Axhalf.decomposition_dict[x] - A / 2) <= TEST_EPS

    Axhalfmult = Ax * 0.5
    assert np.linalg.norm(Axhalfmult.decomposition_dict[x] - Axhalf.decomposition_dict[x]) <= TEST_EPS

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
    # x = A @ x_test

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
    assert np.allclose(y6.const, np.full(m, 1.0))
    assert np.allclose(-np.eye(m), list(y6.decomposition_dict.values())[0].todense())


def test_LinearExpression_constr():
    np.random.seed(0)
    m = 15
    n = 10
    x1 = LinearExpression.from_int(n)

    constr_1 = x1 == 0
    constr_2 = x1 == 1
    constr_3 = x1 >= 3

    assert np.allclose(constr_1.lhs.const, np.full(n, 0.0))
    assert np.allclose(constr_2.lhs.const, np.full(n, -1.0))
    assert np.allclose(constr_3.lhs.const, np.full(n, 3.0))
    # ensure greater than is handled accurately
    assert np.allclose(
        [i.data for i in constr_3.lhs.decomposition_dict.values()][0], np.array([np.full(n, -1)])
    )

    y1 = LinearExpression.from_int(m) + np.ones(m)
    y2 = LinearExpression.from_int(m) + np.full(m, 4)

    constr_4 = y1 <= 2
    constr_5 = y1 <= y2

    assert np.allclose(constr_4.lhs.const, np.full(m, -1.0))
    assert np.allclose(constr_5.lhs.const, np.full(m, -3.0))


def test_LinearExpression_bool():
    n = 10
    M = 10
    x_notbinary = LinearExpression.from_int(n)
    assert not x_notbinary.is_binary

    xb = LinearExpression.from_int(n, is_binary=True)
    assert xb.is_binary
    assert xb.is_leaf

    M_xb = M * xb
    assert not M_xb.is_leaf

    y = LinearExpression.from_int(n)
    constr = y <= M_xb

    assert xb in constr.lhs.decomposition_dict
    assert M_xb not in constr.lhs.decomposition_dict
    assert np.allclose(constr.lhs.decomposition_dict[xb].todense(), -M * np.eye(n))


def test_LinearExpression_slice():
    m, n = 4, 3
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    x1 = LinearExpression.from_int(n, is_binary=True)

    Ax1 = A @ x1
    assert Ax1.ndim == m
    assert not Ax1.is_binary
    assert np.allclose(A, Ax1[x1])

    Ax0 = Ax1[0]
    assert Ax0.ndim == 1
    assert np.allclose(Ax0[x1], A[0])

    B = np.array([[13, 14, 15], [16, 17, 18], [19, 20, 21], [22, 23, 24]])
    x2 = LinearExpression.from_int(n, is_binary=True)

    Bx2 = B @ x2
    assert Bx2.ndim == m
    assert not Bx2.is_binary
    assert np.allclose(B, Bx2[x2])

    y = Ax1 + Bx2
    assert np.allclose(y[x1], A)
    assert np.allclose(y[x2], B)

    y0 = y[0]
    assert np.allclose(y0[x1], A[0])
    assert np.allclose(y0[x2], B[0])

    y1 = y[1:3]
    assert np.allclose(y1[x1], A[1:3])
    assert np.allclose(y1[x2], B[1:3])


def test_LinearExpression_hash():
    """Test that LinearExpression hashing works correctly for dict operations.

    This is important because __eq__ returns an EqualityConstraint (truthy)
    rather than a bool, so hash collisions would cause silent key collisions.
    Using the unique counter as the hash prevents this.
    """
    # Create two leaf variables with the same ndim
    a = LinearExpression.from_int(3)
    b = LinearExpression.from_int(3)

    # Each instance gets a unique counter
    assert a.counter != b.counter

    # Hashes are unique (based on counter)
    assert hash(a) != hash(b)
    assert hash(a) == a.counter
    assert hash(b) == b.counter

    # Dict stores them as separate entries
    d = {}
    d[a] = "value_a"
    d[b] = "value_b"
    assert len(d) == 2

    # Dict lookup returns correct values
    assert d[a] == "value_a"
    assert d[b] == "value_b"

    # Membership tests work correctly
    assert a in d
    assert b in d

    # New variable with same ndim is not in dict
    c = LinearExpression.from_int(3)
    assert c not in d

    # Composite expressions also get unique hashes
    expr1 = a + b
    expr2 = a - b
    assert hash(expr1) != hash(expr2)
    assert expr1.counter != expr2.counter

    d[expr1] = "expr1"
    d[expr2] = "expr2"
    assert len(d) == 4
    assert d[expr1] == "expr1"
    assert d[expr2] == "expr2"
