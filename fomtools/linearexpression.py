from typing import Self, Union

import numpy as np
import scipy.sparse as spa

from fomtools.constraint import EqualityConstraint, InequalityConstraint
from fomtools.utils.dict_utils import merge_dict, prune_dict


class LinearExpression:
    """
    A linear combination of function values, inner products (between points and/ or gradients)
    and constant scalars.

    Attributes:
    ----------
        ndim : int
            The dimension of underlying space, i.e., the size of each coefficient matrix.

        is_leaf : bool
            Indicates whether this expression is a primitive ("leaf") function value (i.e. not
            itself built from other `LinearExpression` objects). If `is_leaf` is True, this
            object is a function value defined from scratch.

        is_binary : bool
            Indicates whether a leaf expression is intended to be binary or not (i.e. constrained to only have values 0 or 1 in each dimension).

        decomposition_dict : dict
            A mapping from each leaf expression to its corresponding coefficient matrix. If
            `is_leaf` is True, this dict maps the object to its own coefficient matrix.

        counter : int
            Tracks the number of leaf `LinearExpression` objects.

    References
    ----------
        .. S. Diamond and S. Boyd. "CVXPY: A Python-Embedded Modeling Language for Convex
           Optimization." Journal of Machine Learning Research, 17(83):1-5, 2016.
        .. B. Goujaud, C. Moucer, F. Glineur, J. Hendrickx, A. Taylor, A. Dieuleveut (2022).
           "PEPit: computer-assisted worst-case analyses of first-order optimization methods
           in Python."

    """

    counter = 0
    # this is necessary else numpy will not even look at the @ operator due to precedence
    __array_priority__ = 1000

    def __init__(self, decomposition_dict: dict, const: np.ndarray = None, is_binary: bool = False):
        """
        Initialize a `LinearExpression` object with decomposition dictionary.

        Parameters
        ----------
        decomposition_dict : dict
            A mapping from each leaf `LinearExpression` to its coefficient matrix (an nxn sparse
            matrix).

        const : ndarray or None, optional
            A one-dimensional array or None (default). If `const` is None, the `const` attribute
            of this object will be set to a `numpy.ndarray` of length `shape` filled with zeros.

        is_binary : bool, optional
            A flag to determine if a leaf element is intended to be binary (0 or 1 in each dimension) or not. Defaults to `False`.

        Raises
        ------
        ValueError
            If all the coefficient matrices in the `decomposition_dict` doesn't have the same number
            of rows.

        ValueError
            If `const` is not one-dimensional.

        ValueError
            If the number of rows in `const` does not equal the number of rows in coefficient
            matrices in the `decomposition_dict`.
        """

        shapes = {mat.shape[0] for mat in decomposition_dict.values()}
        if len(shapes) > 1:
            raise ValueError(
                f"""All coefficient matrices must have the same number of rows;
                    found row counts {len(shapes)}"""
            )

        self.ndim = shapes.pop() if len(shapes) == 1 else 0
        if const is not None:
            if self.ndim == 0:
                self.ndim = const.shape[0]

            try:
                np.broadcast_shapes(const.shape, (1, self.ndim))
            except ValueError:
                raise ValueError(
                    "Matrix for a constant must have same number of rows as decomposition dict."
                )

        self.const = np.zeros(self.ndim) if const is None else const
        self.decomposition_dict = decomposition_dict
        self.counter = LinearExpression.counter
        LinearExpression.counter += 1
        self.is_leaf = False
        self.is_binary = is_binary
        self.lb = [i.lb for i in decomposition_dict.keys()]
        self.ub = [i.ub for i in decomposition_dict.keys()]

    @classmethod
    def from_int(
        cls, ndim: int, is_binary: bool = False, lb: float = -np.inf, ub: float = np.inf
    ) -> Self:
        """
        Initialize a `LinearExpression` object with an integer `shape`. Internally, this
        creates a decomposition dictionary that maps the `LinearExpression` object with a sparse
        identity matrix of shape `(shape, shape)`.

        Parameters
        ----------
        ndim : int
            The dimension of underlying space, i.e., the size of each coefficient matrix.
        is_binary: bool
            Flag to determine if the leaf expression is binary or not (0 or 1 in each dimension).

        Returns
        -------
        LinearExpression
            A new `LinearExpression` instance whose decomposition contains a single
            entry: the instance itself as a key, and a SciPy sparse identity matrix of
            shape `(shape, shape)` as the value.
        """
        obj = cls.__new__(cls)  # uninitialized instance
        obj.is_leaf = True
        obj.decomposition_dict = {obj: spa.eye(ndim)}
        obj.ndim = ndim
        obj.counter = cls.counter
        obj.const = np.zeros(ndim)
        obj.is_binary = is_binary
        obj.lb = np.zeros(ndim) if is_binary else np.full(ndim, lb, dtype=float)
        obj.ub = np.ones(ndim) if is_binary else np.full(ndim, ub, dtype=float)
        cls.counter += 1
        return obj

    def __str__(self):
        return f"Var{self.counter}"

    def __repr__(self):
        return f"<LinearExpression {'Leaf' if self.is_leaf else ''}, dim={self.ndim}>"

    def __hash__(self):
        """
        Return a hash based on this expression's unique counter.

        Each LinearExpression is assigned a unique, monotonically increasing
        counter at construction. We use a tuple hash to avoid collisions with
        integers, which is necessary because __eq__ returns an EqualityConstraint
        (truthy) rather than a bool. Without this, dict[0] could incorrectly
        match a LinearExpression with counter=0.
        """
        return hash(("LinearExpression", self.counter))

    def __getitem__(self, item: Self | int):
        """
        Retrieve a coefficient matrix or a sliced LinearExpression corresponding
        to `item`.

        If the `item` is a leaf expression, this method returns its corresponding
        coefficient matrix. If the `item` is an `int` or a `slice`, this method
        returns a new `LinearExpression` instance representing a specified subset of
        rows from all the corresponding coefficients in the decomposition dictionary.

        Raises
        ------
        KeyError
            If the specified leaf expression is not found in this `LinearExpression` object.
        """
        try:
            return self.decomposition_dict[item]
        except KeyError:
            try:
                return LinearExpression(
                    decomposition_dict={
                        k: v.tocsr()[item] if spa.issparse(v) else np.atleast_2d(v[item])
                        for k, v in self.decomposition_dict.items()
                    },
                    const=np.atleast_1d(self.const[item]),
                )
            except TypeError:
                raise KeyError("""
                The specified leaf expression is not present in this LinearExpression object.
                """)

    def __add__(self, other: Union[Self, np.array, list, int, float]) -> Self:
        """
        Add two `LinearExpression` objects, ensuring they have the same dimension.

        Parameters
        ----------
        other : LinearExpression, np.array, list, int, float
            The expression or a vector to add to this one.

        Returns
        -------
        LinearExpression
            A new `LinearExpression` object whose decomposition maps each leaf in the union of both
            expressiom to the sum of its coefficient matrices, treating any leaf missing from one
            operand as having a zero coefficient matrix.

        Raises:
        ------
        ValueError
            If the objects (`LinearExpression` or a vector) to be added together does not have
            the same dimension.

        ValueError
            If the object to be added is not of one of the following types: `LinearExpression`,
            `np.ndarray`,  `float`, `int`, or `list`.
        """
        try:
            return LinearExpression(
                decomposition_dict=self.decomposition_dict,
                const=self.const + np.array(other).astype(float),
            )
        except ValueError:
            try:
                # Check to see if other is of type list or a numpy array
                if len(other) == self.ndim:
                    return LinearExpression(
                        decomposition_dict=self.decomposition_dict, const=self.const + other
                    )
                else:
                    raise ValueError(
                        f"Cannot add constant of mismatched dimensions. Expected dimension: {self.ndim}, got {len(other)}"
                    )
            except TypeError:
                try:
                    # Check to see if other is of type LinearExpression
                    if self.ndim == other.ndim:
                        merged_decomposition_dict = merge_dict(
                            self.decomposition_dict, other.decomposition_dict
                        )
                        return LinearExpression(
                            decomposition_dict=prune_dict(merged_decomposition_dict),
                            const=self.const + other.const,
                        )
                    else:
                        raise ValueError(
                            f"Cannot add LinearExpression objects of mismatched dimensions. Expected dimension: {self.ndim}, got {other.ndim}"
                        )
                except AttributeError:
                    raise ValueError(
                        f"Can only add LinearExpression objects to a LinearExpression object, numpy arrays, list, float, or an int got type {type(other)}."
                    )

    def __radd__(self, other: Union[Self, np.array, list]) -> Self:
        """
        Reflected addition to support other + expression.

        This method is invoked when a `LinearExpression` object appears on
        the right-hand side of a `+` operation (e.g. `np.array([1,2,3]) + expr`).
        It delegates to `__add__`, so that both `np.array([1,2,3]) + expr` and
        `expr + np.array([1,2,3])` behave identically.

        Parameters
        ----------
        other : LinearExpression, np.array, list
            The expression or a vector to add to this one.

        Returns
        -------
        LinearExpression
            A new `LinearExpression` object whose decomposition maps each leaf in the union of both
            expressiom to the sum of its coefficient matrices, treating any leaf missing from one
            operand as having a zero coefficient matrix.

        Raises:
        ------
        ValueError
            If the objects (`LinearExpression` or a vector) to be added together does not have
            the same dimension.

        ValueError
            If the object to be added is not of one of the following types: `LinearExpression`,
            `np.ndarray`, or `list`.
        """
        return self + other

    def __mul__(self, scalar: Union[float, int]) -> Self:
        """
        Multiply this `LinearExpression` object by a scalar.

        Parameters
        ----------
        scalar : float or int
            The numeric factor to multiply each coefficient matrix by.

        Returns
        -------
        LinearExpression
            A new `LinearExpression` object in which each leaf’s coefficient matrix is multiplied
            by the given `scalar`.

        Raises:
        ------
        TypeError
            If the `scalar` is not an instance of int or float.

        """
        try:
            return LinearExpression(
                decomposition_dict={
                    k: v * np.float64(scalar) for k, v in self.decomposition_dict.items()
                },
                const=self.const * np.float64(scalar),
            )
        except ValueError:
            raise TypeError(
                f"""Can only multiply by int or float, got {type(scalar)}. If intending to use
                 a matrix, use the @ symbol with the matrix on the left."""
            )

    def __rmul__(self, scalar: Union[float, int]) -> Self:
        """
        Reflected multiplication to support scalar * expression.

        This method is invoked when a `LinearExpression` object appears on the
        right-hand side of a `*` operation (e.g. `3 * expr`). It delegates to
        to `__mul__`, so that both `expr * 3` and `3  * expr` behave identically.

        Parameters
        ----------
        scalar : float or int
            The numeric factor to multiply each coefficient matrix by.

        Returns
        -------
        LinearExpression
            A new `LinearExpression` in which each leaf’s coefficient matrix is multiplied
            by the given `scalar`.

        Raises:
        ------
        TypeError
            If the `scalar` is not an instance of int or float.
        """
        return self * scalar

    def __truediv__(self, scalar: Union[float, int]) -> Self:
        """
        Divide this LinearExpression by a scalar factor.

        This method is invoked when the expression appears on the left-hand side
        of a true division (e.g. `expr / 3`).

        Parameters
        ----------
        scalar : float or int
            The numeric divisor to apply to every coefficient matrix.

        Returns
        -------
        LinearExpression
            A new `LinearExpression` in which each leaf's coefficient matrix is divided
            by the given `scalar`.

        Raises
        ------
        TypeError
            If the `scalar` is not an instance of int or float.

        ZeroDivisionError
            If `scalar` is zero.
        """
        if scalar == 0:
            raise ZeroDivisionError("Division of LinearExpression by zero is not allowed.")

        return self * (1 / scalar)

    def __neg__(self) -> Self:
        """
        Compute the negation of this LinearExpression.

        This method is invoked for unary negation (e.g., `-expr`). It returns a new
        expression with each coefficient matrix in the decomposition multiplied by -1.

        Returns
        -------
        LinearExpression
            A new `LinearExpression` object with all coefficient matrices negated.
        """
        return self * -1

    def __sub__(self, other: Self) -> Self:
        """
        Subtract one LinearExpression from another.

        Parameters
        ----------
        other : LinearExpression, np.array, list
            The expression or a vector xto subtract from this one.

        Returns
        -------
        LinearExpression
            A new `LinearExpression` object whose decomposition maps each leaf in the union of
            both expression to the difference of its coefficient matrices, treating any leaf
            missing from one operand as having a zero coefficient matrix.

        Raises
        ------
        ValueError
            If the two `LinearExpression` instances have mismatched dimensions.

        ValueError
            If the object to be subtracted is not of one of the following types: `LinearExpression`,
            `np.ndarray`, or `list`.

        """
        return self + (-other)

    def __rsub__(self, other: Self) -> Self:
        """
        Reflected subtraction to support other - expression.

        This method is invoked when a `LinearExpression` object appears on
        the right-hand side of a `-` operation (e.g. `np.array([1,2,3]) - expr`).

        Parameters
        ----------
        other : LinearExpression, np.array, list
            The expression or a vector that this one is to be subtracted from.

        Returns
        -------
        LinearExpression
            A new `LinearExpression` object whose decomposition maps each leaf in the union of
            both expression to the difference of its coefficient matrices, treating any leaf
            missing from one operand as having a zero coefficient matrix.

        Raises
        ------
        ValueError
            If the two `LinearExpression` instances have mismatched dimensions.

        ValueError
            If the object to be subtracted is not of one of the following types: `LinearExpression`,
            `np.ndarray`, or `list`.

        """
        return (-self) + other

    def __matmul__(self, other):
        # TODO
        raise NotImplementedError("np matrix or sp matrix should be on the left of the @")

    def __rmatmul__(self, matrix: Union[np.ndarray, spa.spmatrix]) -> Self:
        """
        Multiply this `LinearExpression` object by another matrix.

        This method is called when a `LinearExpression` appears on the right‐hand side
        of the `@` operator (e.g., `M @ expr`).

        Parameters
        ----------
        matrix : numpy.ndarray or scipy.sparse.spmatrix
            A matrix of shape `(m, n)`—where `n` is this expression’s dimension—to
            multiply on the left of each coefficient matrix.

        Returns
        -------
        LinearExpression
            A new `LinearExpression` object in which each leaf's coefficient matrix is
            multiplied by the given `matrix`.

        Raises
        ------
        TypeError
            If `other` is not a NumPy array or SciPy sparse matrix.
        ValueError
            If `other.shape[1]` does not match this expression’s dimension `n`.
        """

        if matrix.shape[1] != self.ndim:
            raise ValueError(
                f"""Matrix inner dimension ({matrix.shape[1]}) does not match
                    the LinearExpression dimension ({self.ndim})"""
            )

        try:
            return LinearExpression(
                decomposition_dict={k: matrix @ v for k, v in self.decomposition_dict.items()},
                const=np.asarray(matrix @ self.const).ravel(),
            )
        except TypeError:
            raise TypeError(
                f"""Expected numpy.ndarray or scipy.sparse matrix for multiplication;
                got {type(matrix).__name__}. """
            )

    def __le__(self, rhs: Union[Self, np.ndarray, list]) -> InequalityConstraint:
        """
        Build a less-than-or-equality (`<=`) constraint between this expression and
        another `LinearExpression` object or numpy vector.`

        Parameters
        ----------
        rhs: LinearExpression, np.array, list
            The right-hand side of the inequalit. Scalars are broadcast to the shape of
            `self`. Arrays or expressions must be of same shape with `self`.

        Returns
        -------
        InequalityConstraint
            A constraint object representing `self <= rhs`.

        Raises
        ------
        TypeError
            If `rhs` is not a LinearExpression object, NumPy array or a list.

        ValueError
            If the shapes of `self` and `rhs` are not compatible (i.e., not equal and
            not broadcastable).

        Notes
        -----
        This method does not evaluate the inequality; it only builds a constraint
        for the solver.
        """
        return InequalityConstraint(self, rhs)

    def __eq__(self, rhs: Union[Self, np.ndarray, list]) -> EqualityConstraint:
        """
        Build a equality (`==`) constraint between this expression and
        another `LinearExpression` object or numpy vector.`

        Parameters
        ----------
        rhs: LinearExpression, np.array, list
            The right-hand side of the inequalit. Scalars are broadcast to the shape of
            `self`. Arrays or expressions must be of same shape with `self`.

        Returns
        -------
        EqualityConstraint
            A constraint object representing `self == rhs`.

        Raises
        ------
        TypeError
            If `rhs` is not a LinearExpression object, NumPy array or a list.

        ValueError
            If the shapes of `self` and `rhs` are not compatible (i.e., not equal and
            not broadcastable).

        Notes
        -----
        This method does not evaluate the equality; it only builds a constraint
        for the solver.
        """
        return EqualityConstraint(self, rhs)

    def __ge__(self, rhs: Union[Self, np.ndarray, list]) -> InequalityConstraint:
        """
        Build a greater-than-or-equality (`>=`) constraint between this expression and
        another `LinearExpression` object or numpy vector.`

        Parameters
        ----------
        rhs: LinearExpression, np.array, list
            The right-hand side of the inequality. Scalars are broadcast to the shape of
            `self`. Arrays or expressions must be of same shape with `self`.

        Returns
        -------
        InequalityConstraint
            A constraint object representing `self >= rhs`.

        Raises
        ------
        TypeError
            If `rhs` is not a LinearExpression object, NumPy array or a list.

        ValueError
            If the shapes of `self` and `rhs` are not compatible (i.e., not equal and
            not broadcastable).

        Notes
        -----
        This method does not evaluate the inequality; it only builds a constraint
        for the solver.
        """
        return InequalityConstraint(-self, -rhs)

    def update_bounds(self, lb: np.ndarray, ub: np.ndarray) -> None:
        """Update lower and upper bounds of the `LinearExpression` objects."""
        self.lb = np.maximum(self.lb, lb)
        self.ub = np.minimum(self.ub, ub)

    def bound_propagation(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the elementwise lower and upper bounds of a linear expression
        given the bound constraints of its coefficient matrices.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A pair (lower, upper) representing the elementwise lower and upper
            bounds of the evaluated expression.

        Raises
        ------
        ValueError
            If any computed lower bound exceeds its corresponding upper bound.
        """

        def _interval_bound_propagation(
            a: np.matrix, lb: np.ndarray, ub: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
            """
            Bounds the linear image y = A @ x for any x with lb ≤ x ≤ ub.

            Using midpoint m = (ub + lb)/2 and radius r = (ub - lb)/2:
                y_min = A @ m - |A| @ r
                y_max = A @ m + |A| @ r

            Returns (y_min, y_max).

            References
            ----------
                .. S. Gowal, K. Dvijotham, R. Stanforth, R. Bunel, C. Qin, J. Uesato, R. Arand-
                   jelovi´c, T. Mann, and Sigma. Kohli. "On the effectiveness of interval bound propa-
                   gation for training verifiably robust models. arXiv:1810.12715, 2018."
            """
            return (a @ (ub + lb) - np.abs(a) @ (ub - lb)) / 2, (
                a @ (ub + lb) + np.abs(a) @ (ub - lb)
            ) / 2

        out_lb, out_ub = np.zeros(self.ndim), np.zeros(self.ndim)
        for k, v in self.decomposition_dict.items():
            _lb, _ub = _interval_bound_propagation(v, k.lb, k.ub)

            if np.any(_lb > _ub):
                raise ValueError("Lower bound cannot exceed upper bound .")

            out_lb += _lb
            out_ub += _ub

        # Add constant term to bounds
        out_lb += self.const
        out_ub += self.const

        return out_lb, out_ub

    def eval(self):
        # TODO
        raise NotImplementedError("LinExpr.eval() needs to be implemented.")


def stack(exprs: list[LinearExpression]) -> LinearExpression:
    r"""
    Concatenate ``LinearExpression`` objects along axis 0 into a single view.

    The result shares storage with its components: no new optimization
    variables are introduced and no equality constraint of the form
    ``y == [e_1; ...; e_n]`` is added. Any constraint or bound update on a
    leaf appearing in ``exprs`` automatically applies to the corresponding
    slice of the returned stack, since canonicalizers walk the result's
    ``decomposition_dict`` directly to the underlying leaves.

    Parameters
    ----------
    exprs : list[LinearExpression]
        Components to stack. Each must be a valid ``LinearExpression``
        (leaf or derived). Components may share leaves
        (e.g. ``stack([x, 2*x])``); shared leaves are deduplicated in the
        result's ``decomposition_dict`` via summation of their composed
        coefficient matrices.

    Returns
    -------
    LinearExpression
        A non-leaf ``LinearExpression`` of dimension
        ``sum(e.ndim for e in exprs)``. ``result.const`` is the
        concatenation of each component's ``const``; ``result.lb`` / ``result.ub``
        are computed once at construction by interval propagation from the
        component leaves.

    Raises
    ------
    ValueError
        If ``exprs`` is empty.

    Warnings
    --------
    The stack's ``lb`` / ``ub`` attributes are a **snapshot** taken at
    construction time. They do *not* refresh automatically when a
    component leaf's bounds are tightened later (for example by an OBBT
    pass triggered by ``add_constraint``). Any consumer that needs
    up-to-date stack bounds must explicitly recompute them via::

        stack_expr.lb, stack_expr.ub = stack_expr.bound_propagation()

    This matters most in the dual / KKT use case (the primary motivator
    for ``stack``): if the stack is reused across multiple constraint
    additions, stale snapshot bounds can cause downstream OBBT passes
    to under-tighten. Canonicalizer constraint emission is unaffected
    — it walks ``decomposition_dict`` straight to the underlying leaves,
    whose bounds *are* always current — so this caveat applies only to
    callers that read ``stack_expr.lb`` / ``stack_expr.ub`` directly.
    """
    if len(exprs) == 0:
        raise ValueError("stack() requires at least one expression")

    total_dim = sum(e.ndim for e in exprs)
    decomposition_dict: dict = {}
    const = np.zeros(total_dim)
    offset = 0
    for e in exprs:
        rows = np.arange(offset, offset + e.ndim)
        cols = np.arange(e.ndim)
        sel = spa.csc_matrix(
            (np.ones(e.ndim), (rows, cols)),
            shape=(total_dim, e.ndim),
        )
        for leaf, coeff in e.decomposition_dict.items():
            composed = sel @ coeff
            if leaf in decomposition_dict:
                composed = decomposition_dict[leaf] + composed
            # Keep sparse + sparse sparse; anything touching a dense operand becomes a
            # plain ndarray (sparse + ndarray returns np.matrix, which breaks bound_propagation)
            decomposition_dict[leaf] = composed if spa.issparse(composed) else np.asarray(composed)
        const[offset : offset + e.ndim] = e.const
        offset += e.ndim

    # Drop entries whose composed coefficient is all-zero — possible when a
    # component's leaf coefficient is supported only on rows masked out by
    # the selection matrix. Pruning prevents canonicalizers walking the
    # result's decomposition_dict from looking up irrelevant leaves.
    result = LinearExpression(decomposition_dict=prune_dict(decomposition_dict), const=const)
    result.lb, result.ub = result.bound_propagation()
    return result


def solve_linexpr(M, expr: LinearExpression) -> LinearExpression:
    r"""
    Build a ``LinearExpression`` representing :math:`M^{-1}\,\text{expr}`
    without forming :math:`M^{-1}` explicitly.

    Each leaf coefficient and the constant are computed via
    ``np.linalg.solve(M, ...)`` (dense ``M``) or an LU-factorized
    ``splu.solve(...)`` (sparse ``M``) — the same pattern as
    :attr:`fomtools.steps.affine.Affine.output_mat_inverse`.

    The returned ``LinearExpression`` is intended for **interval-propagating
    bounds through an implicit inverse**; it should not be registered with
    a canonicalizer (the verification model should post ``M @ u == expr``
    instead, leaving the solve to the solver).

    Parameters
    ----------
    M : np.ndarray or scipy.sparse matrix
        Square invertible matrix.
    expr : LinearExpression
        Expression to be "left-multiplied" by ``M^{-1}``. Its
        ``decomposition_dict`` and ``const`` are mapped through the
        per-coefficient solve.

    Returns
    -------
    LinearExpression
        Throwaway expression of dimension equal to ``M.shape[0]``.
    """

    def _to_dense(b):
        return np.asarray(b.todense()) if spa.issparse(b) else np.asarray(b)

    if spa.issparse(M):
        lu = spa.linalg.splu(M.tocsc())

        def solve(b):
            return lu.solve(_to_dense(b))
    else:

        def solve(b):
            return np.asarray(np.linalg.solve(M, _to_dense(b)))

    # Preserve coefficient rank: ``solve`` returns a 2-D ``(n, k)`` block
    # for each 2-D leaf coefficient and a 1-D ``(n,)`` vector for the
    # constant. ``np.squeeze`` would collapse a ``(n, 1)`` block (for
    # 1-dim leaves) into ``(n,)``, breaking the ``a @ (ub + lb)`` matmul
    # in ``bound_propagation``; pin the ranks instead.
    return LinearExpression(
        decomposition_dict={k: np.atleast_2d(solve(v)) for k, v in expr.decomposition_dict.items()},
        const=np.atleast_1d(solve(expr.const)),
    )
