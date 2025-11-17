from typing import Self, Union

import numpy as np
import scipy.sparse as spa

from algoverify.constraint import EqualityConstraint, InequalityConstraint
from algoverify.utils.dict_utils import merge_dict, prune_dict


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
            if const.ndim != 1:
                raise ValueError("Matrix for a constant must be 1-dimensional.")
            if 0 < self.ndim != const.shape[0]:
                raise ValueError(
                    "Matrix for a constant must have same number of rows as decomposition dict."
                )
            # in case the only remaining term in an expression is a constant
            elif self.ndim == 0:
                self.ndim = const.shape[0]

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
        obj.lb = 0 if is_binary else lb
        obj.ub = 1 if is_binary else ub
        cls.counter += 1
        return obj

    def update_bounds(self, lb, ub) -> None:
        self.lb = np.maximum(self.lb, lb)
        self.ub = np.minimum(self.ub, ub)

    def __str__(self):
        return f"Var{self.counter}"

    def __repr__(self):
        return f"<LinearExpression {'Leaf' if self.is_leaf else ''}, dim={self.ndim}>"

    def __hash__(self):
        """
        The class variable `counter` is used to assign each `LinearExpression` a unique,
        monotonically increasing ID at construction. This ID can serve as an
        identity-based hash.
        """
        return hash(str(self))

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
                        k: np.atleast_2d((v.toarray() if spa.issparse(v) else v)[item])
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
                const=self.const + np.full(self.ndim, float(other)),
            )
        except TypeError:
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
                    k: v * float(scalar) for k, v in self.decomposition_dict.items()
                },
                const=self.const * float(scalar),
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

        return out_lb, out_ub

    def get_output_dim(self) -> int:
        """
        Return the dimension of this `LinearExpression`. The dimension is given by
        the number of rows of the coefficient matrices in `decomposition_dict`.
        """

        return [mat.shape[0] for mat in self.decomposition_dict.values()][0]

    def eval(self):
        # TODO
        raise NotImplementedError("LinExpr.eval() needs to be implemented.")

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
