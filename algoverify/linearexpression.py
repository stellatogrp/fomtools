from typing import Union

import numpy as np
import scipy.sparse as spa

from algoverify.dict_utils import merge_dict, prune_dict


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

        decomposition_dict : dict
            A mapping from each leaf expression to its corresponding coefficient matrix. If
            `is_leaf` is True, this dict maps the object to its own coefficient matrix.

        counter : int
            Tracks the number of leaf `LinearExpression` objects.

    References
    ----------
        .. B. Goujaud, C. Moucer, F. Glineur, J. Hendrickx, A. Taylor, A. Dieuleveut (2022).
           "PEPit: computer-assisted worst-case analyses of first-order optimization methods
           in Python."

    """

    counter = 0
    # this is necessary else numpy will not even look at the @ operator due to precedence
    __array_priority__ = 1000

    def __init__(self, decomposition_dict: dict, const: np.ndarray = None):
        """
        Initialize a `LinearExpression` object with decomposition dictionary.

        Parameters
        ----------
        decomposition_dict : dict
            A mapping from each leaf `LinearExpression` to its coefficient matrix (an nxn sparse
            matrix).

        const : ndarray or None, optional
            A one-dimensional array or None (default). If `const` is None, the `const` attribute
            of this object will be set to a `numpy.ndarray` of length `ndim` filled with zeros.


        Raises
        ------
        ValueError
            If all the coefficient matrices in the decomposition dict doesn't have the same number
            of rows.

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
        self.const = const
        self.decomposition_dict = decomposition_dict
        self.counter = None
        self.is_leaf = False

    @classmethod
    def from_int(cls, ndim: int) -> "LinearExpression":
        """
        Initialize a `LinearExpression` object with an integer `ndim`. Internally, this
        creates a decomposition dictionary that maps the `LinearExpression` object with a sparse
        identity matrix of shape `(ndim, ndim)`.

        Parameters
        ----------
        ndim : int
            The dimension of underlying space, i.e., the size of each coefficient matrix.

        Returns
        -------
        LinearExpression
            A new `LinearExpression` instance whose decomposition contains a single
            entry: the instance itself as a key, and a SciPy sparse identity matrix of
            shape `(ndim, ndim)` as the value.
        """
        obj = cls.__new__(cls)  # uninitialized instance
        obj.decomposition_dict = {obj: spa.eye(ndim)}
        obj.is_leaf = True
        obj.ndim = ndim
        obj.counter = cls.counter
        obj.const = np.zeros(ndim)
        cls.counter += 1
        return obj

    def __str__(self):
        return f"Var{self.counter}"

    def __repr__(self):
        return f"<LinearExpression {'Leaf' if self.is_leaf else ''}, dim={self.ndim}>"

    def __add__(self, other: "LinearExpression") -> "LinearExpression":
        """
        Add two `LinearExpression` objects, ensuring they have the same dimension.

        Parameters
        ----------
        other : LinearExpression
            The expression to add to this one.

        Returns
        -------
        LinearExpression
            A new `LinearExpression` object whose decomposition maps each leaf in the union of both
            expressiom to the sum of its coefficient matrices, treating any leaf missing from one
            operand as having a zero coefficient matrix.

        Raises:
        ------
        ValueError
            If the `LinearExpression` objects to be added together does not have the same dimension.
        """
        if self.ndim != other.ndim:
            raise ValueError(f"""Cannot add LinearExpression objects with mismatched dimensions:
                             {self.ndim} != {other.ndim} """)

        merged_decomposition_dict = merge_dict(self.decomposition_dict, other.decomposition_dict)
        return LinearExpression(decomposition_dict=prune_dict(merged_decomposition_dict), const=self.const + other.const)

    def __mul__(self, scalar: Union[float, int]) -> "LinearExpression":
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
                const=self.const * float(scalar)
            )
        except ValueError:
            raise TypeError(
                f"""Can only multiply by int or float, got {type(scalar)}. If intending to use
                 a matrix, use the @ symbol with the matrix on the left."""
            )

    def __rmul__(self, scalar: Union[float, int]) -> "LinearExpression":
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

    def __truediv__(self, scalar: Union[float, int]) -> "LinearExpression":
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

    def __neg__(self) -> "LinearExpression":
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

    def __sub__(self, other: "LinearExpression") -> "LinearExpression":
        """
        Subtract one LinearExpression from another.

        Parameters
        ----------
        other : LinearExpression
            The expression to subtract from this one.

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

        """
        return self + (-other)

    def __matmul__(self, other):
        # TODO
        raise NotImplementedError("np matrix or sp matrix should be on the left of the @")

    def __rmatmul__(self, matrix: Union[np.ndarray, spa.spmatrix]) -> "LinearExpression":
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
                const= np.asarray(matrix @ self.const).ravel()
            )
        except TypeError:
            raise TypeError(
                f"""Expected numpy.ndarray or scipy.sparse matrix for multiplication;
                got {type(matrix).__name__}. """
            )

    def get_output_dim(self) -> int:
        """
        Return the dimension of this `LinearExpression`. The dimension is given by
        the number of rows of the coefficient matrices in `decomposition_dict`.
        """

        return [mat.shape[0] for mat in self.decomposition_dict.values()][0]

    def eval(self):
        # TODO
        raise NotImplementedError("LinExpr.eval() needs to be implemented.")
