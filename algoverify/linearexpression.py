import scipy.sparse as spa

from algoverify.dict_utils import merge_dict, prune_dict
from algoverify.mat_utils import is_valid_mat
import numpy as np


class LinearExpression:
    """
    A linear combination of function values, inner products (between points and/ or gradients)
    and constant scalars.

    Attributes:
    ----------
        n : int
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
    list_of_leaf_LinExprs = list()  # TODO: figure out if we need these
    __array_priority__ = (
        1000  # this is necessary else numpy will not even look at the @ operator due to precedence
    )

    def __init__(self, n: int, decomposition_dict: dict = None):
        """
        Initialize a `LinearExpression` object.

        Parameters
        ----------
        n : int
            The dimension of underlying space, i.e., the size of each coefficient matrix.

        decomposition_dict : dict or None, optional
            Mapping from each leaf `LinearExpression` to its coefficient matrix (an nxn sparse
            matrix). If `decomposition_dict` is None, this object is treated as a leaf and its
            decomposition is set to :: {self: I_n} :: where `I_n` is the nxn sparse identity
            matrix.

        """
        self.n = n

        if decomposition_dict is None:
            self.decomposition_dict = {self: spa.eye(n)}
            self.counter = LinearExpression.counter
            LinearExpression.counter += 1
            LinearExpression.list_of_leaf_LinExprs.append(self)

            assert self.decomposition_dict[self].shape[1] == n
            self.is_leaf = True
        else:
            assert isinstance(decomposition_dict, dict)
            self.decomposition_dict = decomposition_dict
            self.counter = None
            self.is_leaf = False

    def __str__(self):
        return f"Var{self.counter}"

    def __repr__(self):
        return f"<LinearExpression {'Leaf' if self.is_leaf else ''}, dim={self.n}>"

    def __add__(self, other):
        assert self.n == other.n
        merged_decomposition_dict = merge_dict(self.decomposition_dict, other.decomposition_dict)

        return LinearExpression(self.n, decomposition_dict=prune_dict(merged_decomposition_dict))

    def __rmul__(self, other):
        try:
            return LinearExpression(
                n=self.n,
                decomposition_dict={
                    k: v * float(other) for k, v in self.decomposition_dict.items()
                },
            )
        except:
            raise TypeError(
                f"""LinExprs can only be multiplied by scalar values, got {type(other)}.
                If intending to use a matrix, use the @ symbol with the matrix on the left"""
            )

    def __mul__(self, other):
        return self.__rmul__(other=other)

    def __truediv__(self, other):
        return self.__rmul__(1 / other)

    def __neg__(self):
        return self.__rmul__(other=-1)

    def __sub__(self, other):
        return self.__add__(-other)

    def __matmul__(self, other):
        # TODO
        raise NotImplementedError("np matrix or sp matrix should be on the left of the @")

    def __rmatmul__(self, other):
        if is_valid_mat(other):
            new_decomposition_dict = dict()
            for key, value in self.decomposition_dict.items():
                new_decomposition_dict[key] = other @ value
            return LinearExpression(other.shape[0], decomposition_dict=new_decomposition_dict)
        # TODO
        raise NotImplementedError(
            f"Object on the left should be np or sp.sparse matrix. Got {type(other)}"
        )

    def get_output_dim(self):
        shapes = [v.shape[0] for _, v in self.decomposition_dict.items()]
        # TODO: Perhaps this can be checked when the decomposition_dict is initialized or updated
        assert np.allclose(shapes, shapes[0])
        return shapes[0]

    def eval(self):
        # TODO
        raise NotImplementedError("LinExpr.eval() needs to be implemented.")
