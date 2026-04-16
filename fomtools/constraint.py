from typing import TYPE_CHECKING, Union

import numpy as np

if TYPE_CHECKING:
    from fomtools.linearexpression import LinearExpression


class Constraint:
    r"""
    A symbolic relation between two expressions (or an expression and a vector),
    of the form:

    $$
        \text{lhs} \;\mathrm{op}\; \text{rhs},
    $$

    where ``op`` is one of ``"=="``, ``"<="``, or ``">="``. Instances are
    typically created by using comparison operators on expression object(s)
    (e.g., ``expr1 <= expr2``) rather than by calling the constructor directly.

    Attributes
    ----------
    lhs: LinearExpression
        Left-hand side linear expression in the constraint definition.
    rhs: LinearExpression, np.ndarray, or list
        Right-hand side linear expression in the constraint definition. Scalars are
        broadcast to the shape of the `lhs` expression. Arrays or expressions must
        be of same shape with `self`.
    operator: str {'<=', '>=', '=='}
        The relational operator defining the constraint.
    ndim: int
        The dimension of underlying space, i.e., the size of each coefficient matrix
        in the linear expressions.
    """

    def __init__(
        self, lhs: "LinearExpression", rhs: Union["LinearExpression", np.ndarray, list], operator
    ):
        """
        Initize an instance of `Constraint` class.

        Parameters
        ----------
        lhs: LinearExpression
            Left-hand side linear expression in the constraint definition.

        rhs: LinearExpression
            Right-hand side linear expression in the constraint definition. Scalars are
            broadcast to the shape of the `lhs` expression. Arrays or expressions must
            be of same shape with `self`.

        operator: str {'<=', '>=', '=='}
            The relational operator defining the constraint.

        Raises
        ------
        ValueError
            If the expressions `lhs` and `rhs` are not of the same shape.

        TypeError
           If `rhs` is not a LinearExpression object, NumPy array or a list.
        """
        try:
            # If rhs is a scalar, broadcast it to the dimension of the lhs expression
            rhs = np.full(lhs.ndim, float(rhs))
        except TypeError:
            try:
                # Check to see if rhs is of type list or a numpy array
                if len(rhs) == lhs.ndim:
                    rhs = np.array(rhs)
                else:
                    raise ValueError("Cannot have a constraint of mismatched dimensions. ")
            except TypeError:
                try:
                    # Check to see if other is of type LinearExpression
                    if lhs.ndim != rhs.ndim:
                        raise ValueError("Cannot have a constraint of mismatched dimensions. ")
                except AttributeError:
                    raise TypeError(
                        f"""Can only create a constraint of a LinearExpression object,"
                         with a scalar, numpy array, list, or another LinearExpression object got type {type(rhs)}."""
                    )

        self.lhs = lhs - rhs
        self.rhs = np.zeros_like(self.lhs.ndim)
        self.operator = operator
        self.ndim = lhs.ndim

    def __str__(self):
        return f"{self.lhs} {self.operator} {self.rhs}"

    def __repr__(self):
        return f"<{self.__class__.__name__}, dim={self.ndim}>"


class EqualityConstraint(Constraint):
    r"""
    An equality relation between two expressions (or an expression and a vector),
    of the form.

    $$
        \text{lhs} \;=\; \text{rhs},
    $$

    and is typically created via the ``==`` operator on expression objects
    (e.g., ``expr1 == expr2``) rather than by calling the constructor directly.

    Attributes
    ----------
    lhs: LinearExpression
        Left-hand side linear expression in the constraint definition.
    rhs: LinearExpression, np.ndarray, or list
        Right-hand side linear expression in the constraint definition. Scalars are
        broadcast to the shape of the `lhs` expression. Arrays or expressions must
        be of same shape with `self`.
    operator: str {'<=', '>=', '=='}
        The relational operator defining the constraint.
    ndim: int
        The dimension of underlying space, i.e., the size of each coefficient matrix
        in the linear expressions.
    """

    def __init__(self, lhs, rhs):
        """
        Initize an instance of :class:`InequalityConstraint` class.

        Parameters
        ----------
        lhs: LinearExpression
            Left-hand side linear expression in the constraint definition.

        rhs: LinearExpression
            Right-hand side linear expression in the constraint definition. Scalars are
            broadcast to the shape of the `lhs` expression. Arrays or expressions must
            be of same shape with `self`.
        """
        operator = "="
        super().__init__(lhs, rhs, operator)


class InequalityConstraint(Constraint):
    r"""
    An non-strict inequality relation between two expressions (or an expression and a vector),
    of the form.

    $$
        \text{lhs} \;<=\; \text{rhs},
    $$

    and is typically created via the ``<=`` operator on expression objects
    (e.g., ``expr1 <= expr2``) rather than by calling the constructor directly.

    Attributes
    ----------
    lhs: LinearExpression
        Left-hand side linear expression in the constraint definition.
    rhs: LinearExpression, np.ndarray, or list
        Right-hand side linear expression in the constraint definition. Scalars are
        broadcast to the shape of the `lhs` expression. Arrays or expressions must
        be of same shape with `self`.
    operator: str {'<=', '>=', '=='}
        The relational operator defining the constraint.
    ndim: int
        The dimension of underlying space, i.e., the size of each coefficient matrix
        in the linear expressions.
    """

    def __init__(self, lhs, rhs):
        """
        Initize an instance of `InequalityConstraint` class.

        Parameters
        ----------
        lhs: LinearExpression
            Left-hand side linear expression in the constraint definition.

        rhs: LinearExpression
            Right-hand side linear expression in the constraint definition. Scalars are
            broadcast to the shape of the `lhs` expression. Arrays or expressions must
            be of same shape with `self`.
        """
        operator = "<="
        super().__init__(lhs, rhs, operator)
