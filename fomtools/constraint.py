from typing import TYPE_CHECKING, Union

import numpy as np

if TYPE_CHECKING:
    from fomtools.linearexpression import LinearExpression


def is_binary_free(constraint: "Constraint") -> bool:
    """
    Return ``True`` if the constraint involves no binary leaf variables.

    A constraint is "binary-free" iff none of the leaf expressions in
    ``constraint.lhs.decomposition_dict`` is flagged ``is_binary``. Such
    constraints are safe to add to the OBBT handler's LP (which must remain a
    pure linear program). Binary leaves are created via
    :meth:`LinearExpression.from_int` with ``is_binary=True`` (e.g. the big-M
    indicator variables in the MIP canonicalizer).

    Parameters
    ----------
    constraint : Constraint
        The constraint to inspect.

    Returns
    -------
    bool
        ``True`` if no leaf in the constraint is binary, ``False`` otherwise.
    """
    return not any(
        getattr(leaf, "is_binary", False)
        for leaf in constraint.lhs.decomposition_dict
    )


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
    is_linear: bool
        ``True`` for ordinary (in)equality constraints, which are linear and safe to
        forward to the OBBT handler's LP. Non-convex subclasses (e.g.
        :class:`ComplementarityConstraint`) override this to ``False``.
    """

    is_linear = True

    def __init__(
        self, lhs: "LinearExpression", rhs: Union["LinearExpression", np.ndarray, list], operator
    ):
        """
        Initialize an instance of `Constraint` class.

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
        Initialize an instance of :class:`EqualityConstraint` class.

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
        Initialize an instance of `InequalityConstraint` class.

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


class ComplementarityConstraint(Constraint):
    r"""
    Elementwise complementarity between two non-negative expressions:

    $$
        e_1 \ge 0,\quad e_2 \ge 0,\quad e_1 \odot e_2 = 0,
    $$

    i.e. for every coordinate $i$ at most one of $e_{1,i}, e_{2,i}$ is nonzero.

    Unlike :class:`EqualityConstraint`/:class:`InequalityConstraint`, this relation is
    not reducible to a single ``lhs op 0`` expression, so it stores both expressions
    verbatim (it does **not** invoke ``Constraint.__init__``). It is non-convex and is
    lowered to a backend-specific encoding (big-M + binaries in the MIP canonicalizer).
    Instances are typically created via :func:`complementarity` or
    ``Verifier.complementarity`` rather than by the constructor directly.

    Attributes
    ----------
    expr1, expr2 : LinearExpression
        The two expressions held in complementarity. Must have equal ``ndim``.
    operator : str
        Always ``"complementarity"``.
    ndim : int
        Dimension of each expression.
    """

    is_linear = False  # router guard: never forward to the OBBT LP

    def __init__(self, expr1: "LinearExpression", expr2: "LinearExpression"):
        """
        Initialize a :class:`ComplementarityConstraint`.

        Parameters
        ----------
        expr1, expr2 : LinearExpression
            The two expressions held in complementarity.

        Raises
        ------
        ValueError
            If ``expr1`` and ``expr2`` do not have the same dimension.
        """
        if expr1.ndim != expr2.ndim:
            raise ValueError(
                f"complementarity requires equal dimensions, got {expr1.ndim} and {expr2.ndim}."
            )
        # Intentionally bypass Constraint.__init__: keep both expressions intact.
        self.expr1 = expr1
        self.expr2 = expr2
        self.operator = "complementarity"
        self.ndim = expr1.ndim

    def __repr__(self):
        return f"<ComplementarityConstraint, dim={self.ndim}>"


def complementarity(
    expr1: "LinearExpression", expr2: "LinearExpression"
) -> ComplementarityConstraint:
    """
    Build an elementwise complementarity constraint ``expr1 ⊥ expr2`` (both ``≥ 0``).

    Parameters
    ----------
    expr1, expr2 : LinearExpression
        The two expressions to place in complementarity. Must have equal ``ndim``.

    Returns
    -------
    ComplementarityConstraint
        The constraint object; pass it to ``Verifier.add_constraint``.
    """
    return ComplementarityConstraint(expr1, expr2)
