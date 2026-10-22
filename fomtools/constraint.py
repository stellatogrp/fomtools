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
    `LinearExpression.from_int` with ``is_binary=True`` (e.g. the big-M
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
    return not any(getattr(leaf, "is_binary", False) for leaf in constraint.lhs.decomposition_dict)


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
        `ComplementarityConstraint`) override this to ``False``.
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
        Initialize an instance of `EqualityConstraint` class.

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

    Unlike `EqualityConstraint`/`InequalityConstraint`, this relation is
    not reducible to a single ``lhs op 0`` expression, so it stores both expressions
    verbatim (it does **not** invoke ``Constraint.__init__``). It is non-convex and is
    lowered to a backend-specific encoding (big-M + binaries in the MIP canonicalizer).
    Instances are typically created via ``ft.complementarity`` rather than by the
    constructor directly.

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
        Initialize a `ComplementarityConstraint`.

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


class QuadraticConstraint(Constraint):
    r"""
    A sum-of-squares (quadratic) relation between a linear expression and a scalar:

    $$
        \sum_i (a_i^\top x + b_i)^2 \;\mathrm{op}\; r^2,
        \qquad \mathrm{op} \in \{\le, \ge, =\},
    $$

    i.e. ``op`` compares the squared two-norm of ``expr`` against a nonnegative scalar
    ``radius_sq``. Like `ComplementarityConstraint`, this is not a single
    ``lhs op 0`` linear relation, so it stores the expression verbatim (it does **not**
    invoke ``Constraint.__init__``) and sets ``is_linear = False`` so it is never
    forwarded to the OBBT LP. The ``<=`` sense is convex; ``>=`` / ``==`` are nonconvex
    (representable on the SDP backend, which is linear in the lifted matrix, and on
    Gurobi via ``NonConvex=2``; unsupported on HiGHS).

    Instances are typically created via ``sum_squares(expr) <= r**2`` (and ``>=`` / ``==``)
    rather than by calling the constructor directly.

    Attributes
    ----------
    expr : LinearExpression
        The expression whose squared two-norm is bounded.
    radius_sq : float
        The nonnegative scalar right-hand side (e.g. ``r**2``).
    sense : str
        One of ``"<="``, ``">="``, ``"=="``.
    is_convex : bool
        ``True`` iff ``sense == "<="``.
    operator : str
        Always ``"sum_squares"`` (used by the backend dispatch).
    ndim : int
        Dimension of ``expr``.
    """

    is_linear = False  # router guard: never forward to the OBBT LP

    def __init__(self, expr: "LinearExpression", radius_sq, sense: str):
        """
        Initialize a `QuadraticConstraint`.

        Parameters
        ----------
        expr : LinearExpression
            The expression whose squared two-norm is bounded.
        radius_sq : float
            The nonnegative scalar right-hand side. Must be ``>= 0``.
        sense : str {"<=", ">=", "=="}
            The relational operator comparing ``sum_squares(expr)`` to ``radius_sq``.

        Raises
        ------
        ValueError
            If ``radius_sq`` is negative or ``sense`` is not one of the allowed values.
        """
        if sense not in ("<=", ">=", "=="):
            raise ValueError(f"sum_squares sense must be one of <=, >=, ==, got {sense}.")
        radius_sq = float(radius_sq)
        if radius_sq < 0:
            raise ValueError(
                f"sum_squares right-hand side must be nonnegative (a squared radius), "
                f"got {radius_sq}."
            )
        # Intentionally bypass Constraint.__init__: keep the expression intact.
        self.expr = expr
        self.radius_sq = radius_sq
        self.sense = sense
        self.is_convex = sense == "<="
        self.operator = "sum_squares"
        self.ndim = expr.ndim

    def __repr__(self):
        return f"<QuadraticConstraint, sum_squares {self.sense} {self.radius_sq}, dim={self.ndim}>"


class SumSquares:
    r"""
    Symbolic atom representing the sum of squared entries of a linear expression,
    ``sum_squares(expr) = \sum_i (a_i^\top x + b_i)^2`` (the squared two-norm).

    This object is not a `LinearExpression` (it is quadratic). Its only purpose
    is to build a `QuadraticConstraint` via a comparison operator, e.g.
    ``ft.sum_squares(expr) <= r**2``. Created via ``ft.sum_squares``.

    Attributes
    ----------
    expr : LinearExpression
        The wrapped linear expression.
    """

    __hash__ = None  # transient builder; defining __eq__ would otherwise drop hashing

    def __init__(self, expr: "LinearExpression"):
        self.expr = expr

    def __le__(self, bound) -> QuadraticConstraint:
        return QuadraticConstraint(self.expr, bound, "<=")

    def __ge__(self, bound) -> QuadraticConstraint:
        return QuadraticConstraint(self.expr, bound, ">=")

    def __eq__(self, bound) -> QuadraticConstraint:
        return QuadraticConstraint(self.expr, bound, "==")

    def __repr__(self):
        return f"<SumSquares, dim={self.expr.ndim}>"


def sum_squares(expr: "LinearExpression") -> SumSquares:
    r"""
    Build a sum-of-squares atom ``\sum_i (a_i^\top x + b_i)^2`` for use in a quadratic
    constraint.

    Compare the returned object against a nonnegative scalar to form a
    `QuadraticConstraint`::

        vp.add_constraint(ft.sum_squares(expr) <= r**2)   # ||A x + b||_2^2 <= r^2
        vp.add_constraint(ft.sum_squares(expr) >= r**2)   # nonconvex shell
        vp.add_constraint(ft.sum_squares(expr) == r**2)   # nonconvex sphere

    Parameters
    ----------
    expr : LinearExpression
        The expression whose squared two-norm is taken.

    Returns
    -------
    SumSquares
        The atom; compare it with ``<=`` / ``>=`` / ``==`` to obtain a constraint.
    """
    return SumSquares(expr)


class Norm:
    r"""
    Symbolic norm atom ``\|expr\|_p`` used as an objective in `Verifier.maximize`.

    A list of expressions stands for the norm of their stacked vector. Created via
    `norm`.

    Attributes
    ----------
    exprs : tuple[LinearExpression, ...]
        The wrapped linear expressions.
    p : int or float
        The norm order, ``1`` or ``np.inf``.
    """

    def __init__(self, exprs: tuple, p: int | float):
        self.exprs = exprs
        self.p = p

    def __repr__(self):
        return f"<Norm, p={self.p}, dim={sum(e.ndim for e in self.exprs)}>"

    def _not_a_constraint(self, other):
        raise TypeError(
            "ft.norm is an objective for Verifier.maximize, not a constraint. "
            "Constrain the expression itself (expr <= u, expr >= l) or use "
            "ft.sum_squares(expr) <= r**2."
        )

    __le__ = __ge__ = __lt__ = __gt__ = _not_a_constraint


def norm(expr, p) -> Norm:
    r"""
    Build a norm atom ``\|expr\|_p`` to maximize with `Verifier.maximize`::

        vp.maximize(ft.norm(z - z_prev, np.inf))

    Parameters
    ----------
    expr : LinearExpression or Sequence[LinearExpression]
        The expression, or a list of expressions whose stacked vector is taken.
    p : {1, np.inf, 'inf'}
        The norm order. For the squared two-norm, maximize ``ft.sum_squares(expr)``.

    Returns
    -------
    Norm
        The atom.

    Raises
    ------
    ValueError
        If ``p`` is not ``1`` or infinity.
    """
    if p == 2:
        raise ValueError(
            "The two-norm objective is squared: maximize ft.sum_squares(expr) instead "
            "(with method='mip' it is an MIQP and needs solver='gurobi')."
        )
    if p == "inf":
        p = np.inf
    if p not in (1, np.inf):
        raise ValueError(f"Unsupported norm order: {p}. Use 1 or np.inf.")
    exprs = tuple(expr) if isinstance(expr, (list, tuple)) else (expr,)
    return Norm(exprs, p)
