from fomtools.solvers import gurobi, highs
from fomtools.solvers.solver import Solver
from fomtools.steps.step import Step


class Canonicalizer:
    """
    A base class for mathematical-optimization methods. This class
    defines a common interface for mixed-integer programming (MIP),
    and semidefinite programming (SDP).

    Attributes
    ----------
    solver: Solver
        A ready-to-use instance of a subclass of :class:`Solver`
        (e.g., ``gurobi.Gurobi()``).

    """

    def __init__(self, solver_params: dict = None, solver: str = "gurobi"):
        """
        Initialize an instance of :class:`Method`.

        Parameters
        ----------
        solver_params: dict or None, optional
            Key-value pairs with solver parameters and their corresponding
            configuration values.

        solver: str, default "gurobi"
            Case-insensitive identifier for the solver to use. Supported
            values (and common aliases) include:
            - ``"gurobi"``, ``"grb"``
        """
        self.solver_params = solver_params
        self.solver = self._get_solver(solver, solver_params)

    @staticmethod
    def _get_solver(solver: str, solver_params: dict) -> Solver:
        """
        Return an initialized solver instance for the requested backend.

        Parameters
        ----------
        solver: str
            Case-insensitive identifier for the solver to use. Supported values
            (and common aliases) include:
            - ``"gurobi"``, ``"grb"``
            - ``"highs"``

        solver_params: dict or None, optional
            Key-value pairs with solver parameters and their corresponding
            configuration values.

        Returns
        -------
        Solver
            A ready-to-use instance of a subclass of :class:`Solver`
            (e.g., ``gurobi.Gurobi()``).

        Raises
        ------
        ValueError
            If `solver` is not one of the supported options.
        """
        match solver.lower():
            case "gurobi" | "grb":
                return gurobi.GurobiSolver(solver_params)
            case "highs":
                return highs.HiGHSSolver(solver_params)
            case _:
                raise ValueError(f"Solver must be 'gurobi' or 'TBD', got {solver}.")

    def canonicalize(self, step: Step):
        """
        Dispatch the provided step into a backend-specific encoding.

        This method converts a `Step` instance (e.g., ``Relu``, ``SoftThreshold``,
        ``Affine``) into a collection of backend-specific constraints for the
        underlying optimization problem. The canonicalizer inspects the concrete
        type of ``step`` and forwards the call to a method of the form
        ``canonicalize_<step_class_name>()`` defined on the subclasses,
        for example ``canonicalize_relu`` or ``canonicalize_soft_threshold``.

        Parameters
        ----------
        step: Step
            The concrete step class to canonicalize. This class encapsulates
            the thresholding or activation operator.

        Returns
        -------
        None

        Raises
        ------
        NotImplementedError
            If this canonicalizer does not provide an encoder for the concrete
            type of ``step`` (i.e. no matching ``canonicalize_<step_class_name>`` method
            is implemented).

        """
        method_name = f"canonicalize_{type(step).__name__.lower()}"
        try:
            canonicalizer = getattr(self, method_name)
        except AttributeError:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support {type(step).__name__}"
            )
        canonicalizer(step)

    def add_var(self, iterate):
        raise NotImplementedError
