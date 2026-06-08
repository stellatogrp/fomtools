from typing import ClassVar

from fomtools.linearexpression import LinearExpression
from fomtools.stepdecomposers.step_decomposer import StepDecomposer
from fomtools.steps.step import Step


class SimulatorStepDecomposer(StepDecomposer):
    """
    Simulator Step Decomposer.

    Currently forwards all steps as primitive operations, as they are handled
    directly by the Simulator canonicalizer. This may evolve if future proxable
    steps require explicit decomposition.
    """

    def _process_primitive(self, step: Step) -> LinearExpression:
        # Skip sync_lhs_bounds — the simulator evaluates numerically and
        # handles affine solves itself via canonicalize_affine.
        for leaf in step.lhs_leaves:
            self.register_var(leaf, var_type="param")
        self.canonicalizer.canonicalize(step)
        return step.lhs

    def _implicit_iterate_not_supported(self, prox, rhs, gamma=1.0):
        # TODO(simulator): Implement implicit-iterate proxes on Simulator.
        # Shared deferral for prox lowerings that register a fresh iterate
        # defined implicitly by an equality constraint (Quadratic,
        # LeastSquares, IndAffine, …). The Simulator backend has no
        # per-step activation callback to compute the iterate's value per
        # sample — fixing this would need a per-sample np.linalg.solve on
        # the prox-specific linear system. The error message reports the
        # concrete class via ``type(prox).__name__``.
        raise NotImplementedError(
            f"{type(prox).__name__} prox is not yet supported on the Simulator backend"
        )

    _PROX_DECOMPOSERS: ClassVar[dict] = {
        **StepDecomposer._PROX_DECOMPOSERS,
        "quadratic": _implicit_iterate_not_supported,
        "leastsquares": _implicit_iterate_not_supported,
        "indaffine": _implicit_iterate_not_supported,
        "indpolyhedral": _implicit_iterate_not_supported,
        "indballl1": _implicit_iterate_not_supported,
        "norminf": _implicit_iterate_not_supported,
        "indsimplex": _implicit_iterate_not_supported,
    }
