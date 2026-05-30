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

    def prox_quadratic(self, prox, rhs, gamma=1.0):
        # TODO(simulator): Implement Quadratic prox on the Simulator backend.
        # The prox is defined implicitly by (I + gamma*Q) u == rhs - gamma*q,
        # which has no per-step activation callback the simulator can use.
        # Implementation requires solving the linear system per sample
        # (e.g., via np.linalg.solve on the gamma-weighted matrix).
        raise NotImplementedError(
            "Quadratic prox is not yet supported on the Simulator backend"
        )

    _PROX_DECOMPOSERS: ClassVar[dict] = {
        **StepDecomposer._PROX_DECOMPOSERS,
        "quadratic": prox_quadratic,
    }
