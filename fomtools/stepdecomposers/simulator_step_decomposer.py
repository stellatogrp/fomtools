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
