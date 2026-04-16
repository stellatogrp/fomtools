from fomtools.stepdecomposers.step_decomposer import StepDecomposer


class SimulatorStepDecomposer(StepDecomposer):
    """
    Simulator Step Decomposer.

    Currently forwards all steps as primitive operations, as they are handled
    directly by the Simulator canonicalizer. This may evolve if future proxable
    steps require explicit decomposition.
    """

    pass  # All steps handled as primitives via _process_primitive
