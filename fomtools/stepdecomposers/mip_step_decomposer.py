from fomtools.stepdecomposers.step_decomposer import StepDecomposer


class MIPStepDecomposer(StepDecomposer):
    """
    MIP step decomposer.

    Currently passes all steps through as primitives since the MIP canonicalizer
    handles them directly. Future proxable steps may require decomposition.

    The MIP canonicalizer natively supports:
    - ReLU via big-M constraints
    - SoftThreshold via convex hull formulation
    - SaturatedLinear via convex hull formulation
    """

    pass  # All steps handled as primitives via _process_primitive
