"""Code-generation targets (embedded C, large-scale CUDA/OpenMP)."""

from fomtools.codegen.config import Backend, CodegenConfig, Version
from fomtools.codegen.targets.base import TargetDescriptor
from fomtools.codegen.targets.embedded_c import EmbeddedCTarget


def get_target(config: CodegenConfig) -> TargetDescriptor:
    """Return the target descriptor for a configuration."""
    if config.version == Version.EMBEDDED and config.backend == Backend.C:
        return EmbeddedCTarget(config)
    if config.version == Version.LARGE_SCALE:
        if config.backend == Backend.CUDA:
            from fomtools.codegen.targets.large_scale_cuda import CudaTarget

            return CudaTarget(config)
        if config.backend == Backend.OPENMP:
            from fomtools.codegen.targets.large_scale_openmp import OpenMpTarget

            return OpenMpTarget(config)
    raise ValueError(
        f"No target for version={config.version.value!r} backend={config.backend.value!r}"
    )


__all__ = ["TargetDescriptor", "EmbeddedCTarget", "get_target"]
