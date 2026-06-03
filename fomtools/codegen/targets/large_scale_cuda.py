"""Large-scale GPU (CUDA) target.

Reuses the embedded C target's ABI header and numeric data emission, emits a
CUDA solver source (``src/fom_solver.cu``), and a ``nvcc``-based build. Because
CI typically has no GPU/``nvcc``, the verify command is a *compile check*
(``make compile``); run-verification is deferred to a GPU machine.
"""

from __future__ import annotations

from fomtools.codegen.targets.embedded_c import EmbeddedCTarget


class CudaTarget(EmbeddedCTarget):
    name = "large_scale_cuda"

    def solver_source_path(self) -> str:
        return "src/fom_solver.cu"

    def verify_command(self) -> list[str]:
        # No GPU in CI: compile-check only (build the verify binary, do not run).
        return ["make", "compile"]

    def static_checks(self, source: str) -> list[str]:
        return []

    def _makefile(self) -> str:
        return "\n".join(
            [
                "CC ?= cc",
                "NVCC ?= nvcc",
                "CFLAGS ?= -O3 -Iinclude -Itest",
                "NVCCFLAGS ?= -O3 -Iinclude -Itest",
                "",
                ".PHONY: compile verify lib clean",
                "",
                "# Compile-check: build the verify binary (no GPU run).",
                "compile: src/fom_solver.cu src/fom_data.c src/fom_linsolve.c test/verify_main.c",
                "\t$(NVCC) $(NVCCFLAGS) -c src/fom_solver.cu -o fom_solver.o",
                "\t$(CC) $(CFLAGS) -c src/fom_data.c -o fom_data.o",
                "\t$(CC) $(CFLAGS) -c src/fom_linsolve.c -o fom_linsolve.o",
                "\t$(CC) $(CFLAGS) -c test/verify_main.c -o verify_main.o",
                "\t$(NVCC) $(NVCCFLAGS) fom_solver.o fom_data.o fom_linsolve.o verify_main.o "
                "-o verify_bin",
                "",
                "# Run-verification (requires a GPU).",
                "verify: compile",
                "\t./verify_bin",
                "",
                "lib: src/fom_solver.cu src/fom_data.c",
                "\t$(NVCC) $(NVCCFLAGS) --shared -Xcompiler -fPIC src/fom_solver.cu src/fom_data.c "
                "-o libfomsolver.so",
                "",
                "clean:",
                "\trm -f verify_bin *.o libfomsolver.so",
                "",
            ]
        )

    def _readme(self, ir) -> str:
        return "\n".join(
            [
                f"# Generated solver: `{ir.name}` (large-scale, CUDA GPU)",
                "",
                "GPU solver for a verified first-order method.",
                "",
                "## Build",
                "```",
                "make compile   # compile-check (no GPU needed to build)",
                "make verify    # build and run on a GPU (checks vs the reference oracle)",
                "make lib       # shared library for the Python wrapper",
                "```",
                "",
                "`src/fom_solver.cu` implements the iteration with CUDA kernels; all",
                "numeric data lives in `src/fom_data.c`. See `test/SPEC.md`.",
                "",
            ]
        )
