"""Large-scale multithreaded-CPU (OpenMP) target.

Reuses the embedded C target's ABI header and numeric data emission, but:

* relaxes the embedded invariants (dynamic allocation and division are allowed —
  large problems may need scratch buffers and the agent may keep an explicit
  factorization with divisions),
* builds an OpenMP-annotated solver.

The default ``make verify`` builds *serially* (``OMPFLAGS`` empty) so the
generated solver can be numerically checked against the reference oracle on any
machine with a C compiler; pass ``OMPFLAGS=-fopenmp`` for the real parallel
build.
"""

from __future__ import annotations

from fomtools.codegen.targets.embedded_c import EmbeddedCTarget


class OpenMpTarget(EmbeddedCTarget):
    name = "large_scale_openmp"

    def static_checks(self, source: str) -> list[str]:
        # Large-scale targets are not bound by the embedded no-malloc /
        # division-free invariants.
        return []

    def _makefile(self) -> str:
        return "\n".join(
            [
                "CC ?= cc",
                "CFLAGS ?= -O3 -Wall -Iinclude",
                "# Serial by default for portable verification; real builds use -fopenmp.",
                "OMPFLAGS ?=",
                "SRC = src/fom_solver.c src/fom_data.c",
                "",
                "UNAME_S := $(shell uname -s)",
                "ifeq ($(UNAME_S),Darwin)",
                "LIBEXT = dylib",
                "else",
                "LIBEXT = so",
                "endif",
                "",
                ".PHONY: verify lib clean",
                "",
                "verify: $(SRC) test/verify_main.c",
                "\t$(CC) $(CFLAGS) $(OMPFLAGS) -Itest $(SRC) test/verify_main.c -o verify_bin -lm",
                "\t./verify_bin",
                "",
                "lib: $(SRC)",
                "\t$(CC) $(CFLAGS) $(OMPFLAGS) -shared -fPIC $(SRC) -o libfomsolver.$(LIBEXT) -lm",
                "",
                "clean:",
                "\trm -f verify_bin libfomsolver.so libfomsolver.dylib",
                "",
            ]
        )

    def _readme(self, ir) -> str:
        return "\n".join(
            [
                f"# Generated solver: `{ir.name}` (large-scale, OpenMP CPU)",
                "",
                "Multithreaded-CPU solver for a verified first-order method.",
                "",
                "## Build & verify",
                "```",
                "make verify                 # serial build, checked vs the reference oracle",
                "make verify OMPFLAGS=-fopenmp   # multithreaded build",
                "make lib OMPFLAGS=-fopenmp      # shared library for the Python wrapper",
                "```",
                "",
                "`src/fom_solver.c` implements the iteration with OpenMP pragmas; all",
                "numeric data lives in `src/fom_data.c`. See `test/SPEC.md`.",
                "",
            ]
        )
