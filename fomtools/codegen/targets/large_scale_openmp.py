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

from fomtools.codegen.ir import FomIR
from fomtools.codegen.targets.base import objective_violations
from fomtools.codegen.targets.embedded_c import _KERNEL_CHECK_RECIPE, EmbeddedCTarget


class OpenMpTarget(EmbeddedCTarget):
    name = "large_scale_openmp"

    #: Emit OpenMP pragmas on the data-parallel loops (matrix-vector products and
    #: the CG kernels). Guarded by ``#ifdef _OPENMP``, so the default serial build
    #: is unaffected and threading is enabled only with ``OMPFLAGS=-fopenmp``.
    parallel = True

    def static_checks(self, source: str, ir: FomIR | None = None) -> list[str]:
        # Large-scale targets are not bound by the embedded no-malloc /
        # division-free invariants; the objective-termination contract holds.
        return objective_violations(source, ir)

    def _makefile(self, ir: FomIR | None = None) -> str:
        guard, guard_rules = self._objective_guard(ir)
        return "\n".join(
            [
                "CC ?= cc",
                "CFLAGS ?= -O3 -Wall -Iinclude",
                "# Link-time optimization lets the compiler inline fom_apply_* (fom_ops.c)",
                "# into the solver; override with LTOFLAGS= to build without it.",
                "LTOFLAGS ?= -flto",
                "# Serial by default for portable verification; real builds use -fopenmp.",
                "OMPFLAGS ?=",
                "BENCHFLAGS ?= -O3 -Iinclude",
                "# The sources a rewrite may change, rebuilt (with LTO) on every build; the",
                "# matrix data and the verify samples are large and change only with the",
                "# problem (scaffold leaves an unchanged file untouched), so they are objects",
                "# built once, without LTO: there is no code in them to inline.",
                "CODE = src/fom_solver.c src/fom_linsolve.c src/fom_ops.c src/fom_objective.c",
                "DATA = build/fom_data.o",
                "SRC = $(CODE) src/fom_data.c",
                "# Package-written reference matvecs for verify's direct check of fom_ops.c.",
                "VERIFY_SRC = $(wildcard test/ops_ref.c)",
                "",
                "UNAME_S := $(shell uname -s)",
                "ifeq ($(UNAME_S),Darwin)",
                "LIBEXT = dylib",
                "else",
                "LIBEXT = so",
                "endif",
                "",
                ".PHONY: verify bench profile lib release clean objective_guard",
                "",
                "build/fom_data.o: src/fom_data.c include/fom_data.h include/fom_workspace.h",
                "\t@mkdir -p build",
                "\t$(CC) -O2 -fPIC -Iinclude -c src/fom_data.c -o $@",
                "VERIFY_DEPS = test/verify_main.c $(wildcard test/*.h) $(wildcard include/*.h)",
                "build/verify_main.o: $(VERIFY_DEPS)",
                "\t@mkdir -p build",
                "\t$(CC) $(CFLAGS) $(OMPFLAGS) -Itest -c test/verify_main.c -o $@",
                "",
                f"verify:{guard} $(CODE) $(VERIFY_SRC) $(DATA) build/verify_main.o",
                _KERNEL_CHECK_RECIPE,
                "\t$(CC) $(CFLAGS) $(LTOFLAGS) $(OMPFLAGS) -Itest $(CODE) $(VERIFY_SRC) $(DATA) "
                "build/verify_main.o -o verify_bin -lm",
                "\t./verify_bin",
                "",
                "release: release/fom_solver.c $(DATA) release/fom_linsolve.c "
                "release/fom_ops.c src/fom_objective.c",
                "\t$(CC) $(CFLAGS) $(LTOFLAGS) $(OMPFLAGS) -shared -fPIC release/fom_solver.c $(DATA) "
                "release/fom_linsolve.c release/fom_ops.c src/fom_objective.c "
                "-o libfomsolver.$(LIBEXT) -lm",
                "",
                "bench: $(CODE) $(DATA) test/bench_main.c",
                "\t$(CC) $(BENCHFLAGS) $(LTOFLAGS) $(OMPFLAGS) -Itest $(CODE) $(DATA) test/bench_main.c "
                "-o bench_bin -lm",
                "\t./bench_bin",
                "",
                "profile: $(CODE) $(DATA) src/fom_profile.c test/bench_main.c",
                "\t$(CC) $(BENCHFLAGS) $(LTOFLAGS) $(OMPFLAGS) -DFOM_PROFILE -Itest $(CODE) $(DATA) "
                "src/fom_profile.c test/bench_main.c -o profile_bin -lm",
                "\t./profile_bin",
                "",
                "lib: $(CODE) $(DATA)",
                "\t$(CC) $(CFLAGS) $(LTOFLAGS) $(OMPFLAGS) -shared -fPIC $(CODE) $(DATA) "
                "-o libfomsolver.$(LIBEXT) -lm",
                "",
                "clean:",
                "\trm -rf verify_bin bench_bin profile_bin libfomsolver.so libfomsolver.dylib build",
                "",
                *guard_rules,
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
                "The matrix-vector products (`fom_ops.c`) and the conjugate-gradient",
                "solve (`fom_linsolve.c`, when `solve_method=cg`) carry OpenMP pragmas on",
                "their data-parallel loops, guarded by `#ifdef _OPENMP`: the default",
                "serial build ignores them, and `OMPFLAGS=-fopenmp` turns threading on.",
                "The triangular solves of the direct (LDL^T / LU) factorizations are",
                "inherently sequential and run single-threaded.",
                "",
                "`src/fom_solver.c` implements the iteration (the agent may add further",
                "OpenMP pragmas); all numeric data lives in `src/fom_data.c`. See",
                "`SPEC.md`.",
                "",
            ]
        )
