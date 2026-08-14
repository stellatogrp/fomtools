"""Drive a generated C solver from Python: build it, time it, fly it.

The generated directory ships a ctypes wrapper (``python/fom_solver.py``) whose
structures mirror the emitted C ABI exactly.  This reuses those structures and
the library it loads, but not its ``solve``: that method copies parameters in and
the solution out one ``ctypes`` element at a time, several hundred microseconds
per call on a problem this size, which would swamp the solve it is wrapping.
Here the parameter block is filled once and refreshed with :func:`ctypes.memmove`,
and the solution is read through :func:`numpy.ctypeslib.as_array`.

Two things make one generated solver enough for the whole experiment.  The
emitted loop reads ``work->settings.max_iter`` at run time rather than the
``FOM_MAX_ITER`` it was generated with, so the iteration count is a setting; and
``fom_update_params`` calls ``fom_reset_state``, so every solve cold starts,
exactly as the simulated controller does.

The library is built at the same optimization level as the emitted benchmark, so
the binary that is timed is the binary that flies.
"""

from __future__ import annotations

import ctypes
import importlib.util
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np

from common.simulate import IterationCost, slope_us

#: Compiler flags for ``make lib``.  The generated Makefile builds the library at
#: ``-O2`` and the benchmark at ``-O3``; timing one and flying the other would
#: report a solver that is not the one under test.
LIB_CFLAGS = "-O3 -Wall -Iinclude"


class CSolver:
    """A generated embedded-C solver, loaded and callable."""

    def __init__(self, directory: str | Path, cflags: str = LIB_CFLAGS, build: bool = True):
        self.directory = Path(directory)
        if build:
            self._build(cflags)
        module = self._load_wrapper()
        self.state_dim = module.STATE_DIM
        self.param_names = list(module.PARAM_NAMES)
        self.param_dims = dict(module.PARAM_DIMS)

        self._lib = ctypes.CDLL(str(self._library_path()))
        self._lib.fom_set_default_settings.argtypes = [ctypes.POINTER(module.FomSettings)]
        self._lib.fom_setup.argtypes = [
            ctypes.POINTER(module.FomWorkspace),
            ctypes.POINTER(module.FomSettings),
        ]
        self._lib.fom_update_params.argtypes = [
            ctypes.POINTER(module.FomWorkspace),
            ctypes.POINTER(module.FomParams),
        ]
        self._lib.fom_solve.argtypes = [ctypes.POINTER(module.FomWorkspace)]
        self._lib.fom_solve.restype = ctypes.c_int

        self._work = module.FomWorkspace()
        self._params = module.FomParams()
        settings = module.FomSettings()
        self._lib.fom_set_default_settings(ctypes.byref(settings))
        self._lib.fom_setup(ctypes.byref(self._work), ctypes.byref(settings))
        self._work_ref = ctypes.byref(self._work)
        self._params_ref = ctypes.byref(self._params)
        self.default_max_iter = int(self._work.settings.max_iter)

    # -- setup ------------------------------------------------------------

    def _build(self, cflags: str) -> None:
        proc = subprocess.run(
            ["make", "lib", f"CFLAGS={cflags}"],
            cwd=str(self.directory),
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"could not build {self.directory}:\n{proc.stdout}\n{proc.stderr}")

    def _library_path(self) -> Path:
        for name in ("libfomsolver.dylib", "libfomsolver.so"):
            candidate = self.directory / name
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"no shared library in {self.directory}; run 'make lib' there")

    def _load_wrapper(self):
        """Import the emitted ``python/fom_solver.py`` by path.

        Loaded as a uniquely named module so two generated solvers can be open at
        once, which is the whole point of the experiment.
        """
        path = self.directory / "python" / "fom_solver.py"
        name = f"fom_solver_{abs(hash(str(self.directory))):x}"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"could not load the generated wrapper at {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    # -- use --------------------------------------------------------------

    @property
    def max_iter(self) -> int:
        return int(self._work.settings.max_iter)

    @max_iter.setter
    def max_iter(self, iterations: int) -> None:
        self._work.settings.max_iter = int(iterations)

    def set_params(self, **values: np.ndarray) -> None:
        """Copy parameter vectors into the workspace and cold start the solver."""
        for name in self.param_names:
            if name not in values:
                raise ValueError(f"missing parameter {name!r}")
            vec = np.ascontiguousarray(values[name], dtype=np.float64).reshape(-1)
            if vec.size != self.param_dims[name]:
                raise ValueError(
                    f"parameter {name!r} expects {self.param_dims[name]} entries, got {vec.size}"
                )
            field = getattr(self._params, name)
            ctypes.memmove(field, vec.ctypes.data, vec.nbytes)
        self._lib.fom_update_params(self._work_ref, self._params_ref)

    def run(self) -> None:
        """Run the solve loop from the current (cold) state."""
        self._lib.fom_solve(self._work_ref)

    def solution(self) -> np.ndarray:
        """The state the last solve produced."""
        return np.ctypeslib.as_array(self._work.solution.x).astype(float, copy=True)

    def residual(self) -> float:
        """Whatever the last solve reported as its residual.

        Meaningful only for a solver generated with an early-exit criterion;
        under a fixed iteration count the emitted loop is free to leave this
        alone, so it is reported rather than relied on.
        """
        return float(self._work.solution.residual)

    def iterations_taken(self) -> int:
        """Iterations the last solve actually ran."""
        return int(self._work.solution.iter)

    def solve(self, iterations: int | None = None, **values: np.ndarray) -> np.ndarray:
        """Set the parameters, run ``iterations`` steps, and return the state."""
        if iterations is not None:
            self.max_iter = iterations
        self.set_params(**values)
        self.run()
        return self.solution()

    # -- timing -----------------------------------------------------------

    def us_per_iteration(
        self,
        params: dict[str, np.ndarray],
        low: int = 20,
        high: int = 200,
        repeats: int = 7,
        solves: int = 20,
        trials: int = 5,
    ) -> IterationCost:
        """Microseconds per ADMM iteration, as a two-point slope in the loop count.

        Timing one iteration count would charge the loop for everything that
        happens once per solve: the parameter copy, ``fom_reset_state``, and the
        two ``ctypes`` calls that get us there.  Those costs do not depend on the
        iteration count, so differencing two counts removes them and leaves the
        loop alone, which is the quantity a compute budget buys.

        The same measurement as :func:`common.simulate.us_per_iteration`, so the
        NumPy controller and the generated solvers are timed alike.
        """

        def measure(iterations: int) -> float:
            self.max_iter = iterations
            samples = []
            for _ in range(repeats):
                start = time.perf_counter()
                for _ in range(solves):
                    self.set_params(**params)
                    self.run()
                samples.append((time.perf_counter() - start) * 1e6 / solves)
            return statistics.median(samples)

        return slope_us(measure, low, high, trials)

    def us_per_solve(
        self, params: dict[str, np.ndarray], iterations: int, repeats: int = 7
    ) -> float:
        """Wall-clock microseconds for one complete solve, reset included."""
        self.max_iter = iterations
        self.set_params(**params)
        self.run()
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            self.set_params(**params)
            self.run()
            samples.append((time.perf_counter() - start) * 1e6)
        return statistics.median(samples)
