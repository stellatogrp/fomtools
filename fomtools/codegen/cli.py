"""``fomtools-codegen`` command-line interface.

Generate a low-level solver for a built-in example FOM or one defined in a
Python module exposing ``build_fom() -> dict``. Three shapes of dict select the
three generation paths, all with optional ``name``, ``state`` and ``params``:

* ``verifier`` and ``iterates`` (the sequence the recorded loop produced, or an
  ``IterateTrajectory``) generates that whole run, rolling it back up when
  repeating one iteration is verified to reproduce it. Use ``--no-roll`` to
  unroll regardless.
* ``verifier`` and ``next_state`` generates a rolled solver from one stationary
  iteration.
* ``build`` and ``schedules`` generates a rolled solver whose declared
  iteration-varying coefficients are tabulated.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

from fomtools.codegen.agent import AgentBackendError, available_backends
from fomtools.codegen.config import (
    Backend,
    CodegenConfig,
    SolveMethod,
    TerminationKind,
    TerminationSpec,
    Version,
)
from fomtools.codegen.examples import EXAMPLES, get_example
from fomtools.codegen.generate import generate_code


def _load_module_fom(path: str) -> dict:
    spec = importlib.util.spec_from_file_location("_fom_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "build_fom"):
        raise ValueError(f"{path} must define build_fom() returning a dict")
    return module.build_fom()


def _build_config(args) -> CodegenConfig:
    kind = TerminationKind.INF_NORM_CHANGE if args.tol is not None else TerminationKind.MAX_ITERS
    return CodegenConfig(
        version=Version(args.version),
        backend=Backend(args.backend),
        termination=TerminationSpec(kind=kind, max_iters=args.max_iter, tol=args.tol),
        solve_method=SolveMethod(args.solve_method),
        dtype=args.dtype,
    )


def _make_driver(args):
    if args.driver == "stub":
        from fomtools.codegen.agent.stub import StubAgentDriver

        if not args.golden:
            raise SystemExit("--driver stub requires --golden PATH (a known-good solver source)")
        return StubAgentDriver(Path(args.golden))
    if args.driver == "agent":
        from fomtools.codegen.agent import make_agent_driver

        try:
            return make_agent_driver(backend=args.agent)
        except ValueError as exc:  # unknown backend name
            raise SystemExit(str(exc)) from exc
    return None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fomtools-codegen", description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--example", choices=sorted(EXAMPLES), help="built-in example FOM")
    src.add_argument("--module", help="path to a Python module defining build_fom()")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--driver", choices=["agent", "stub"], default="agent")
    p.add_argument(
        "--agent",
        # Not a hard `choices=`: user-registered CLI agents (from config) are
        # valid too. make_agent_driver validates and lists available backends.
        metavar="{" + ",".join(available_backends()) + ",...}",
        default=None,
        help="real coding-agent backend for --driver agent (default: claude, or "
        "$FOMTOOLS_CODEGEN_BACKEND); custom CLI agents can be defined in "
        "~/.config/fomtools/agents.yaml",
    )
    p.add_argument("--golden", help="known-good solver source (for --driver stub)")
    p.add_argument("--version", choices=[v.value for v in Version], default="embedded")
    p.add_argument("--backend", choices=[b.value for b in Backend], default="c")
    p.add_argument(
        "--solve-method",
        choices=[m.value for m in SolveMethod],
        default="auto",
        help="affine linear-solve strategy: auto/ldlt/lu (direct) or cg "
        "(indirect, large-scale only; requires an SPD system)",
    )
    p.add_argument("--max-iter", type=int, default=1000)
    p.add_argument("--tol", type=float, default=None, help="enable inf-norm-change termination")
    p.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    p.add_argument(
        "--code-optimization",
        dest="code_optimization",
        action="store_true",
        default=None,
        help="run the code optimization phase (default: on for --driver agent)",
    )
    p.add_argument(
        "--no-code-optimization",
        dest="code_optimization",
        action="store_false",
        help="skip the code optimization phase",
    )
    p.add_argument(
        "--no-profile", dest="profile", action="store_false", help="skip timing/profiling"
    )
    p.add_argument("--scaffold-only", action="store_true", help="emit files but skip the agent")
    p.add_argument(
        "--no-roll",
        dest="roll",
        action="store_false",
        help="for a FOM given as a recorded run, emit the unrolled body even when it "
        "could be rolled back up into a repeating iteration",
    )
    return p


def _extract_fom(fom: dict, config, roll: bool = True):
    """Extract the IR for a FOM dict: unrolled, stationary, or scheduled.

    Unpacks the dict and hands the same arguments to the same dispatch
    :func:`~fomtools.codegen.generate.generate_code` uses, so ``--scaffold-only``
    cannot preview an IR the full run would have rejected.
    """
    from fomtools.codegen.generate import extract_fom_ir

    return extract_fom_ir(
        fom["build"] if fom.get("schedules") else fom["verifier"],
        fom.get("next_state"),
        config=config,
        state=fom.get("state"),
        params=fom.get("params"),
        name=fom.get("name", "fom"),
        schedules=fom.get("schedules"),
        iterates=fom.get("iterates"),
        param_values=fom.get("param_values"),
        roll=roll,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    fom = _load_module_fom(args.module) if args.module else get_example(args.example)
    config = _build_config(args)

    if args.scaffold_only:
        from fomtools.codegen.emit import scaffold

        ir = _extract_fom(fom, config, roll=args.roll)
        scaffold(ir, args.out, config)
        print(f"scaffolded {fom.get('name', 'fom')} into {args.out}")
        return 0

    driver = _make_driver(args)
    # The code optimization phase defaults on for the real agent (the product flow) and off for the
    # stub (which cannot meaningfully optimize), unless the user is explicit.
    code_optimization = (
        args.code_optimization if args.code_optimization is not None else args.driver == "agent"
    )
    try:
        result = generate_code(
            fom["build"] if fom.get("schedules") else fom["verifier"],
            fom.get("next_state"),
            args.out,
            config=config,
            driver=driver,
            state=fom.get("state"),
            params=fom.get("params"),
            name=fom.get("name", "fom"),
            code_optimization=code_optimization,
            profile=args.profile,
            schedules=fom.get("schedules"),
            iterates=fom.get("iterates"),
            param_values=fom.get("param_values"),
            roll=args.roll,
        )
    except AgentBackendError as exc:
        # A missing / unauthenticated backend is a clear user-fixable setup
        # problem, not an internal error: report it plainly without a traceback.
        sys.stderr.write(f"codegen FAILED: {exc}\n")
        return 2
    status = "OK" if result.success else "FAILED"
    print(f"codegen {status}: {result.out_dir}")
    if result.timing is not None:
        print(result.timing.summary())
    if result.optimizations_md is not None:
        print(f"optimizations recorded in {result.optimizations_md}")
    if result.production_path is not None:
        print(
            f"marker-free production solver: {result.production_path} (build with `make release`)"
        )
    if not result.success:
        sys.stderr.write(result.translate.log[-2000:] + "\n")
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
