"""``fomtools-codegen`` command-line interface.

Generate a low-level solver for a built-in example FOM or one defined in a
Python module exposing ``build_fom() -> dict`` (with keys ``verifier`` and
``next_state``; optional ``name``, ``state``, ``params``).
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

from fomtools.codegen.config import (
    Backend,
    CodegenConfig,
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
        dtype=args.dtype,
    )


def _make_driver(args):
    if args.driver == "stub":
        from fomtools.codegen.agent.stub import StubAgentDriver

        if not args.golden:
            raise SystemExit("--driver stub requires --golden PATH (a known-good solver source)")
        return StubAgentDriver(Path(args.golden))
    if args.driver == "agent":
        from fomtools.codegen.agent.claude_sdk import ClaudeAgentDriver

        return ClaudeAgentDriver()
    return None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fomtools-codegen", description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--example", choices=sorted(EXAMPLES), help="built-in example FOM")
    src.add_argument("--module", help="path to a Python module defining build_fom()")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--driver", choices=["agent", "stub"], default="agent")
    p.add_argument("--golden", help="known-good solver source (for --driver stub)")
    p.add_argument("--version", choices=[v.value for v in Version], default="embedded")
    p.add_argument("--backend", choices=[b.value for b in Backend], default="c")
    p.add_argument("--max-iter", type=int, default=1000)
    p.add_argument("--tol", type=float, default=None, help="enable inf-norm-change termination")
    p.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    p.add_argument("--phase-b", action="store_true", help="run the performance-optimization phase")
    p.add_argument("--scaffold-only", action="store_true", help="emit files but skip the agent")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    fom = _load_module_fom(args.module) if args.module else get_example(args.example)
    config = _build_config(args)

    if args.scaffold_only:
        from fomtools.codegen.emit import scaffold
        from fomtools.codegen.extract import extract_ir

        ir = extract_ir(
            fom["verifier"],
            fom["next_state"],
            state=fom.get("state"),
            params=fom.get("params"),
            config=config,
            name=fom.get("name", "fom"),
        )
        scaffold(ir, args.out, config)
        print(f"scaffolded {fom.get('name', 'fom')} into {args.out}")
        return 0

    driver = _make_driver(args)
    result = generate_code(
        fom["verifier"],
        fom["next_state"],
        args.out,
        config=config,
        driver=driver,
        state=fom.get("state"),
        params=fom.get("params"),
        name=fom.get("name", "fom"),
        run_phase_b=args.phase_b,
    )
    status = "OK" if result.success else "FAILED"
    print(f"codegen {status}: {result.out_dir}")
    if not result.success:
        sys.stderr.write(result.translate.log[-2000:] + "\n")
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
