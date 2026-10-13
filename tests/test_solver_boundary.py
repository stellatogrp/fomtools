"""
Information flows one way: the canonicalizer stage builds a solver-agnostic model
and only ``fomtools/solvers`` knows about a specific backend.

Scans every module outside ``fomtools/solvers`` that takes part in building the
model and fails on backend imports, backend parameter or attribute names, and
branching on solver names. Solver names may still appear as opaque values (for
example a default argument) that are handed to ``fomtools.solvers.make_solver``.
"""

import ast
from pathlib import Path

import pytest

import fomtools

ROOT = Path(fomtools.__file__).parent
PACKAGES = ["canonicalizers", "stepdecomposers", "steps", "sets", "proxable_functions"]
MODULES = ["verifier.py", "linearexpression.py", "constraint.py"]

BACKEND_IMPORTS = {"gurobipy", "highspy", "fomtools.solvers.gurobi", "fomtools.solvers.highs"}
BACKEND_NAMES = {
    "OutputFlag",
    "output_flag",
    "MIPFocus",
    "BranchPriority",
    "ObjBound",
    "NonConvex",
    "TimeLimit",
    "MIPGap",
}
SOLVER_NAMES = {"gurobi", "grb", "highs"}


def _files():
    files = [ROOT / m for m in MODULES]
    for pkg in PACKAGES:
        files += sorted((ROOT / pkg).rglob("*.py"))
    return files


def _docstring_nodes(tree):
    """The string constants that are docstrings, which may mention backends freely."""
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                ids.add(id(body[0].value))
    return ids


def _violations(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    docstrings = _docstring_nodes(tree)
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [f"import {a.name}" for a in node.names if a.name in BACKEND_IMPORTS]
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod in BACKEND_IMPORTS:
                found.append(f"from {mod} import ...")
            if mod == "fomtools.solvers":
                found += [f"from {mod} import {a.name}" for a in node.names if a.name in ("gurobi", "highs")]
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in docstrings and node.value in BACKEND_NAMES:
                found.append(f"backend name {node.value!r} (line {node.lineno})")
        elif isinstance(node, ast.Compare):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str) and sub.value.lower() in SOLVER_NAMES:
                    found.append(f"comparison against solver name {sub.value!r} (line {node.lineno})")
        elif isinstance(node, ast.MatchValue):
            if isinstance(node.value, ast.Constant) and str(node.value.value).lower() in SOLVER_NAMES:
                found.append(f"match on solver name {node.value.value!r} (line {node.lineno})")
    return found


@pytest.mark.parametrize("path", _files(), ids=lambda p: str(p.relative_to(ROOT)))
def test_no_backend_knowledge_outside_solvers(path):
    assert _violations(path) == []


def test_scanner_catches_a_leak(tmp_path):
    leak = tmp_path / "leak.py"
    leak.write_text(
        "import gurobipy as gp\n"
        "def f(params, solver):\n"
        "    if solver in ('gurobi', 'grb'):\n"
        "        return params['OutputFlag']\n"
    )
    found = _violations(leak)
    assert any("gurobipy" in v for v in found)
    assert any("OutputFlag" in v for v in found)
    assert any("solver name" in v for v in found)
