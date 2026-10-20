"""Rules that keep agent-rewritten kernels tied to the package's checked data.

``make verify`` checks every ``fom_apply_<name>`` directly against its matrix,
but only the code inside ``fom_apply_<name>`` goes through that check. These
rules keep the coefficients there, where it can see them:

* **No new floating-point constants.** A rewritten solver or ``fom_ops.c`` may
  not add floating-point literals (decimal or hexadecimal) or integer literals
  cast to a floating type, other than the structural constants 0, 0.5, 1 and 2.
  This is a syntactic rule, not a proof that no coefficient is hardcoded: an
  integer literal used in arithmetic (``3 * x[j]``) or a constant computed from
  allowed ones (``1.0 + 1.0 + 1.0``) gets past it. The numerical checks are the
  defence there: the exact unit-vector probes of each ``fom_apply_<name>``, and
  the sampled end-to-end comparison for the solver.
* **Products through the checked routines.** A rewritten solver may not read the
  ``FOM_MAT_<name>`` data of a matrix it multiplies by (it calls
  ``fom_apply_<name>`` instead), and may not newly call ``fom_apply_*`` from
  ``fom_setup`` / ``fom_update_params``, or from any function of the file they
  reach, which is how a matrix would be rebuilt by probing it on unit vectors.
  In ``fom_ops.c``, a routine is ``fom_apply_<name>`` together with every
  function it reaches (a helper, or the CUDA target's ``__global__`` kernel); a
  routine may read only its own matrix's data, no code outside every routine (a
  macro, static data, an unreached function) may read matrix data at all, and
  every function but the ``fom_apply_<name>`` entry points must be ``static``,
  so nothing reaches a routine's helpers except through it.
* **Checkable structure.** Each function these rules inspect (``fom_setup`` and
  ``fom_update_params`` in the solver, every ``fom_apply_<name>`` in
  ``fom_ops.c``) must be found as a single plain C definition with balanced
  braces; a function the rules cannot find, or one defined twice (in ``#if``
  branches, say), is a violation, never a silently skipped rule.

The translated solver is held to the matrix-data rule too (``make verify`` applies
it during translation), so code optimization starts from a solver that already
multiplies only through the checked routines.

Literals the translated version of the same file already had are exempt: that
version passed verify under a prompt that forbids hardcoding, so only what a
rewrite adds is new. In ``fom_ops.c`` the exemption is per routine: a literal in
a routine is exempt only if the translated routine had it, and one outside every
routine only if the translated file had it outside too.

Not covered by a rule: state a rewritten ``fom_ops.c`` keeps across calls (a
``static`` cache, say). The direct check runs each routine before ``fom_setup``
and repeatedly on many probes, so a cache filled at setup, or a result that
depends on earlier calls, fails it; scratch buffers stay allowed because the
embedded target has no other way to get temporary storage.

Standard library only: the scaffold copies this file into the generated
directory, where ``make verify`` runs it (``python3 test/kernel_check.py``) so the
agent sees a violation as it works. The pipeline applies the same rules when it
collects the rewrite, and that is the check that decides.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_C_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_C_LINE_COMMENT = re.compile(r"//[^\n]*")
_C_STRING = re.compile(r'"(?:\\.|[^"\\])*"')
_C_CHAR = re.compile(r"'(?:\\.|[^'\\])*'")

#: A decimal floating-point literal (decimal point and/or exponent, optional
#: suffix), not part of an identifier or of a longer number.
_C_FLOAT_LITERAL = re.compile(
    r"(?<![\w.])((?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?|\d+[eE][-+]?\d+)[fFlL]?(?![\w.])"
)

#: A hexadecimal floating-point literal (``0x1.8p0``): the binary exponent is
#: mandatory, which tells it apart from a hex integer.
_C_HEX_FLOAT_LITERAL = re.compile(
    r"(?<![\w.])(0[xX](?:[0-9a-fA-F]+\.?[0-9a-fA-F]*|\.[0-9a-fA-F]+)[pP][-+]?\d+)[fFlL]?(?![\w.])"
)

#: An integer literal cast to a floating type, ``(fom_float)3``: a coefficient
#: written without a decimal point.
_C_INT_CAST_LITERAL = re.compile(
    r"\(\s*(?:fom_float|double|float)\s*\)\s*\(?\s*(\d+)[uUlL]*\b(?!\s*[.\w])"
)

#: Float literals a rewritten kernel may contain: structural constants, not
#: coefficients (zeroing, unit steps, doubling and halving).
ALLOWED_FLOAT_LITERALS = frozenset({0.0, 0.5, 1.0, 2.0})

#: Functions that run once per setup or parameter update, not per iteration.
SETUP_FUNCTIONS = ("fom_setup", "fom_update_params")

#: A use of ``fom_apply_<name>``: a call, or the function's address.
_APPLY_REF = re.compile(r"\bfom_apply_(\w+)\b")

_FUNCTION_DEF = re.compile(r"\b([A-Za-z_]\w*)\s*\([^;{)]*\)\s*\{")
_C_KEYWORDS = frozenset({"if", "for", "while", "switch", "return", "sizeof"})


def strip_c_comments_and_strings(source: str) -> str:
    """Remove C comments and string literals so token checks don't false-match.

    Comments/strings are replaced with spaces (preserving rough offsets).
    """
    source = _C_BLOCK_COMMENT.sub(" ", source)
    source = _C_LINE_COMMENT.sub(" ", source)
    source = _C_STRING.sub(" ", source)
    source = _C_CHAR.sub(" ", source)
    return source


def float_literals(code: str) -> list[str]:
    """Floating-point constants in (stripped) ``code`` that are not structural:
    decimal and hexadecimal float literals, and integer literals cast to a
    floating type."""
    found = [(m.group(0), float(m.group(1))) for m in _C_FLOAT_LITERAL.finditer(code)] + [
        (m.group(0), float.fromhex(m.group(1))) for m in _C_HEX_FLOAT_LITERAL.finditer(code)
    ]
    found += [(m.group(0), float(m.group(1))) for m in _C_INT_CAST_LITERAL.finditer(code)]
    return list(dict.fromkeys(text for text, value in found if value not in ALLOWED_FLOAT_LITERALS))


def _definitions(code: str, name: str) -> list[re.Match]:
    """Each place (stripped) ``code`` defines the C function ``name``."""
    return list(re.finditer(rf"\b{re.escape(name)}\s*\([^;{{)]*\)\s*\{{", code))


def _function_span(code: str, name: str) -> tuple[int, int] | None:
    """``(start, end)`` of the body of the C function ``name`` in (stripped)
    ``code``, or ``None`` if no definition is found or its braces never balance
    (e.g. ``#if`` branches that each open a brace)."""
    match = next(iter(_definitions(code, name)), None)
    if match is None:
        return None
    depth, i = 1, match.end()
    while depth and i < len(code):
        depth += {"{": 1, "}": -1}.get(code[i], 0)
        i += 1
    if depth:
        return None
    return match.end(), i - 1


def function_body(code: str, name: str) -> str | None:
    """The body of the C function ``name`` defined in (stripped) ``code``, or
    ``None`` if it cannot be found (see `_function_span`)."""
    span = _function_span(code, name)
    return None if span is None else code[span[0] : span[1]]


def _not_found(name: str) -> str:
    return (
        f"could not find a definition of {name} with balanced braces, so the kernel "
        "rules cannot check it; define it as a plain C function"
    )


def _iter_functions(code: str):
    """``(name, prefix, body)`` of each C function defined at file scope in
    (stripped) ``code``, in order; ``prefix`` is its declaration specifiers
    (``static``, return type), and ``body`` the ``(start, end)`` of its body.
    Stops at a definition whose braces never balance."""
    end = 0
    for match in _FUNCTION_DEF.finditer(code):
        if match.start() < end or match.group(1) in _C_KEYWORDS:
            continue
        depth, i = 1, match.end()
        while depth and i < len(code):
            depth += {"{": 1, "}": -1}.get(code[i], 0)
            i += 1
        if depth:
            return
        head = code[
            max(
                code.rfind(";", end, match.start()), code.rfind("}", end, match.start()), end
            ) : match.start()
        ]
        prefix = " ".join(ln for ln in head.splitlines() if not ln.lstrip().startswith("#"))
        yield match.group(1), prefix, (match.end(), i - 1)
        end = i


def functions(code: str) -> tuple[dict[str, list[str]], str]:
    """The C functions defined at file scope in (stripped) ``code``: their bodies
    by name (a list: ``#if`` branches may define one twice), and the file scope
    left over (declarations, macros, static data) with the bodies blanked out."""
    bodies: dict[str, list[str]] = {}
    scope = code
    for name, _, (start, stop) in _iter_functions(code):
        bodies.setdefault(name, []).append(code[start:stop])
        scope = scope[:start] + " " * (stop - start) + scope[stop:]
    return bodies, scope


def reachable(bodies: dict[str, list[str]], roots) -> set[str]:
    """The functions of ``bodies`` that ``roots`` use, directly or through each
    other (a call or the function's address), ``roots`` included."""
    seen, todo = set(roots), list(roots)
    while todo:
        for body in bodies.get(todo.pop(), []):
            for fn in set(re.findall(r"\b\w+\b", body)) & bodies.keys() - seen:
                seen.add(fn)
                todo.append(fn)
    return seen


def setup_apply_calls(code: str) -> set[str] | None:
    """Matrices whose ``fom_apply_<name>`` the setup functions use, directly or
    through the functions of the file they reach, or ``None`` if a setup function
    cannot be found."""
    if any(function_body(code, fn) is None for fn in SETUP_FUNCTIONS):
        return None
    bodies, _ = functions(code)
    return {
        name
        for fn in reachable(bodies, SETUP_FUNCTIONS)
        for body in bodies.get(fn, [])
        for name in _APPLY_REF.findall(body)
    }


def _mat_data(name: str) -> re.Pattern:
    """A reference to the data arrays of matrix ``name`` (not its dimensions): the
    host ``FOM_MAT_<name>`` arrays, or the CUDA target's device copies
    ``FOM_OP_<NAME>_*``."""
    return re.compile(
        rf"\bFOM_MAT_{re.escape(name)}(?:_(?:ROWPTR|COLIDX|VALS))?\b"
        rf"|\bFOM_OP_{re.escape(name.upper())}_(?:ROWPTR|COLIDX|VALS|DENSE)\b"
    )


def _new_literals(code: str, known: set[str]) -> list[str]:
    return [
        f"floating-point literal '{lit}' looks like a hardcoded coefficient; "
        "read coefficients from fom_data.c"
        for lit in float_literals(code)
        if lit not in known
    ]


def _routine_regions(code: str, names) -> tuple[dict[str, set[str]], dict, str, list[str]]:
    """``fom_ops.c`` split by routine: the functions each ``fom_apply_<name>``
    reaches (itself included), the function bodies, the code outside every
    region (file scope and functions no routine reaches), and violations of
    its structure."""
    bodies, scope = functions(code)
    regions, violations = {}, []
    for name in names:
        fn = f"fom_apply_{name}"
        if fn not in bodies:
            violations.append(_not_found(fn))
        elif len(bodies[fn]) > 1:
            violations.append(
                f"{fn} is defined more than once (e.g. in #if branches); the kernel "
                "rules check one plain definition"
            )
        else:
            regions[name] = reachable(bodies, [fn])
    covered = set().union(*regions.values())
    outside = " ".join([scope] + [b for fn, bs in bodies.items() if fn not in covered for b in bs])
    return regions, bodies, outside, violations


def _check_ops(code: str, base: str, matvec_names) -> list[str]:
    """``fom_ops.c``: each routine (with the helpers it reaches) on its own data,
    literals per routine, and no entry points but the routines."""
    regions, bodies, outside, violations = _routine_regions(code, matvec_names)
    base_regions, base_bodies, base_outside, _ = _routine_regions(base, matvec_names)
    entry = {f"fom_apply_{name}" for name in matvec_names}
    for fn, prefix, _ in _iter_functions(code):
        if fn not in entry and not re.search(r"\bstatic\b", prefix):
            violations.append(
                f"{fn} is not static; every function of fom_ops.c but the "
                "fom_apply_<name> routines must be, so it is reached only through them"
            )
    for name, region in regions.items():
        fn = f"fom_apply_{name}"
        text = " ".join(b for g in sorted(region) for b in bodies[g])
        base_text = " ".join(b for g in base_regions.get(name, ()) for b in base_bodies[g])
        violations += [f"{fn}: {v}" for v in _new_literals(text, set(float_literals(base_text)))]
        for other in matvec_names:
            if other != name and _mat_data(other).search(text):
                violations.append(
                    f"{fn} reads FOM_MAT_{other}, the data of another product; each "
                    "routine (and every helper it calls) may read only its own matrix"
                )
    for name in matvec_names:
        if _mat_data(name).search(outside):
            violations.append(
                f"FOM_MAT_{name} is read outside fom_apply_{name} and the helpers it "
                "calls, where the rules cannot see which routine uses it"
            )
    return _new_literals(outside, set(float_literals(base_outside))) + violations


def _solver_data_reads(code: str, matvec_names) -> list[str]:
    return [
        f"the solver reads FOM_MAT_{name} directly; apply it with "
        f"fom_apply_{name}, which verify checks against the matrix"
        for name in matvec_names
        if _mat_data(name).search(code)
    ]


def check(kind: str, source: str, matvec_names, baseline: str | None = None) -> list[str]:
    """Rule violations in a rewritten ``kind`` (``"solver"`` or ``"ops"``) source.

    ``baseline`` is the translated version of the same file; whatever it already
    contains is exempt. Kind ``"translation"`` is the translated solver itself:
    it has no baseline, so only the rule without one applies (it reads no
    matrix data, so every product goes through a checked ``fom_apply_<name>``).
    """
    code = strip_c_comments_and_strings(source)
    base = strip_c_comments_and_strings(baseline) if baseline is not None else ""
    if kind == "ops":
        return _check_ops(code, base, matvec_names)
    if kind == "translation":
        return _solver_data_reads(code, matvec_names)
    violations = _new_literals(code, set(float_literals(base)))
    if kind != "solver":
        return violations
    violations += _solver_data_reads(code, matvec_names)
    calls = setup_apply_calls(code)
    if calls is None:
        missing = [fn for fn in SETUP_FUNCTIONS if function_body(code, fn) is None]
        return violations + [_not_found(fn) for fn in missing]
    for name in sorted(calls - (setup_apply_calls(base) or set())):
        violations.append(
            f"fom_apply_{name} is called from {' / '.join(SETUP_FUNCTIONS)}; apply "
            "matrices in the iteration, not to rebuild them at setup"
        )
    return violations


def main(config_path: str) -> int:
    """Check the files named in a generated ``kernel_check.json``; 0 if all pass."""
    config = json.loads(Path(config_path).read_text())
    root = Path(config_path).resolve().parent.parent
    failed = False
    for entry in config["files"]:
        source = (root / entry["path"]).read_text()
        for v in check(entry["kind"], source, config["matvec"], entry.get("baseline")):
            print(f"VERIFY FAILED: kernel rule: {entry['path']}: {v}")
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "test/kernel_check.json"))
