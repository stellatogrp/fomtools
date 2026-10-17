"""Rules that keep agent-rewritten kernels tied to the package's checked data.

``make verify`` checks every ``fom_apply_<name>`` directly against its matrix,
but only the code inside ``fom_apply_<name>`` goes through that check. These
rules keep the coefficients there, where it can see them:

* **No hardcoded coefficients.** A rewritten solver or ``fom_ops.c`` may not add
  floating-point literals other than structural constants (0, 0.5, 1, 2).
* **Products through the checked routines.** A rewritten solver may not read the
  ``FOM_MAT_<name>`` data of a matrix it multiplies by (it calls
  ``fom_apply_<name>`` instead), and may not newly call ``fom_apply_*`` from
  ``fom_setup`` / ``fom_update_params``, which is how a matrix would be rebuilt by
  probing it on unit vectors.

Each rule exempts what the translated version of the same file already did:
that version passed verify under a prompt that forbids hardcoding, so only what
a rewrite adds is new.

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

#: A floating-point literal (decimal point and/or exponent, optional suffix), not
#: part of an identifier or of a longer number.
_C_FLOAT_LITERAL = re.compile(
    r"(?<![\w.])((?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?|\d+[eE][-+]?\d+)[fFlL]?(?![\w.])"
)

#: Float literals a rewritten kernel may contain: structural constants, not
#: coefficients (zeroing, unit steps, doubling and halving).
ALLOWED_FLOAT_LITERALS = frozenset({0.0, 0.5, 1.0, 2.0})

#: Functions that run once per setup or parameter update, not per iteration.
SETUP_FUNCTIONS = ("fom_setup", "fom_update_params")

_APPLY_CALL = re.compile(r"\bfom_apply_(\w+)\s*\(")


def strip_c_comments_and_strings(source: str) -> str:
    """Remove C comments and string literals so token checks don't false-match.

    Comments/strings are replaced with spaces (preserving rough offsets).
    """
    source = _C_BLOCK_COMMENT.sub(" ", source)
    source = _C_LINE_COMMENT.sub(" ", source)
    source = _C_STRING.sub(" ", source)
    return source


def float_literals(code: str) -> list[str]:
    """Float literals in (stripped) ``code`` that are not structural constants."""
    return list(
        dict.fromkeys(
            m.group(0)
            for m in _C_FLOAT_LITERAL.finditer(code)
            if float(m.group(1)) not in ALLOWED_FLOAT_LITERALS
        )
    )


def function_body(code: str, name: str) -> str:
    """The body of the C function ``name`` defined in (stripped) ``code``, or ``""``."""
    match = re.search(rf"\b{re.escape(name)}\s*\([^;{{)]*\)\s*\{{", code)
    if match is None:
        return ""
    depth, i = 1, match.end()
    while depth and i < len(code):
        depth += {"{": 1, "}": -1}.get(code[i], 0)
        i += 1
    return code[match.end() : i - 1]


def setup_apply_calls(code: str) -> set[str]:
    """Matrices whose ``fom_apply_<name>`` the setup functions call."""
    return {name for fn in SETUP_FUNCTIONS for name in _APPLY_CALL.findall(function_body(code, fn))}


def check(kind: str, source: str, matvec_names, baseline: str | None = None) -> list[str]:
    """Rule violations in a rewritten ``kind`` (``"solver"`` or ``"ops"``) source.

    ``baseline`` is the translated version of the same file; whatever it already
    contains is exempt.
    """
    code = strip_c_comments_and_strings(source)
    base = strip_c_comments_and_strings(baseline) if baseline is not None else ""
    known = set(float_literals(base))
    violations = [
        f"floating-point literal '{lit}' looks like a hardcoded coefficient; "
        "read coefficients from fom_data.c"
        for lit in float_literals(code)
        if lit not in known
    ]
    if kind != "solver":
        return violations
    for name in matvec_names:
        if re.search(rf"\bFOM_MAT_{re.escape(name)}(?:_(?:ROWPTR|COLIDX|VALS))?\b", code):
            violations.append(
                f"the solver reads FOM_MAT_{name} directly; apply it with "
                f"fom_apply_{name}, which verify checks against the matrix"
            )
    for name in sorted(setup_apply_calls(code) - setup_apply_calls(base)):
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
