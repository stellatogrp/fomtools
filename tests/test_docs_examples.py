"""Execute the canonical documentation examples rather than maintaining second copies."""

import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _run_first_block(relpath):
    page = ROOT / relpath
    code = re.search(r"```python3?\n(.*?)```", page.read_text(), re.S).group(1)
    scope = {}
    exec(compile(code, str(page), "exec"), scope)
    return scope


def test_intro(capfd):
    scope = _run_first_block("docs/index.md")
    out = capfd.readouterr().out
    # The solvers are silent by default: only the example's own prints remain.
    assert out.splitlines()[0].startswith("worst-case residual:")
    assert len(out.splitlines()) == 2
    q = scope["q"].value
    assert q.shape == (scope["n"],) and np.all(np.abs(q) <= 1 + 1e-9)
    z, z_prev = scope["z"], scope["z_prev"]
    assert np.max(np.abs((z - z_prev).value)) >= 0


def test_quick_start(capfd):
    scope = _run_first_block("docs/quick_start.md")
    assert capfd.readouterr().out == ""
    assert len(scope["results"]) == scope["K"]
    assert np.isfinite(scope["results"]).all()
    assert (np.asarray(scope["results"]) >= 0).all()


def test_readme():
    scope = _run_first_block("README.md")
    assert len(scope["results"]) == scope["K"]
    assert len(scope["s"]) == scope["K"] + 1
    assert np.isfinite(scope["results"]).all()
