"""Execute the canonical documentation examples rather than maintaining second copies."""

import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _blocks(relpath):
    return re.findall(r"```python3?\n(.*?)```", (ROOT / relpath).read_text(), re.S)


def _run(relpath, blocks, scope=None):
    scope = {} if scope is None else scope
    exec(compile("\n".join(blocks), str(ROOT / relpath), "exec"), scope)
    return scope


def _run_first_block(relpath):
    return _run(relpath, _blocks(relpath)[:1])


def test_intro(capfd):
    scope = _run_first_block("docs/index.md")
    out = capfd.readouterr().out
    # The solvers are silent by default: only the example's own prints remain.
    assert out.splitlines()[0].startswith("worst-case residual:")
    assert len(out.splitlines()) == 2
    q = scope["q"].value
    assert q.shape == (scope["n"],) and np.all(np.abs(q) <= 1 + 1e-9)
    # The printed bound is attained by the incumbent, up to the MIP gap.
    bound = float(out.splitlines()[0].split(":")[1])
    residual = np.max(np.abs((scope["z"] - scope["z_prev"]).value))
    assert abs(residual - bound) <= 1e-4 * max(1.0, bound)


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


def test_quick_start_fixed_point():
    blocks = _blocks("docs/quick_start.md")
    fixed_point = [b for b in blocks if "fixed_point" in b or "s - s_star" in b]
    assert len(fixed_point) == 2
    scope = _run("docs/quick_start.md", blocks[:1] + fixed_point)
    # A time limit keeps the test fast; the best bound is still a valid finite bound.
    assert np.isfinite(scope["vp"].solve(time_limit=5.0))


def test_user_guide_blocks():
    scope = _run_first_block("docs/quick_start.md")
    # Every block except code generation, which needs a coding agent.
    blocks = [b for b in _blocks("docs/user_guide/index.md") if ".codegen(" not in b]
    scope = _run("docs/user_guide/index.md", blocks, scope)
    assert np.isfinite(scope["demo"].solve(time_limit=5.0))
