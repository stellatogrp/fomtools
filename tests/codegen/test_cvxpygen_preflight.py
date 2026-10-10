"""generate_solver fails fast when cvxpygen cannot build its Python wrapper.

Regression: in a uv environment without pip, cvxpygen's `python -m pip` build
step failed after the coding agent had run, and the caller saw only
``ImportError: cannot import name 'cpg_module'``.
"""

import importlib.util

import pytest

cp = pytest.importorskip("cvxpy")
from tests.codegen.conftest import importorskip_cvxpygen  # noqa: E402

importorskip_cvxpygen()

from fomtools.cvxpygen import api  # noqa: E402


def _problem():
    x = cp.Variable(2)
    q = cp.Parameter(2, name="q", value=[0.0, 0.0])
    return cp.Problem(cp.Minimize(cp.sum_squares(x) + q @ x), [x <= 1])


def _no_pip(monkeypatch):
    real = importlib.util.find_spec
    monkeypatch.setattr(
        api.importlib.util, "find_spec", lambda name, *a: None if name == "pip" else real(name, *a)
    )


def test_missing_pip_raises_before_any_codegen(monkeypatch, tmp_path):
    _no_pip(monkeypatch)
    called = []
    monkeypatch.setattr(api, "register", lambda: called.append("register"))
    with pytest.raises(RuntimeError, match="pip"):
        api.generate_solver(_problem(), str(tmp_path / "out"), wrapper=True)
    assert called == []  # nothing was canonicalized or generated


def test_missing_cmake_is_reported(monkeypatch, tmp_path):
    monkeypatch.setattr(api.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="cmake"):
        api.generate_solver(_problem(), str(tmp_path / "out"), wrapper=True)


def test_wrapper_false_skips_the_check(monkeypatch, tmp_path):
    _no_pip(monkeypatch)
    sentinel = RuntimeError("reached register")

    def boom():
        raise sentinel

    monkeypatch.setattr(api, "register", boom)
    with pytest.raises(RuntimeError) as err:
        api.generate_solver(_problem(), str(tmp_path / "out"), wrapper=False)
    assert err.value is sentinel
