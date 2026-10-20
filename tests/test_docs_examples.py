"""Execute the canonical Quick Start rather than maintaining a second copy."""

import re
from pathlib import Path

import numpy as np


def test_quick_start():
    page = Path(__file__).resolve().parents[1] / "docs/index.md"
    code = re.search(r"```python\n(.*?)```", page.read_text(), re.S).group(1)
    scope = {}
    exec(compile(code, str(page), "exec"), scope)
    assert len(scope["results"]) == scope["K"]
    assert np.isfinite(scope["results"]).all()
    assert (np.asarray(scope["results"]) >= 0).all()
