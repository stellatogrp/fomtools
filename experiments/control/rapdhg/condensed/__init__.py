"""Put the shared ``common`` package on the path.

Each study runs as a module from its own directory (``python -m
condensed.generate`` from ``experiments/control/osqp``), which puts that
directory on ``sys.path`` but not its parent, where the ``common`` package
shared with the other studies lives.  Importing anything from this package runs
this file first, so the shared modules resolve without an environment variable.

The parent directory is appended rather than prepended: ``condensed/admm.py``
imports the real ``osqp`` solver for its reference solves, and the sibling
``osqp`` study directory must not shadow it.  Appending leaves the installed
package ahead of it; a regular package also beats a namespace portion, so this
is belt and braces.
"""

from __future__ import annotations

import sys
from pathlib import Path

_shared = str(Path(__file__).resolve().parents[2])
if _shared not in sys.path:
    sys.path.append(_shared)
