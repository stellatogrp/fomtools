"""A deterministic, LLM-free agent driver for testing.

The stub drops a pre-written ("golden") solver source into the workspace and
runs the verify command, so the entire code-generation pipeline can be exercised
in CI without any model calls or network access.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from fomtools.codegen.agent.driver import AgentDriver, AgentResult, AgentTask


class StubAgentDriver(AgentDriver):
    """Writes a fixed solver source, then runs the verify command.

    Parameters
    ----------
    solver_source:
        Either C source text or a path to a ``.c`` file to copy into the
        workspace as the solver body.
    """

    def __init__(self, solver_source: str | Path):
        if isinstance(solver_source, Path):
            self._source = solver_source.read_text()
            return
        # A str argument may be either a path to a .c file or the source itself.
        # Guard the filesystem probe: long/multi-line source confuses Path.
        self._source = solver_source
        if "\n" not in solver_source and len(solver_source) < 4096:
            try:
                path = Path(solver_source)
                if path.is_file():
                    self._source = path.read_text()
            except OSError:
                pass

    def run(self, task: AgentTask) -> AgentResult:
        target = task.workdir / task.solver_rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self._source)

        if task.restore_hook is not None:
            task.restore_hook()
        proc = subprocess.run(
            task.verify_cmd,
            cwd=task.workdir,
            capture_output=True,
            text=True,
        )
        log = proc.stdout + proc.stderr
        return AgentResult(success=proc.returncode == 0, log=log, turns_used=1, artifacts=[target])
