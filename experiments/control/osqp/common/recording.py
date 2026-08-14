"""Containers shared by the two formulations' code-generation recordings.

A recording is one ADMM iteration written against a ``Verifier``, plus the boxes
the solve-time parameters live in.  Extraction names the parameters ``p0``,
``p1``, ... in the order they were added, and the emitted C, the reference
oracle, and the ctypes wrapper all use those names, so the boxes carry them too
and nothing downstream has to remember an ordering.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ParameterBox:
    """One solve-time parameter: where it lives, and one value inside it."""

    name: str  # "p0", "p1", ... as extraction names it
    lb: np.ndarray
    ub: np.ndarray
    mid: np.ndarray  # a representative value, taken from mid-flight

    @property
    def dim(self) -> int:
        return int(self.lb.size)


@dataclass
class ParameterBoxes:
    """The region of parameter space a flight visits, one box per parameter.

    Code generation draws its verification samples uniformly from these boxes, so
    they should describe the flight rather than a generous guess around it: a box
    much larger than the truth spends the samples on states the controller never
    reaches.
    """

    boxes: list[ParameterBox]
    state: np.ndarray  # the measured state at the representative instant

    def __iter__(self):
        return iter(self.boxes)

    def __getitem__(self, name: str) -> ParameterBox:
        for box in self.boxes:
            if box.name == name:
                return box
        raise KeyError(name)

    def named_values(self) -> dict[str, np.ndarray]:
        """The representative parameter values, keyed by emitted name."""
        return {box.name: box.mid for box in self.boxes}

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"state": self.state, "names": np.array([b.name for b in self.boxes])}
        for box in self.boxes:
            payload[f"{box.name}_lb"] = box.lb
            payload[f"{box.name}_ub"] = box.ub
            payload[f"{box.name}_mid"] = box.mid
        np.savez(path, **payload)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ParameterBoxes":
        data = np.load(path)
        names = [str(n) for n in data["names"]]
        return cls(
            boxes=[
                ParameterBox(name=n, lb=data[f"{n}_lb"], ub=data[f"{n}_ub"], mid=data[f"{n}_mid"])
                for n in names
            ],
            state=data["state"],
        )


def box_around(values: np.ndarray, margin: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """The elementwise range of ``values``, widened by ``margin`` of its extent.

    The widening lets a flight flown at a different iteration count stay inside a
    box drawn from a well-converged one.
    """
    lo, hi = values.min(axis=0), values.max(axis=0)
    pad = margin * np.maximum(hi - lo, 1e-6)
    return lo - pad, hi + pad


@dataclass
class Recording:
    """One recorded ADMM iteration, with everything extraction needs.

    ``params`` is ordered: extraction names ``params[i]`` as ``p{i}``, which is
    the correspondence :class:`ParameterBoxes` relies on.
    """

    verifier: object
    next_state: list  # the iterate after one iteration
    state: list  # the loop-carried iterate leaves
    params: list  # parameter leaves, in the order they were added
    iterates: list  # successive iterates, for the stationarity check

    def param_values(self, boxes: ParameterBoxes) -> dict:
        """Representative parameter values keyed by *leaf*, as the checks expect."""
        return {leaf: box.mid for leaf, box in zip(self.params, boxes.boxes)}
