from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fomtools")
except PackageNotFoundError:
    __version__ = "unknown"

from fomtools.utils.dict_utils import merge_dict, prune_dict
from fomtools.linearexpression import LinearExpression, solve_linexpr, stack
from fomtools.constraint import complementarity, norm, sum_squares
from fomtools.verifier import Verifier
from fomtools.iteratetrajectory import IterateTrajectory
from fomtools.steps.step import Step
from fomtools.sets.set import Set
