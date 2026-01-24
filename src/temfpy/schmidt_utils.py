# Copyright (C) TeMFPy Developers, MIT license
r"""Utilities for generating the most significant Schmidt states."""

import logging
import heapq
from numbers import Number
from dataclasses import dataclass
from collections.abc import Callable, Iterable

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_SVD_MIN = 1e-6
_DEFAULT_DEG_TOL = 1e-12


@dataclass(frozen=True)
class StoppingCondition:
    """Describes a stopping condition for enumerating Schmidt states."""

    sectors: Callable[[int], bool] | Iterable[int] | int | None = None
    r"""Specifies which charge sectors to retain in one of the following ways:

    * An :obj:`int` → :class:`bool` function that returns whether a given
      charge value should be kept or not.
    * A list of integer charge values to be kept
    * A single integer charge value to be kept
    * :obj:`None` if all charge sectors should be kept (default).

    Internally, all options are converted to an :obj:`int` → :class:`bool` 
    function :func:`is_sector`.

    Note
    ----
        Does not affect methods :func:`__call__` and :func:`truncate` directly, 
        but :func:`is_sector` is used in :func:`lowest_sums` to filter the subsets on the fly.
    """
    chi_max: int | None = None
    """Maximum number of Schmidt states to keep."""
    svd_min: float | None = None
    """Lowest Schmidt value to be kept, as a fraction of the largest one.
    
    Defaults to 1e-6 if not supplied."""
    degeneracy_tol: float | None = None
    r"""Don't cut between neighboring Schmidt values with 
    :math:`|\log(S_i/S_j)|` below :attr:`degeneracy_tol`.

    In other words, keep either both :math:`i` and :math:`j` or neither
    if the Schmidt values are degenerate with a relative error smaller than
    :attr:`degeneracy_tol`.

    Defaults to 1e-12 if not supplied.
    """

    def __post_init__(self):
        # Set default svd_min
        if self.svd_min is None:
            logger.debug(f"trunc_par.svd_min set to default {_DEFAULT_SVD_MIN}")
            object.__setattr__(self, "svd_min", _DEFAULT_SVD_MIN)

        # set default degeneracy_tol
        if self.degeneracy_tol is None:
            logger.debug(f"trunc_par.degeneracy_tol set to default {_DEFAULT_DEG_TOL}")
            object.__setattr__(self, "degeneracy_tol", _DEFAULT_DEG_TOL)

        # Normalise `sectors` to a function
        if self.sectors is None:
            is_sector = lambda _: True
        elif isinstance(self.sectors, Number):
            is_sector = lambda x: x == self.sectors
        elif isinstance(self.sectors, Iterable):
            is_sector = lambda x: x in self.sectors
        elif isinstance(self.sectors, Callable):
            is_sector = self.sectors
        else:
            raise TypeError(f"Unexpected `sectors` parameter {self.sectors!r}")
        object.__setattr__(self, "is_sector", is_sector)

        # sanitise chi_max
        assert (
            self.chi_max is None or self.chi_max > 0
        ), f"`chi_max` must be a positive integer or None, got {self.chi_max!r}"

        # sanitise svd_min
        assert (
            0 < self.svd_min < 1
        ), f"`svd_min` must be between 0 and 1, got {self.svd_min!r}"

        # sanitise degeneracy_tol
        assert (
            self.degeneracy_tol > 0
        ), f"`degeneracy_tol` must be positive, got {self.degeneracy_tol!r}"

        # Find stopping criterion for negative logarithm including degeneracy_tol
        max_logval = -np.log(self.svd_min) + self.degeneracy_tol
        object.__setattr__(self, "max_logval", max_logval)

    def __call__(self, logvals: Iterable[float]) -> bool:
        """Check if any of the stopping conditions had been satisfied.

        Allows for generating slightly more states than the
        stopping conditions require, to make sure degeneracy
        requirements are satisfied:

        - :attr:`chi_max` + 1 states
        - Schmidt values down to :attr:`svd_min` / exp(:attr:`degeneracy_tol`)

        Parameters
        ----------
        logvals:
            Negative logarithms of Schmidt values.
            Must be sorted in increasing order.

        Returns
        -------
            whether more sets need to be generated (:obj:`True`) or we have
            enough (:obj:`False`)

        Note
        ----
            Results generated using this function must be passed through
            :func:`~StoppingCondition.truncate` to finish the truncation
            considering every stopping condition.
        """

        logvals = np.asarray(logvals)
        assert logvals.ndim == 1, f"`logvals` must be a 1D array, got {logvals.ndim!r}"

        if self.chi_max is not None:
            if len(logvals) > self.chi_max:
                return False

        if self.max_logval is not None:
            if logvals[-1] - logvals[0] > self.max_logval:
                return False

        return True

    def truncate(self, logvals: Iterable[float]) -> int:
        """Finds number of Schmidt states to retain to be consistent with every
        constraint (including near-degeneracy).

        Parameters
        ----------
        logvals:
            Negative logarithms of Schmidt values.
            Must be sorted in increasing order.

        Returns
        -------
            Number of Schmidt states to keep

        Note
        ----
        The logic used within the function is based on
        the TeNpy function :func:`~tenpy.linalg.truncation.truncate`.
        """

        logvals = np.asarray(logvals)
        assert logvals.ndim == 1, f"`logvals` must be a 1D array, got {logvals.ndim!r}"

        # good[i] is True if cutting between i and i+1 is OK
        good = np.ones(len(logvals), dtype=np.bool_)

        if self.chi_max is not None:
            good2 = np.zeros(len(logvals), dtype=np.bool_)
            good2[: self.chi_max] = True
            good = good & good2

        if self.svd_min is not None:
            good2 = logvals - logvals[0] < -np.log(self.svd_min)
            good = good & good2

        if self.degeneracy_tol is not None:
            # don't cut between i and i+1 if logvals differ by less than degeneracy_tol
            good2 = np.empty(len(logvals), dtype=np.bool_)
            good2[:-1] = (logvals[1:] - logvals[:-1]) > self.degeneracy_tol
            good2[-1] = True
            good = good & good2

        # keep as many states as allowed by all constraints
        cut = np.nonzero(good)[0][-1]

        return cut + 1


def to_stopping_condition(trunc_par: dict | StoppingCondition) -> StoppingCondition:
    """Standardises stopping conditions.

    Parameters
    ----------
        trunc_par:
            a :class:`StoppingCondition` object or a dictionary with entries
            that can be used to construct one.

    Returns
    -------
        The given stopping conditions as a :class:`StoppingCondition` object."""

    if isinstance(trunc_par, StoppingCondition):
        return trunc_par
    elif isinstance(trunc_par, dict):
        return StoppingCondition(**trunc_par)
    else:
        raise TypeError(
            f"Expected a dictionary or a `StoppingCondition` object, got {trunc_par!r}"
        )


def lowest_sums(
    a: Iterable[float],
    trunc_par: StoppingCondition,
    *,
    filled_left: int | None = None,
    filled_right: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Generates subsets of a given set with the lowest sums.

    This functions generates all possible subsets of ``a`` in order of
    increasing sum, until ``trunc_par`` is satisfied.
    The subsets are then truncated according to ``trunc_par.truncate``.

    For generating the leading Schmidt values, ``a`` should be
    :math:`\log(\lambda_R / \lambda_L)` for the entangled orbitals.

    Uses the algorithm laid out in
    https://stackoverflow.com/a/72117947/27202449.

    Parameters
    ----------
    a:
        The set whose subsets with the lowest sums are to be generated.
    trunc_par:
        Condition to stop generating more subsets and truncate the generated.
    filled_left:
        Number of filled orbitals on the left side,
        used to offset sector labels in the stopping condition.
    filled_right:
        Number of filled orbitals on the right side,
        used to offset sector labels in the stopping condition.

        ``filled_right`` is ignored if ``filled_left`` is given too.

    Returns
    -------
    sums:
        The lowest subset sums of ``a``, sorted, cut off according to
        ``trunc_par``.
    sets:
        The subsets of ``a`` that realise these sums as a boolean array.
    """

    a = np.asarray(a)
    assert a.ndim == 1, f"`a` must be a 1D array, got {a.ndim!r}"

    def N(set: np.ndarray):
        """Number of particles on left/right side of cut in state set."""
        n = set.sum()
        if filled_left is None:
            if filled_right is None:
                return n
            else:
                return filled_right + set.size - n  # number of particles to the right
        else:
            return filled_left + n  # number of particles to the left

    if a.size == 0:  # edge case of the empty array
        # whether to return the empty set
        n_set = int(trunc_par.is_sector(N(np.zeros(0))))
        return np.zeros(n_set), np.zeros((n_set, 0), bool)

    # Find smallest sum
    min_sum = np.sum(a[a < 0])
    min_set = np.array([x < 0 for x in a])

    if trunc_par.is_sector(N(min_set)):
        sums = [min_sum]
        sets = [min_set]
    else:
        sums = []
        sets = []

    # Sort absolute values
    av = np.abs(a)
    idx = np.argsort(av)

    # Find next smallest sum, initialise heap
    # Add seq. no to break degeneracies
    seq = 0
    set = min_set.copy()
    set[idx[0]] = not set[idx[0]]
    heap = [(min_sum + av[idx[0]], seq, 0, set)]

    n_checked = 1  # including the one with the lowest sum
    # Loop to generate subsets in order of increasing sum
    while len(heap) > 0 and trunc_par(sums):
        n_checked += 1
        sum, _, i, set = heapq.heappop(heap)
        if trunc_par.is_sector(N(set)):
            sums.append(sum)
            sets.append(set)

        if i < len(a) - 1:
            c1 = set.copy()
            c1[idx[i + 1]] = not c1[idx[i + 1]]
            sum += av[idx[i + 1]]
            seq += 1
            heapq.heappush(heap, (sum, seq, i + 1, c1))

            c2 = c1.copy()
            c2[idx[i]] = not c2[idx[i]]
            sum -= av[idx[i]]
            seq += 1
            heapq.heappush(heap, (sum, seq, i + 1, c2))

    logger.info("Checked %d subsets", n_checked)

    sums = np.asarray(sums)
    sets = np.asarray(sets)
    cut = trunc_par.truncate(sums)
    logger.info("Kept %d subsets in charge sectors of interest", cut)

    return np.asarray(sums[:cut]), np.asarray(sets[:cut])
