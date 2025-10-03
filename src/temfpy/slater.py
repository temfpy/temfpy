# Copyright (C) TeMFPy Developers, MIT license
"""Tools for converting Slater determinants into matrix product states (MPS)."""

# Standard library
# ---------------
import logging
from typing import Type
from dataclasses import dataclass

# Third-party libraries
# ---------------
import numpy as np
from numpy.linalg import eigh, det, inv

import tenpy.linalg.np_conserved as npc
from tenpy import networks

# Local imports
# -------------
from .schmidt_utils import lowest_sums, StoppingCondition, to_stopping_condition
from .utils import n_slice, HT, block_svd, normalize_SV
from .testing import assert_allclose, check_schmidt_decomposition, _DIAG_TOL
from . import iMPS

logger = logging.getLogger(__name__)


#### TENPY BINDINGS ####
#### -------------- ####
fermion_site = networks.site.FermionSite()
"""Lattice site prototype for the number-conserving fermion MPS."""
fermion_leg = fermion_site.leg
""":class:`~tenpy.linalg.charges.LegCharge` for the single-site Hilbert space
of the number-conserving fermion MPS."""
chinfo = fermion_leg.chinfo
""":class:`~tenpy.linalg.charges.ChargeInfo` for fermion number conservation."""


#### SCHMIDT ORBITALS ####
#### ---------------- ####
@dataclass(frozen=True)
class SchmidtModes:
    """Mean-field orbitals that generate the Schmidt vectors of a Slater determinant."""

    e: np.ndarray
    """array (:attr:`n_entangled`,) -- 
    Entangled eigenvalues of the left-left block of the correlation matrix,
    in decreasing order."""
    vL: np.ndarray | None
    r"""array (:attr:`nL`, :attr:`nL`) --
    Eigenvectors of the left-left block of the correlation matrix, if computed.

    The eigenvectors are the columns of the matrix in the order

    - filled orbitals (eigenvalue 1);
    - entangled orbitals (eigenvalue between 0 and 1, decreasing order);
    - empty orbitals (eigenvalue 0).

    In particular, the eigenvalues corresponding to
    ``v_LE = vL[:, ixL["entangled"]]`` are given in order by ``e``.

    Note that the entangled orbital vectors ``v_LE`` are also left singular vectors
    of the offdiagonal block :math:`C^{LR}` of the correlation matrix.
    """
    vR: np.ndarray | None
    r"""array (:attr:`nR`, :attr:`nR`) --
    Eigenvectors of the right-right block of the correlation matrix, if computed. 

    The eigenvectors are the columns of the matrix in the order

    - empty orbitals (eigenvalue 0);
    - entangled orbitals (eigenvalue between 0 and 1, decreasing order);
    - filled orbitals (eigenvalue 1).

    In particular, the eigenvalues corresponding to
    ``v_RE = vR[:, ixR["entangled"]]`` are given in order by ``1-e[::-1]``.

    Note that the entangled orbital vectors ``v_RE`` are also right singular
    vectors of the offdiagonal block :math:`C^{LR}` of the correlation matrix.
    
    If both :attr:`vL` and :attr:`vR` are computed, the singular vectors
    ``v_LE[:, i]`` and ``v_RE[:, n_entangled-1-i]`` correspond to each other,
    with eigenvalues :math:`\lambda_i` and :math:`1-\lambda_i`, so that the
    SVD of :math:`C^{LR}` is

    .. math::
        C^{LR} = \sum_{i=0}^{n_E-1} \sqrt{\lambda_i (1-\lambda_i)} \;
                                    v_{L,E,i} \; v_{R,E,n_E-1-i}^\dagger.

    However, to better handle fermion anticommutation, the entangled orbitals
    at odd indices include an additional negative sign."""
    ixL: dict[str, slice] | None
    """Maps the labels ``"empty"``, ``"filled"``, ``"entangled"`` to column
    slices of the corresponding orbitals in :attr:`vL`, if that is computed."""
    ixR: dict[str, slice] | None
    """Maps the labels ``"empty"``, ``"filled"``, ``"entangled"`` to column
    slices of the corresponding orbitals in :attr:`vR`, if that is computed."""
    nL: int
    """Size of the left half of the system."""
    nR: int
    """Size of the right half of the system."""
    n_fermion: int
    """Number of fermions in the mean-field state."""

    def __post_init__(self):
        err = "`ixL` and `vL` must be specified together"
        assert (self.vL is None) == (self.ixL is None), err
        err = "`ixR` and `vR` must be specified together"
        assert (self.vR is None) == (self.ixR is None), err
        err = "Must specify at least one of `vL`, `vR`"
        assert (self.vL is not None) or (self.vR is not None), err
        if self.vL is not None:
            assert self.nL == len(self.vL), "`nL` must match the size of `vL`"
        if self.vR is not None:
            assert self.nR == len(self.vR), "`nR` must match the size of `vR`"

    @property
    def n_entangled(self) -> int:
        """Number of entangled orbitals."""
        return self.e.size

    def size(self, which: str = "T") -> int:
        """Size of the specified half or the whole of the system.

        Parameters
        ----------
        which:
            Either "L" for left or "R" for right side
            or "T" (default) for total size.

        Returns
        -------
            :attr:`nL`, :attr:`nR`, or their sum, depending on ``which``.
        """
        which_ = which[0].upper()
        if which_ == "L":
            return self.nL
        elif which_ == "R":
            return self.nR
        elif which_ == "T":
            return self.nL + self.nR
        else:
            raise ValueError("`which` must start with L, R, or T, got " + repr(which))

    def n_filled(self, which: str) -> int:
        """Number of filled orbitals on the specified half of the system.

        Based on :attr:`vL` or :attr:`vR` if they exist, otherwise inferred from
        :attr:`n_fermion` and the number of entangled and filled orbitals
        on the other side.

        Parameters
        ----------
        which:
            Whether to return the number of filled orbitals
            to the left ("L") or the right ("R").

        Returns
        -------
            Number of filled orbitals on the specified side.
        """
        which_ = which[0].upper()
        if which_ == "L":
            if self.ixL is not None:
                return n_slice(self.ixL["filled"])
            else:
                return self.n_fermion - self.n_entangled - n_slice(self.ixR["filled"])
        elif which_ == "R":
            if self.ixR is not None:
                return n_slice(self.ixR["filled"])
            else:
                return self.n_fermion - self.n_entangled - n_slice(self.ixL["filled"])
        else:
            raise ValueError("`which` must start with L or R, got " + repr(which))

    @property
    def vL_entangled(self) -> np.ndarray | None:
        """Entangled left Schmidt mode orbitals, if computed."""
        return None if self.vL is None else self.vL[:, self.ixL["entangled"]]

    @property
    def vR_entangled(self) -> np.ndarray | None:
        """Entangled right Schmidt mode orbitals, if computed."""
        return None if self.vR is None else self.vR[:, self.ixR["entangled"]]

    def mode_vectors(self, which: str, entangled: bool = False) -> np.ndarray | None:
        """Returns the Schmidt mode orbitals on the specified side.

        Parameters
        ----------
        which:
            Either "L" for left or "R" for right side.
        entangled:
            Whether to return the entangled (:obj:`True`) or
            all (:obj:`False`, default) eigenvectors.

        Returns
        -------
            Either :attr:`vL` or :attr:`vR`, depending on ``which``,
            truncated to entangled modes if ``entangled``.
        """
        which_ = which[0].upper()
        if which_ == "L":
            return self.vL_entangled if entangled else self.vL
        elif which_ == "R":
            return self.vR_entangled if entangled else self.vR
        else:
            raise ValueError("`which` must start with L or R, got " + which)

    def eigenvalues(self, which: str, entangled: bool = False) -> np.ndarray | None:
        """Returns the Schmidt mode eigenvalues on the specified side.

        Parameters
        ----------
        which:
            Either "L" for left or "R" for right side.
        entangled:
            Whether to return the entangled (:obj:`True`) or
            all (:obj:`False`, default) eigenvalues.

        Returns
        -------
            The eigenvalues corresponding to :meth:`mode_vectors`
            with the same parameters.
        """
        which_ = which[0].upper()
        if which_ == "L":
            if self.vL is None:
                return None
            e = self.e
            if entangled:
                return e
            else:
                E = np.zeros(len(self.vL))
                E[self.ixL["filled"]] = 1
                E[self.ixL["entangled"]] = e
                return E
        elif which_ == "R":
            if self.vR is None:
                return None
            e = 1 - self.e[::-1]
            if entangled:
                return e
            else:
                E = np.zeros(len(self.vR))
                E[self.ixR["filled"]] = 1
                E[self.ixR["entangled"]] = e
                return E
        else:
            raise ValueError("`which` must start with L or R, got " + repr(which))

    @property
    def singular_values(self) -> np.ndarray | None:
        """Singular values of the offdiagonal correlation matrix blocks.

        If :attr:`vL_entangled` and :attr:`vR_entangled` are both known, satisfies

        .. code::

            C_LR == vL_entangled @ diag(S) @ vR_entangled[:, ::-1].T.conj()

        Otherwise, :obj:`None`.
        """
        if (self.vL is None) or (self.vR is None):
            return None
        SV = (self.e * (1 - self.e)) ** 0.5
        sign = (-1) ** (np.arange(SV.size)[::-1])  # anticommutation signs on right SV
        return SV * sign

    @classmethod
    def from_correlation_matrix(
        cls: Type["SchmidtModes"],
        C: np.ndarray,
        x: int,
        trunc_par: dict | StoppingCondition,
        *,
        which: str = "LR",
        diag_tol: float = _DIAG_TOL,
    ) -> "SchmidtModes":
        r""":class:`~SchmidtModes` of a mean-field state with correlation matrix ``C``
        for an entanglement cut between sites ``x-1`` and ``x`` (zero-indexed).

        We start by diagonalising the respective diagonal blocks.
        The eigenvectors give the Schmidt modes, the eigenvalues their
        relative weight in the entangled mode. We only treat modes with
        eigenvalues away from 0 or 1 by at least :attr:`trunc_par.svd_min`
        `squared` as entangled, as only these can contribute to a Schmidt value
        less than :attr:`trunc_par.svd_min`.

        Parameters
        ----------
        C:
            The correlation matrix, :math:`C_{ij} = \langle c_j^\dagger c_i\rangle`.
        x:
            Position of the entanglement cut.
        trunc_par:
            Which Schmidt modes should be kept as entangled.

            Must be either a :class:`~temfpy.schmidt_sets.StoppingCondition` object
            or a dictionary with matching keys.
        which:
            Whether to return left and/or right Schmidt modes.

            Must be a combination of ``"L"`` and ``"R"``.
        diag_tol:
            If ``which == "LR"``, largest allowed offdiagonal matrix element in
            diagonalised / SVD correlation submatrices before an error is
            raised.

        Note
        ----
        - If :attr:`trunc_par.svd_min` is not provided, a default of 1e-6
          (i.e., a truncation threshold of 1e-12) is used.
        - If :attr:`trunc_par.degeneracy_tol` is not provided, the degeneracy tolerance
          defaults to 1e-12.
        """
        trunc_par = to_stopping_condition(trunc_par)
        cutoff = trunc_par.svd_min**2  # eigenvalues -> squared Schmidt values

        which = which.upper()
        err = "`which` must specify at least one of (L)eft or (R)ight"
        assert ("L" in which) or ("R" in which), err

        def diag_and_separate(c, order, needed):
            """Returns spectrum of c separated into entangled, filled, empty
            blocks.

            Modes are sorted by decreasing eigenvalue.
            """
            if not needed:  # just need dummies for tuple unpacking
                return (None,) * 4

            n, m = c.shape
            assert n == m, f"Got non-square {c.shape} submatrix"

            if n == 0:
                e = np.zeros((0,), float)
                v = np.zeros((0, 0), c.dtype)
                ix = {
                    "filled": slice(0, 0),
                    "entangled": slice(0, 0),
                    "empty": slice(0, 0),
                }
                return e, v, ix, 0

            # diagonalise c
            e, v = eigh(c)

            # split empty and filled modes, determine number of each
            x0, x1 = np.searchsorted(e, [cutoff, 1 - cutoff])
            n0, k, n1 = x0, x1 - x0, n - x1

            # order as desired
            idx = np.arange(n)
            if order == "L":
                idx = idx[::-1]
                ix = {
                    "filled": slice(0, n1),
                    "entangled": slice(n1, n1 + k),
                    "empty": slice(n1 + k, n),
                }
            elif order == "R":
                idx[x0:x1] = idx[x0:x1][::-1]
                ix = {
                    "empty": slice(0, x0),
                    "entangled": slice(x0, x1),
                    "filled": slice(x1, n),
                }
            e = e[idx]
            v = v[:, idx]

            # only keep Schmidt values for entangled modes
            e = e[ix["entangled"]]

            return e, v, ix, k

        which = which.upper()

        eL, vL, ixL, kL = diag_and_separate(C[:x, :x], "L", "L" in which)
        eR, vR, ixR, kR = diag_and_separate(C[x:, x:], "R", "R" in which)

        if eL is None:
            if eR is None:
                raise RuntimeError()  # should error earlier
            else:
                e = 1.0 - eR[::-1]
                k = kR
        else:
            if eR is None:
                e = eL
                k = kL
            else:
                # both "L" and "R" were done, need consistency checks
                assert kL == kR  # number of entangled modes must match
                k = kL

                # Shorthands
                vLE = vL[:, ixL["entangled"]]
                vRE = vR[:, ixR["entangled"]]
                deg_tol = trunc_par.degeneracy_tol

                # Check that entangled mode weights sum to 1
                err = "Eigenvalues of C_LL and C_RR do not match"
                assert_allclose(eL + eR[::-1], 1.0, rtol=0, atol=deg_tol, err_msg=err)

                e = eL
                block_svd(C[:x, x:], vLE, vRE[:, ::-1], e, deg_tol)

                # add extra anticommutation signs
                vRE[:, 1::2] *= -1

        logger.info("%d Schmidt modes found", k)

        n_fermion = int(np.round(np.trace(C).real))
        nR = len(C) - x
        modes = cls(
            n_fermion=n_fermion, e=e, vL=vL, vR=vR, ixL=ixL, ixR=ixR, nL=x, nR=nR
        )

        if (eL is not None) and (eR is not None):
            check_schmidt_decomposition(modes, C, diag_tol)

        return modes

    @property
    def e_ratio(self) -> np.ndarray:
        r""":math:`\log((1-\lambda)/\lambda` for all eigenvalues in :attr:`e`."""
        return np.log((1 - self.e) / self.e)

    def embed_subsets(
        self, sets: np.ndarray
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        r"""Given an array of subsets of entangled orbitals occupied on the left
        side, generates the full sets of occupied orbitals on either side.

        Parameters
        ----------
        sets: bool :class:`~np.ndarray` (n, :attr:`n_entangled`)
            Array of occupation numbers.

            Each row specifies one Schmidt state by listing on which side of the
            entanglement cut each entangled orbital is filled:
            True if on the left side, False if on the right side.

        Returns
        -------
        left_sets: bool :class:`~np.ndarray` (n, :attr:`size`\[0]) | :obj:`None`
            Occupation of every left orbital in the input Schmidt states.

            Returned if :attr:`vL` is not :obj:`None`.
        right_sets: bool :class:`~np.ndarray` (n, :attr:`size`\[1]) | :obj:`None`
            Occupation of every right orbital in the input Schmidt states.

            Returned if :attr:`vR` is not :obj:`None`.
        """
        if self.vL is not None:
            left_sets = np.zeros((len(sets), self.nL), dtype=bool)
            left_sets[:, self.ixL["entangled"]] = sets
            left_sets[:, self.ixL["filled"]] = True
        else:
            left_sets = None

        if self.vR is not None:
            right_sets = np.zeros((len(sets), self.nR), dtype=bool)
            right_sets[:, self.ixR["entangled"]] = np.logical_not(sets[:, ::-1])
            right_sets[:, self.ixR["filled"]] = True
        else:
            right_sets = None

        return left_sets, right_sets

    def schmidt_values(modes, sets: np.ndarray) -> np.ndarray:
        r"""Schmidt values of the Schmidt vectors with given occupation numbers.

        Parameters
        ----------
        sets: bool :class:`~np.ndarray` (n, :attr:`n_entangled`)
            Array of occupation numbers.

            Each row specifies one Schmidt state by listing on which side of the
            entanglement cut each entangled orbital is filled:
            True if on the left side, False if on the right side.

        Returns
        -------
        λ: :class:`np.ndarray` (n,)
            Schmidt values corresponding to the input Schmidt states.
        """
        return np.where(sets, modes.e, 1 - modes.e).prod(axis=1) ** 0.5


#### SCHMIDT Vectors ####
#### ---------------- ####
@dataclass(frozen=True)
class SchmidtVectors:
    r"""Schmidt vectors of a Slater determinant.

    The Schmidt decomposition of the state :math:`|\psi\rangle` is given by

    .. math::

        |\psi\rangle &= \sum_\alpha \lambda_\alpha |L_\alpha\rangle \otimes_g
                                                   |R_\alpha\rangle

        |L_\alpha\rangle &= \prod_a (d^\dagger_{L,a})^{n^L_{\alpha a}} |0\rangle

        |R_\alpha\rangle &= \prod_a (d^\dagger_{R,a})^{n^R_{\alpha a}} |0\rangle,

    where :math:`\lambda_\alpha` are the Schmidt values,
    :math:`d^\dagger_{L,a}` and :math:`d^\dagger_{L,a}` are the creation operators
    of the left and right Schmidt mode orbitals, and
    :math:`n^L_{\alpha a}` and :math:`n^R_{\alpha a}` are the occupation numbers
    of these orbitals in the Schmidt vectors.
    """

    modes: SchmidtModes
    """The mean-field orbitals underlying the Schmidt vectors."""
    left_sets: np.ndarray | None
    r"""bool (:attr:`n_schmidt`, :attr:`nL`) --
    Left Schmidt vectors.
     
    Each row contains the occupation :math:`n^L_{\alpha a}` of all left 
    Schmidt modes in one left Schmidt vector :math:`|L_\alpha\rangle`,
    if :attr:`vL` is not :obj:`None`."""
    right_sets: np.ndarray | None
    r"""bool (:attr:`n_schmidt`, :attr:`nR`) --
    Right Schmidt vectors.
      
    Each row contains the occupation :math:`n^R_{\alpha a}` of all left 
    Schmidt modes in one left Schmidt vector :math:`|R_\alpha\rangle`,
    if :attr:`vR` is not :obj:`None`."""
    schmidt_values: np.ndarray
    r"""(:attr:`n_schmidt`,) --
    Schmidt values :math:`\lambda_\alpha` corresponding to each Schmidt vector.
     
    Sorted in increasing order of charge of the left Schmidt vector and
    in decreasing order within each charge sector."""
    idx_L: dict[int, slice]
    r"""Maps the total charge to the left of the entanglement cut to the
    slice of sets/singular values with that charge.

    That is, all Schmidt vectors in ``left_sets[idx_L[n]]`` contain n particles.
    """

    @property
    def n_schmidt(self) -> int:
        """The number of Schmidt vectors."""
        return len(self.schmidt_values)

    @property
    def n_entangled(self) -> int:
        """Number of entangled orbitals."""
        return self.modes.n_entangled

    @property
    def nL(self) -> int:
        """Size of the left half of the system."""
        return self.modes.nL

    @property
    def nR(self) -> int:
        """Size of the right half of the system."""
        return self.modes.nR

    @property
    def n_fermion(self) -> int:
        """Number of fermions in the mean-field state."""
        return self.modes.n_fermion

    def size(self, which: str = "T") -> int:
        """Size of the specified half or the whole of the system.

        Parameters
        ----------
        which:
            Either "L" for left or "R" for right side
            or "T" (default) for total size.

        Returns
        -------
            the appropriate system size.
        """
        return self.modes.size(which)

    @property
    def vL(self) -> np.ndarray | None:
        """Left Schmidt mode orbitals :attr:`~SchmidtModes.vL`."""
        return self.modes.vL

    @property
    def vR(self) -> np.ndarray | None:
        """Right Schmidt mode orbitals :attr:`~SchmidtModes.vR`."""
        return self.modes.vR

    def mode_vectors(self, which: str, entangled: bool = False) -> np.ndarray | None:
        """Returns the Schmidt mode orbitals on the specified side.

        Parameters
        ----------
        which:
            Either "L" for left or "R" for right side.
        entangled:
            Whether to return the entangled (:obj:`True`) or
            all (:obj:`False`, default) eigenvectors.

        Returns
        -------
            Either :attr:`vL` or :attr:`vR`, depending on ``which``,
            truncated to entangled modes if ``entangled``.
        """
        return self.modes.mode_vectors(which, entangled)

    def sets(self, which: str):
        """Returns the sets of occupied orbitals on the specified side.

        Parameters
        ----------
        which:
            Either "L" for left or "R" for right side.

        Returns
        -------
            Either :attr:`left_sets` or :attr:`right_sets`, depending on ``which``.
        """
        which_ = which[0].upper()
        if which_ == "L":
            return self.left_sets
        elif which_ == "R":
            return self.right_sets
        else:
            raise ValueError("`which` must start with L or R, got " + which)

    @classmethod
    def from_schmidt_modes(
        cls: Type["SchmidtVectors"],
        modes: SchmidtModes,
        trunc_par: dict | StoppingCondition,
    ) -> "SchmidtVectors":
        """The most significant :class:`SchmidtVectors` from the given
        :class:`SchmidtModes`.

        Parameters
        ----------
        modes:
            The Schmidt modes.
        trunc_par:
            Specifies which Schmidt states should be kept.

            Must be either a :class:`~temfpy.schmidt_sets.StoppingCondition` object
            or a dictionary with matching keys.

            If ``modes`` contains left modes,
            the filtering function :attr:`~StoppingCondition.sectors`
            is applied to the total number of particles to the left,
            otherwise, to the total number of particles to the right.
        """
        trunc_par = to_stopping_condition(trunc_par)

        # find the sets of mixed left states with the largest Schmidt
        # value, bzw. the largest product of the corresponding L (R for the rest)
        # this is equivalent to finding the subsets with the lowest sum of log(R/L)
        _, sets = lowest_sums(
            modes.e_ratio / 2,  # apply svd_min to the Schmidt values not squared
            trunc_par,
            filled_left=modes.n_filled("L"),
            filled_right=modes.n_filled("R"),
        )
        if len(sets) == 0:
            err = "No Schmidt vectors left after filtering by `trunc_par.sectors`!"
            raise ValueError(err)

        # compute particle number in Schmidt vectors for sorting
        n_L = modes.n_filled("L") + sets.sum(axis=1)

        # sort by n_L
        idx = np.argsort(n_L, kind="stable")
        n_L = n_L[idx]
        sets = sets[idx]

        # cluster by n_L
        n_L, idx_L = np.unique(n_L, return_index=True)
        idx_L = np.concatenate((idx_L, [len(sets)]))
        idx_L = {n: slice(idx_L[i], idx_L[i + 1]) for i, n in enumerate(n_L)}

        # embed `sets` into the full span of Schmidt modes
        left_sets, right_sets = modes.embed_subsets(sets)

        # compute Schmidt values
        λ = modes.schmidt_values(sets)

        logger.info("%d Schmidt vectors generated", len(λ))
        logger.info("Dynamical range: %.3e", λ.max() / λ.min())

        return cls(
            modes=modes,
            left_sets=left_sets,
            right_sets=right_sets,
            schmidt_values=λ,
            idx_L=idx_L,
        )

    @classmethod
    def from_correlation_matrix(
        cls: Type["SchmidtVectors"],
        C: np.ndarray,
        x: int,
        trunc_par: dict | StoppingCondition,
        *,
        which: str = "LR",
        diag_tol: float = _DIAG_TOL,
    ) -> "SchmidtVectors":
        r"""Most significant :class:`~SchmidtVectors` of a Slater determinant with
        correlation matrix ``C`` for an entanglement cut between sites
        ``x-1`` and ``x`` (zero-indexed).

        Parameters
        ----------
        C:
            The correlation matrix, :math:`C_{ij} = \langle c_j^\dagger c_i\rangle`.
        x:
            Position of the entanglement cut.
        trunc_par:
            Specifies which Schmidt states should be kept.

            Must be either a :class:`~temfpy.schmidt_sets.StoppingCondition` object
            or a dictionary with matching keys.

            If left Schmidt modes are calculated,
            the filtering function :attr:`~StoppingCondition.sectors`
            is applied to the total number of particles to the left,
            otherwise, to the total number of particles to the right.
        which:
            Whether to return left and/or right Schmidt modes.

            Must be a combination of ``"L"`` and ``"R"``.
        diag_tol:
            If ``which == "LR"``, largest allowed offdiagonal matrix element in
            diagonalised / SVD correlation submatrices before an error is
            raised.

        Note
        ----
        - If :attr:`trunc_par.svd_min` is not provided, the truncation threshold
          defaults to 1e-6.
        - If :attr:`trunc_par.degeneracy_tol` is not provided, the degeneracy tolerance
          defaults to 1e-12.
        """
        which = which.upper()
        trunc_par = to_stopping_condition(trunc_par)

        modes = SchmidtModes.from_correlation_matrix(
            C, x, trunc_par, which=which, diag_tol=diag_tol
        )
        vectors = cls.from_schmidt_modes(modes, trunc_par)
        return vectors


#### MPSTensorData from SCHMIDT Vectors ####
#### ---------------------------------- ####
def _select_orbitals(
    sets: np.ndarray, V: np.ndarray, mode: str
) -> tuple[np.ndarray, np.ndarray, int]:
    """Crops Schmidt orbitals to always + sometimes orbitals
    and computed anticommutation signs due to reordering.

    If MPS tensors are to be computed, the on-site degree of freedom
    must be explicitly added to ``sets`` and ``V`` as one of the
    "sometimes" orbitals.

    Parameters
    ----------
    sets:
        :attr:`SchmidtVectors.left_sets` or :attr:`SchmidtVectors.right_sets`,
        depending on ``mode``.
    V:
        :attr:`SchmidtModes.vL` or :attr:`SchmidtModes.vR`, depending on ``mode``.
    mode:
        whether ``sets`` corresponds to a "left" or "right" set

    Returns
    -------
    sets:
        Input `sets` trimmed to always + sometimes orbitals.
    V:
        Input `V` trimmed to always + sometimes orbitals.

        Includes commutation sign to move sometimes orbitals to the "right"
        or to the "left" of the occupied orbitals, depending on ``mode``.
    k:
        Number of always orbitals.
    """
    always = np.all(sets, axis=0)
    never = np.logical_not(np.any(sets, axis=0))
    sometimes = np.logical_not(always | never)

    (always,) = np.nonzero(always)
    (sometimes,) = np.nonzero(sometimes)
    k = len(always)

    # There may be some entangled orbitals that are always occupied,
    # due to the additional cutoff of Schmidt vectors by `chi_max`. We now move
    # these orbitals to one side of the occupied block resulting in
    # commutation signs, keeping the relative order of the sometimes
    # orbitals. The commutation sign is calculated for the sometimes orbitals
    # because the always orbitals are the same for all Schmidt vectors.
    if mode == "left":
        idx = np.concatenate((always, sometimes))
        # For the left Schmidt vectors, we want to move the sometimes orbitals
        # to the right. The number of commutations for each sometimes orbital
        # is therefore the number of always orbitals to its right,
        # which is the total number of always orbitals minus the number of
        # always orbitals to its left.
        sign = (-1) ** (k - np.searchsorted(always, sometimes))
        sign = np.concatenate((np.ones(k), sign))
    elif mode == "right":
        idx = np.concatenate((sometimes, always))
        # For the right Schmidt vectors, we want to move the entangled orbitals
        # to the left. The number of commutations for each sometimes orbital is
        # therefore the number of always orbitals to its left.
        sign = (-1) ** np.searchsorted(always, sometimes)
        sign = np.concatenate((sign, np.ones(k)))
    else:
        raise ValueError('mode needs to be either "left" or "right"')

    return sets[:, idx], V[:, idx] * sign, k


def _tensor_block(
    sometimes_matrix: np.ndarray, new_sets_bra: np.ndarray, new_sets_ket: np.ndarray
) -> np.ndarray:
    """Computes a fixed-particle-number block of the MPS tensor.

    Parameters
    ----------
    new_sets_bra:
        rows of :attr:`new_sets_bra` corresponding to a given charge
        physical leg must be set appropriately too
    new_sets_ket:
        rows of :attr:`new_sets_ket` corresponding to a given charge

    Returns
    -------
    The MPS tensor block for the given charges.
    The row index always corresponds to the `bra` leg
    """
    # check that every new_set has the same number of particles
    n_bra = new_sets_bra.sum(axis=1)
    assert np.all(n_bra == n_bra[0])
    n_bra = n_bra[0]

    n_ket = new_sets_ket.sum(axis=1)
    assert np.all(n_ket == n_ket[0])
    n_ket = n_ket[0]

    assert n_bra == n_ket

    # convert new_sets to lists of positions for convenience
    nsb = len(new_sets_bra)
    _, new_sets_bra = new_sets_bra.nonzero()
    new_sets_bra = new_sets_bra.reshape(nsb, n_bra)

    nsk = len(new_sets_ket)
    _, new_sets_ket = new_sets_ket.nonzero()
    new_sets_ket = new_sets_ket.reshape(nsk, n_ket)

    O = sometimes_matrix[new_sets_bra][:, :, new_sets_ket]
    O = np.transpose(O, (0, 2, 1, 3))

    return det(O)


@dataclass(frozen=True)
class MPSTensorData:
    r"""Data for computing one MPS tensor of a Slater determinant.

    - If :attr:`mode` is ``"left"``, contains an implicit description of the
      left canonical tensor

      .. math::

        A^{n_i}_{\alpha\beta} =
        (\langle n_i | \otimes_g \langle L^{(i-1)}_\alpha|)
        | L^{(i)}_\beta \rangle.

    - If :attr:`mode` is ``"right"``, contains an implicit description of the
      right canonical tensor

      .. math::

        B^{n_i}_{\beta\alpha} =
        (\langle R^{(i)}_\alpha | \otimes_g \langle n_i |)
        | R^{(i-1)}_\beta \rangle.

    Schmidt vector overlaps for equal-length chains can also be computed:

    .. math::

        A_{\alpha\beta} &= \langle L'_\alpha | L_\beta \rangle

        B_{\beta\alpha} &= \langle R'_\beta | R_\alpha \rangle

    For Slater determinants, such overlaps are the determinants of the
    overlaps of the single-particle orbital wave functions. Many of these
    orbitals are shared between every Schmidt vector, so we can compute
    the determinants efficiently using the identity

    .. math::

        \det \begin{bmatrix}A & B \\ C & D\end{bmatrix} =
        \det(A) \det\left(D - C A^{-1} B\right) =
        \det(D) \det\left(A - B D^{-1} C\right).

    Namely, if the block :math:`A` or block :math:`D` contains the overlaps
    of the always occupied orbitals, its determinant is only computed once.
    Furthermore, each determinant entry :math:`\left(D - C A^{-1} B\right)_{ij}` /
    :math:`\left(A - B D^{-1} C\right)_{ij}` depends only on "bra" orbital `i`
    and "ket" orbital `j`, so they can be precomputed for all pairs of
    sometimes-occupied orbitals.

    - For left Schmidt vectors, the always occupied orbitals are listed before
      the entangled ones (cf. :attr:`SchmidtModes.vL`).
      Therefore, we use the first form of the identity if ``mode == "left"``.
    - For right Schmidt vectors, the always occupied orbitals are listed after
      the entangled ones (cf. :attr:`SchmidtModes.vR`).
      Therefore, we use the second form of the identity if ``mode == "right"``.
    """

    mode: str
    """Whether the overlap is between ``"left"`` or ``"right"`` Schmidt vectors."""
    physical_leg: bool
    """Whether an MPS tensor with a physical leg (:obj:`True`) or an array of
    overlaps (:obj:`False`) is to be computed."""
    det_always: float | complex
    r"""Overlap determinant of the always occupied orbitals,
    :math:`\det(A)` or :math:`\det(D)`."""
    sometimes_matrix: np.ndarray
    r"""Entries of the matrix :math:`\left(D - C A^{-1} B\right)` or
    :math:`\left(A - B D^{-1} C\right)` for all pairs of sometimes-occupied orbitals."""
    idx_bra: dict[int, slice]
    """:attr:`~SchmidtVectors.idx_L` of the bra Schmidt vector."""
    idx_ket: dict[int, slice]
    """:attr:`~SchmidtVectors.idx_L` of the ket Schmidt vector."""
    new_sets_bra: np.ndarray
    """Bra Schmidt vectors as occupation numbers of the sometimes-occupied orbitals.
    
    Used to index the rows of :attr:`sometimes_matrix`.
    
    If :attr:`physical` is :obj:`True`, it is double the length of
    :attr:`~SchmidtVectors.sets` and contains all Schmidt vectors with the on-site
    degree of freedom once empty, one filled. The overall array is still sorted
    by total charge to the left; within each sector, we first take the "Schmidt
    vectors" with empty on-site orbital."""
    new_sets_ket: np.ndarray
    """Ket Schmidt vectors as occupation numbers of the sometimes-occupied orbitals.
    
    Used to index the columns of :attr:`sometimes_matrix`."""
    qtotal: int
    """Total charge of the tensor to ensure matching fermion numbers in
    Schmidt vectors.
    
    - For ``mode == "left"``, always 0.
    - For ``mode == "right"``, equals the difference of :attr:`~SchmidtVectors.n_fermion`
      between the ket and bra Schmidt vectors.
    """

    @property
    def idx_physical(self) -> int | None:
        """Row index of the onsite degree of freedom in :attr:`sometimes_matrix`
        or :obj:`None` if there is no such degree of freedom."""
        if self.physical_leg:
            return len(self.sometimes_matrix) - 1 if self.mode == "left" else 0
        else:
            return None

    @classmethod
    def from_schmidt_vectors(
        cls: Type["MPSTensorData"],
        Schmidt_bra: SchmidtVectors,
        Schmidt_ket: SchmidtVectors,
        mode: str,
    ) -> "MPSTensorData":
        """
        Constructs :class:`MPSTensorData` from Schmidt vectors on the two
        entanglement cuts next to a site.

        Depending on the value of ``mode``, it uses either "left" or "right"
        Schmidt vectors, resulting in a left or right canonical MPS tensor,
        respectively.
        In both cases, ``Schmidt_bra`` corresponds to Schmidt vectors on the
        shorter chain.

        Parameters
        ----------
        Schmidt_bra:
            Schmidt vectors corresponding to the bra states in the overlap.

            That is, for the entanglement cut to the left if ``mode=="left"``
            or to the right if ``mode="right"``.
        Schmidt_ket:
            Schmidt vectors corresponding to the ket states in the overlap.

            That is, for the entanglement cut to the right if ``mode=="left"``
            or to the left if ``mode="right"``.
        mode:
            Whether to construct the tensor from left or right Schmidt vectors.

            Must be either "left" or "right".
        """
        mode = mode.lower()
        if mode not in ["left", "right"]:
            raise ValueError("mode must be either 'left' or 'right', got " + repr(mode))

        # shorthands
        v_bra = Schmidt_bra.mode_vectors(mode)
        assert v_bra is not None, f"`Schmidt_bra` contains no {mode} Schmidt vectors"
        sets_bra = Schmidt_bra.sets(mode)

        v_ket = Schmidt_ket.mode_vectors(mode)
        assert v_ket is not None, f"`Schmidt_ket` contains no {mode} Schmidt vectors"
        sets_ket = Schmidt_ket.sets(mode)

        # bra must be 1 longer than ket
        if sets_bra.shape[1] == sets_ket.shape[1]:
            physical = False
        elif sets_bra.shape[1] + 1 == sets_ket.shape[1]:
            physical = True
            ns_bra, n_bra = sets_bra.shape
            # add the physical orbital and combine sets_bra and sets_phys
            # sets_phys shall be more major to sets_bra
            if mode == "left":
                # Add the physical orbital to the end
                v_bra = np.block(
                    [[v_bra, np.zeros((n_bra, 1))], [np.zeros((1, n_bra)), 1]]
                )
                sets_bra = np.block(
                    [
                        [sets_bra, np.zeros((ns_bra, 1), bool)],
                        [sets_bra, np.ones((ns_bra, 1), bool)],
                    ]
                )
            else:  # mode == "right":
                # Add the physical orbital to the front
                v_bra = np.block(
                    [[1, np.zeros((1, n_bra))], [np.zeros((n_bra, 1)), v_bra]]
                )
                sets_bra = np.block(
                    [
                        [np.zeros((ns_bra, 1), bool), sets_bra],
                        [np.ones((ns_bra, 1), bool), sets_bra],
                    ]
                )
            # sort by total charge to the left, keeping each sector in original order
            q_bra_sets = sets_bra.sum(axis=1)
            ix_bra_sets = np.argsort(
                q_bra_sets if mode == "left" else -q_bra_sets,  # proxy for q_left
                kind="stable",  # charge sectors have to remain in original order
            )
            sets_bra = sets_bra[ix_bra_sets]
        else:
            raise ValueError(
                f"{mode.capitalize()} sides `Schmidt_bra` and `Schmidt_ket` must match\n"
                f"or `Schmidt_bra` must be one bond to the {mode} of `Schmidt_ket`,\n"
                f"got lengths {sets_bra.shape[1]} and {sets_ket.shape[1]}."
            )

        sets_bra, v_bra, k_bra = _select_orbitals(sets_bra, v_bra, mode)
        sets_ket, v_ket, k_ket = _select_orbitals(sets_ket, v_ket, mode)

        k = min(k_bra, k_ket)  # need square "always" matrix

        O = HT(v_bra) @ v_ket

        if k == 0:
            # no "always" orbitals, default to standard method with "sometimes" orbitals
            det_always = 1.0
            sometimes_matrix = O
        elif mode == "left":
            # det(a,b;c,d) = det(a) det[d - c inv(a) b]
            det_always = det(O[:k, :k])
            sometimes_matrix = O[k:, k:] - O[k:, :k] @ inv(O[:k, :k]) @ O[:k, k:]
            sets_bra = sets_bra[:, k:]
            sets_ket = sets_ket[:, k:]
        else:  # mode == "right":
            # det(a,b;c,d) = det(d) det[a - b inv(d) c]
            det_always = det(O[-k:, -k:])
            sometimes_matrix = (
                O[:-k, :-k] - O[:-k, -k:] @ inv(O[-k:, -k:]) @ O[-k:, :-k]
            )
            sets_bra = sets_bra[:, :-k]
            sets_ket = sets_ket[:, :-k]

        qtotal = 0 if mode == "left" else Schmidt_ket.n_fermion - Schmidt_bra.n_fermion

        return cls(
            mode=mode,
            physical_leg=physical,
            det_always=det_always,
            sometimes_matrix=sometimes_matrix,
            idx_bra=Schmidt_bra.idx_L,
            idx_ket=Schmidt_ket.idx_L,
            new_sets_bra=sets_bra,
            new_sets_ket=sets_ket,
            qtotal=qtotal,
        )

    def to_npc_array(self) -> npc.Array:
        """The MPS tensor as a TeNPy :class:`~tenpy.linalg.np_conserved.Array` object."""

        # Legs of the tensor
        # ------------------
        qconj = (+1, -1) if self.mode == "left" else (-1, +1)
        name_bra, name_ket = ("vL", "vR") if self.mode == "left" else ("vR", "vL")
        leg_bra = npc.LegCharge.from_qdict(chinfo, self.idx_bra, qconj=qconj[0])
        leg_ket = npc.LegCharge.from_qdict(chinfo, self.idx_ket, qconj=qconj[1])
        if self.physical_leg:
            # merge the physical leg into leg_bra
            # the physical leg must be more major to align with new_sets_bra
            leg_bra = npc.LegPipe([fermion_leg, leg_bra], qconj=leg_bra.qconj)
            name_bra = f"(p.{name_bra})"

        # create the tensor
        # ----------------
        B = npc.zeros(
            [leg_bra, leg_ket],
            labels=[name_bra, name_ket],
            dtype=self.sometimes_matrix.dtype,
            qtotal=(self.qtotal,),
        )

        # Fill all charge blocks of the tensor
        # ------------------------------------
        qdict_bra = leg_bra.to_qdict()
        for q_ket, slice_ket in self.idx_ket.items():
            q_bra = q_ket + self.qtotal * qconj[0]
            if (q_bra,) in qdict_bra:
                slice_bra = qdict_bra[(q_bra,)]
                B[slice_bra, slice_ket] = self.det_always * _tensor_block(
                    self.sometimes_matrix,
                    self.new_sets_bra[slice_bra],
                    self.new_sets_ket[slice_ket],
                )

        return B.split_legs()  # splits the bra leg if `physical_leg`


#### High-level functions ####
#### -------------------- ####


def correlation_matrix(H: np.ndarray, N: int | None = None) -> tuple[np.ndarray, int]:
    r"""Ground-state correlation matrix of a mean-field Hamiltonian.

    Parameters
    ----------
    H:
        Real-space mean-field Hamiltonian.
    N:
        Number of occupied orbitals.

        If not specified (default), all orbitals with negative energy are filled.

    Returns
    -------
        C: :class:`np.ndarray`
            The correlation matrix, :math:`C_{ij} = \langle c_j^\dagger c_i\rangle`.
        N: :class:`int`
            The number of occupied orbitals.
    """

    e, v = eigh(H)
    if N is None:
        occupied = e < 0
        v = v[:, occupied]  # filter out occupied orbitals
        N = int(occupied.sum())
    else:
        v = v[:, :N]  # fill lowest N orbitals
    C = v @ HT(v)
    if np.iscomplexobj(C) and np.allclose(C.imag, 0.0, rtol=0, atol=1e-14):
        C = C.real  # eliminate zero imaginary parts
    return C, N


def C_to_MPS(
    C: np.ndarray,
    trunc_par: dict | StoppingCondition,
    *,
    diag_tol: float = _DIAG_TOL,
    ortho_center: int = None,
) -> networks.MPS:
    r"""MPS representation of a Slater determinant from its correlation matrix.

    Parameters
    ----------
    C:
        The correlation matrix, :math:`C_{ij} = \langle c_j^\dagger c_i\rangle`.
    trunc_par:
        Specifies which Schmidt states should be kept.

        Must be either a :class:`~temfpy.schmidt_sets.StoppingCondition` object
        or a dictionary with matching keys.

        Only specify the field `sectors` if you know what you are doing!!
    diag_tol:
        Largest allowed offdiagonal matrix element in diagonalised / SVD
        correlation submatrices before an error is raised.
    ortho_center:
        Orthogonality centre of the mixed canonical MPS.
        Midpoint of the chain by default.

    Returns
    -------
        The wave function as a TeNPy :class:`~tenpy.networks.mps.MPS` object.

    Note
    ----
    - If :attr:`trunc_par.svd_min` is not provided, the truncation threshold
      defaults to 1e-6.
    - If :attr:`trunc_par.degeneracy_tol` is not provided, the degeneracy tolerance
      defaults to 1e-12.
    """
    trunc_par = to_stopping_condition(trunc_par)

    L = len(C)
    assert C.shape == (L, L), f"Got non-square {C.shape} correlation matrix"

    # lists for accumulating the tensors and singular values
    tensors = [None] * L
    λs = [None] * (L + 1)

    # Main bipartition, in the middle if not otherwise specified
    ortho_center = ortho_center or L // 2
    logger.info("Central bond %d", ortho_center)
    Schmidt_center = SchmidtVectors.from_correlation_matrix(
        C, ortho_center, trunc_par=trunc_par, diag_tol=diag_tol
    )
    λs[ortho_center] = normalize_SV(Schmidt_center.schmidt_values, logger)

    # Right half of the chain
    # -----------------------
    Schmidt = Schmidt_center
    for i in range(ortho_center, L):
        logger.info("Site %d", i)
        Schmidt_new = SchmidtVectors.from_correlation_matrix(
            C, i + 1, trunc_par, which="R", diag_tol=diag_tol
        )
        λs[i + 1] = normalize_SV(Schmidt_new.schmidt_values, logger)

        # compute the tensor
        B = MPSTensorData.from_schmidt_vectors(Schmidt_new, Schmidt, "right")
        B = B.to_npc_array()
        tensors[i] = B

        logger.info(f"Tensor norm on site {i}: {npc.norm(B) / len(λs[i]) ** 0.5}")
        if logger.isEnabledFor(logging.DEBUG):
            # check accuracy of canonical form
            e = npc.tensordot(B, B.conj(), [["vR", "p"], ["vR*", "p*"]])
            e = e.to_ndarray()
            deviation = np.linalg.norm(e - np.eye(len(e))) / len(e)
            logger.debug(f"RMS RC deviation: {deviation}")

        Schmidt = Schmidt_new

    # Left half of the chain
    # ----------------------
    Schmidt = Schmidt_center
    for i in reversed(range(ortho_center)):
        logger.info("Site %d", i)
        Schmidt_new = SchmidtVectors.from_correlation_matrix(
            C, i, trunc_par, which="L", diag_tol=diag_tol
        )
        λs[i] = normalize_SV(Schmidt_new.schmidt_values, logger)

        # compute the tensor
        A = MPSTensorData.from_schmidt_vectors(Schmidt_new, Schmidt, "left")
        A = A.to_npc_array()
        tensors[i] = A

        logger.info(f"Tensor norm on site {i}: {npc.norm(A) / len(λs[i+1]) ** 0.5}")
        if logger.isEnabledFor(logging.DEBUG):
            # check accuracy of left canonical form
            e = npc.tensordot(A, A.conj(), [["vL", "p"], ["vL*", "p*"]])
            e = e.to_ndarray()
            deviation = np.linalg.norm(e - np.eye(len(e))) / len(e)
            logger.debug(f"RMS LC deviation: {deviation}")

        Schmidt = Schmidt_new

    form = ["A"] * ortho_center + ["B"] * (L - ortho_center)
    mps = networks.mps.MPS([fermion_site] * L, tensors, λs, form=form)

    return mps


def C_to_iMPS(
    C_short: np.ndarray,
    C_long: np.ndarray,
    trunc_par: dict | StoppingCondition,
    sites_per_cell: int,
    cut: int,
    *,
    diag_tol: float = _DIAG_TOL,
    unitary_tol: float = iMPS._UNITARY_TOL,
    schmidt_tol: float = iMPS._SCHMIDT_TOL,
) -> tuple[networks.MPS, iMPS.iMPSError]:
    r"""iMPS representation of a Slater determinant from correlation matrices.

    The two correlation matrices are expected to represent the ground states
    of a gapped, translation invariant Hamiltonian on two system sizes that
    differ by one repeating unit cell.

    The method is analogous to :func:`.iMPS.MPS_to_iMPS`, with two differences:

    - No explicit MPS tensors are computed for the environment of the iMPS unit
      cell. Instead, the Schmidt vector overlaps needed for gauge fixing are
      computed using the Slater determinant overlap formulas implemented in
      :class:`MPSTensorData`.
    - The rightmost tensor is computed directly using the right Schmidt vectors
      of the shorter chain. This means that no separate :class:`~.iMPS.iMPSError`\ s
      are returned for the right side.

    Parameters
    ----------
    C_short:
        The correlation matrix, :math:`C_{ij} = \langle c_j^\dagger c_i\rangle`,
        for the shorter chain.
    C_long:
        The correlation matrix for the longer chain.
    trunc_par:
        Specifies which Schmidt states should be kept.

        Must be either a :class:`~temfpy.schmidt_sets.StoppingCondition` object
        or a dictionary with matching keys.

        Only specify the field ``sectors`` if you know what you are doing!!
    sites_per_cell:
        Size of the iMPS unit cell.
    cut:
        First site of the repeating unit cell in ``C_long``.
    diag_tol:
        Largest allowed offdiagonal matrix element in diagonalised / SVD
        correlation submatrices before an error is raised.
    unitary_tol:
        Maximum deviation of the gauge rotation matrices from unitarity
        before a warning is raised.
    schmidt_tol:
        Maximum mixing of unequal Schmidt values by the gauge rotation matrices
        before a warning is raised.

    Returns
    -------
    iMPS: :class:`~tenpy.networks.mps.MPS`
        iMPS with unit cell size ``sites_per_cell``, constructed from the
        additional unit cell of ``mps_long``.
    validation_metric: :class:`~.iMPS.iMPSError`
        Errors introduced during the conversion.

    Note
    ----
    - If :attr:`trunc_par.svd_min` is not provided, the truncation threshold
      defaults to 1e-6.
    - If :attr:`trunc_par.degeneracy_tol` is not provided, the degeneracy tolerance
      defaults to 1e-12.
    """
    trunc_par = to_stopping_condition(trunc_par)

    L_short = len(C_short)
    err = f"Got non-square {C_short.shape} correlation matrix"
    assert C_short.shape == (L_short, L_short), err

    L_long = len(C_long)
    err = f"Got non-square {C_long.shape} correlation matrix"
    assert C_long.shape == (L_long, L_long), err

    assert L_short + sites_per_cell == L_long, (
        "The given two MPS must differ by one unit cell, got "
        f"{L_long} - {L_short} != {sites_per_cell}"
    )

    # lists for accumulating the tensors and singular values
    tensors = []
    λs = []

    # Reference bipartition in the two chains
    Schmidt_short = SchmidtVectors.from_correlation_matrix(
        C_short, cut, trunc_par=trunc_par, diag_tol=diag_tol
    )
    λs.append(normalize_SV(Schmidt_short.schmidt_values, logger))
    Schmidt_long = SchmidtVectors.from_correlation_matrix(
        C_long, cut, trunc_par=trunc_par, diag_tol=diag_tol
    )

    # Right canonical tensors
    Schmidt = Schmidt_long
    for i in range(sites_per_cell):
        logger.info("Site %d", i)
        if i == sites_per_cell - 1:
            Schmidt_new = Schmidt_short  # compare with right env of short chain
            λs.append(λs[0])  # ensure first and last λ identical
        else:
            Schmidt_new = SchmidtVectors.from_correlation_matrix(
                C_long, cut + i + 1, trunc_par, which="R", diag_tol=diag_tol
            )
            λs.append(normalize_SV(Schmidt_new.schmidt_values, logger))

        # compute the tensor
        B = MPSTensorData.from_schmidt_vectors(Schmidt_new, Schmidt, "right")
        B = B.to_npc_array()
        tensors.append(B)

        logger.info(f"Tensor norm on site {i}: {npc.norm(B) / len(λs[i]) ** 0.5}")
        if logger.isEnabledFor(logging.DEBUG):
            # check accuracy of canonical form
            e = npc.tensordot(B, B.conj(), [["vR", "p"], ["vR*", "p*"]])
            e = e.to_ndarray()
            deviation = np.linalg.norm(e - np.eye(len(e))) / len(e)
            logger.debug(f"RMS RC deviation: {deviation}")

        Schmidt = Schmidt_new

    # Gauge fix first tensor
    C = MPSTensorData.from_schmidt_vectors(Schmidt_short, Schmidt_long, "left")
    C = C.to_npc_array()
    C, left_unitary, left_schmidt = iMPS.basis_rotation(
        C,
        Schmidt_short.schmidt_values,
        Schmidt_long.schmidt_values,
        mode="left",
        unitary_tol=unitary_tol,
        schmidt_tol=schmidt_tol,
    )
    tensors[0] = npc.tensordot(C, tensors[0], axes=["vR", "vL"])

    iMPS_ = networks.MPS(
        [fermion_site] * sites_per_cell, tensors, λs, bc="infinite", form="B"
    )
    error = iMPS.iMPSError(left_unitary, left_schmidt, 0.0, 0.0)
    return iMPS_, error
