# Copyright (C) TeMFPy Developers, MIT license
r"""Tools for converting Pfaffian wave functions into matrix product states (MPS).

.. _nambu:

Representing Nambu correlation matrices
---------------------------------------

The input of the algorithm is the Nambu correlation matrix that contains
:math:`\langle cc \rangle` and :math:`\langle c^\dagger c \rangle` correlators
in addition to the standard :math:`\langle c^\dagger c \rangle` ones.
Such a correlation matrix may be specified in two different bases:

* In the standard **complex-fermion** basis, the Nambu correlation matrix
  is listed as 2×2 blocks ``C[2*i : 2*i+2, 2*j : 2*j+2]`` of 

  .. math::
  
        \begin{pmatrix}
            \langle c_j^\dagger c_i         \rangle & \langle c_j c_i         \rangle \\
            \langle c_j^\dagger c_i^\dagger \rangle & \langle c_j c_i^\dagger \rangle
        \end{pmatrix},

  where :math:`c_i` and :math:`c_i^\dagger` are the standard fermion
  annihilation and creation operators on site :math:`0\le i < L.`

* We can also use the **Majorana** basis, defined by

  .. math::

        \gamma_{2n} &= (c^\dagger_n + c_n) / \sqrt{2},\\
        \gamma_{2n+1} &= i(c^\dagger_n - c_n) / \sqrt{2}

  for all :math:`0\le n < L`. In this basis, the correlation matrix is specified
  as the matrix :math:`\langle \gamma_j\gamma_i \rangle.`
"""

from typing import Type
import warnings
import logging
from dataclasses import dataclass
from functools import partial

import numpy as np
from numpy.linalg import eigh, inv, svd
import tenpy.linalg.np_conserved as npc
from tenpy import networks
from pfapack.ctypes import pfaffian as cpf

from .utils import HT, block_svd, normalize_SV
from .testing import (
    assert_allclose,
    assert_array_less,
    check_schmidt_decomposition,
    _DIAG_TOL,
)
from .schmidt_utils import lowest_sums, StoppingCondition, to_stopping_condition
from . import iMPS

logger = logging.getLogger(__name__)

fermion_site = networks.site.FermionSite(conserve="parity")
"""Lattice site prototype for the parity-conserving fermion MPS."""
fermion_leg = fermion_site.leg
""":class:`~tenpy.linalg.charges.LegCharge` for the single-site Hilbert space
of the parity-conserving fermion MPS."""
chinfo = fermion_leg.chinfo
""":class:`~tenpy.linalg.charges.ChargeInfo` for fermion parity conservation."""

#### BASIS TRANSFORMATIONS BETWEEN COMPLEX FERMION AND MAJORANA BASIS ####
#### ---------------------------------------------------------------- ####


def vector_C2M(v: np.ndarray) -> np.ndarray:
    r"""Converts mode vectors from complex-fermion to Majorana basis.

    Parameters
    ----------
    v:
        A vector or a multidimensional array of vectors in the complex-fermion basis.

        In the latter case, the "site" dimension is expected as first
        (i.e. several vectors should be supplied as columns of a matrix).

        The complex-fermion basis should be listed in the order
        :math:`c^\dagger_1, c_1, \dots, c^\dagger_L, c_L`.

    Returns
    -------
        The given vectors in the Majorana basis.
    """
    n = v.shape[0]
    assert n % 2 == 0, "Got vector(s) of odd size (cannot be Nambu)"
    n = n // 2
    # unitary acts on second index of the reshaped v
    v = v.reshape(n, 2, *v.shape[1:])
    M = np.array([[1, 1], [1j, -1j]]) / 2**0.5
    v = np.einsum("xa...,ca->xc...", v, M)
    return v.reshape(2 * n, *v.shape[2:])


def vector_M2C(v: np.ndarray) -> np.ndarray:
    r"""Converts mode vectors from Majorana to complex-fermion basis.

    Parameters
    ----------
    v:
        A vector or a multidimensional array of vectors in the Majorana basis.

        In the latter case, the "site" dimension is expected as first
        (i.e. several vectors should be supplied as columns of a matrix).

    Returns
    -------
        The given vectors in the complex-fermion basis.

        The complex-fermion basis is listed in the order
        :math:`c^\dagger_1, c_1, \dots, c^\dagger_L, c_L`.
    """
    n = v.shape[0]
    assert n % 2 == 0, "Got vector(s) of odd size (cannot be Nambu)"
    n = n // 2
    # unitary acts on second index of the reshaped v
    v = v.reshape(n, 2, *v.shape[1:])
    M = np.array([[1, -1j], [1, 1j]]) / 2**0.5
    v = np.einsum("xa...,ca->xc...", v, M)
    return v.reshape(2 * n, *v.shape[2:])


def matrix_C2M(H: np.ndarray) -> np.ndarray:
    r"""Converts Hamiltonian or correlation matrices
    from complex-fermion to Majorana basis.

    Parameters
    ----------
    H:
        A matrix in the complex-fermion basis.

        See the module documentation for the expected layout.

    Returns
    -------
        The given matrix in the Majorana basis.
    """
    n, m = H.shape
    assert n % 2 == 0, "Got a matrix with odd side length (cannot be Nambu)"
    assert m % 2 == 0, "Got a matrix with odd side length (cannot be Nambu)"
    n //= 2
    m //= 2

    # isolate site and Nambu indices
    H = H.reshape(n, 2, m, 2)
    M = np.array([[1, 1], [1j, -1j]]) / 2**0.5
    H = np.einsum("xayb,ca,db->xcyd", H, M, M.conj())
    return H.reshape(2 * n, 2 * m)


def matrix_M2C(H: np.ndarray) -> np.ndarray:
    r"""Converts Hamiltonian or correlation matrices
    from Majorana to complex-fermion basis.

    Parameters
    ----------
    H:
        A matrix in the Majorana basis.

    Returns
    -------
        The given matrix in the complex-fermion basis.

        See the module documentation for the detailed layout.
    """
    n, m = H.shape
    assert n % 2 == 0, "Got a matrix with odd side length (cannot be Nambu)"
    assert m % 2 == 0, "Got a matrix with odd side length (cannot be Nambu)"
    n //= 2
    m //= 2

    # isolate site and Nambu indices
    H = H.reshape(n, 2, m, 2)
    M = np.array([[1, -1j], [1, 1j]]) / 2**0.5
    H = np.einsum("xayb,ca,db->xcyd", H, M, M.conj())
    return H.reshape(2 * n, 2 * m)


#### UTILITIES FOR NAMBU CORRELATION MATRICES ####
#### ---------------------------------------- ####
def assert_nambu(
    C: np.ndarray,
    basis: str = None,
    offset: float = None,
    name: str = "",
    rtol: float = 0,
    atol: float = 1e-10,
) -> np.ndarray:
    r"""Indicates if a matrix is not Nambu symmetric.
    
    In Majorana basis, Nambu symmetry requires that the matrix be imaginary
    and antisymmetric, except for a constant ``offset / 2`` on the diagonal.

    In complex-fermion basis, the matrix can be split into blocks
    by creation and annihilation operators, e.g.,

    .. math::

        C = \begin{pmatrix}
            \langle c_j^\dagger c_i         \rangle & \langle c_j c_i         \rangle \\
            \langle c_j^\dagger c_i^\dagger \rangle & \langle c_j c_i^\dagger \rangle
        \end{pmatrix} 
        =: \begin{pmatrix} C_{11} & C_{12} \\ C_{21} & C_{22} \end{pmatrix},

    which must satisfy 

    .. math::

        C_{11} + C_{22}^* &= \texttt{offset}\times\mathbb{I} \\
        C_{12} &= -C_{21}^*.

    Parameters
    ----------
    C:
        Nambu matrix to be checked.
    basis:
        Whether the matrix is supplied in
        Majorana (``"M"``) or complex-fermion (``"C"``) basis.

        If unspecified, only checks if the matrix is Hermitian.
    offset:
        Constant diagonal offset.

        Should be 0 for Hamiltonians and 1 for correlation matrices.
    name:
        Type of matrix (e.g. ``"Hamiltonian"``)
    rtol:
        Relative tolerance for matrix-element checks.
    atol:
        Absolute tolerance for matrix-element checks.

    Returns
    -------
        Regularised Nambu matrix.

        - Skew-Hermitian component is removed.
        - In the Majorana basis, small real-part deviations are deleted.
        - In the complex-fermion basis, small imaginary parts are pruned
          if the matrix is almost real.

    Raises
    ------
    AssertionError | testing.ComparisonWarning
        If ``C`` is not Hermitian or Nambu symmetric up to the given tolerance.
        
        Whether an exception or a warning is raised is determined by
        :data:`.testing.TEST_ACTION`.
    """
    # check shape
    n, m = C.shape
    assert n == m > 0, f"Got non-square {name}"
    assert n % 2 == 0, f"Got {name} with odd side length (cannot be Nambu)"
    n //= 2

    # Hermiticity
    tol = dict(atol=atol, rtol=rtol)
    assert_allclose(C, HT(C), **tol, err_msg=f"{name} is not Hermitian")
    C = (C + HT(C)) / 2

    # check Nambu symmetry of result
    if basis == "M":
        err = "Unexpected real parts in Majorana basis"
        real = np.eye(2 * n) * offset / 2
        assert_allclose(C.real, real, **tol, err_msg=err)
        C.real = real
    elif basis == "C":
        err = f"{name.capitalize()} is not Nambu symmetric"
        assert_allclose(
            C[::2, ::2], offset * np.eye(n) - C[1::2, 1::2].conj(), **tol, err_msg=err
        )
        assert_allclose(C[1::2, ::2], -C[::2, 1::2].conj(), **tol, err_msg=err)
        # often the correlation matrix in complex-fermion operators is real
        if np.allclose(C.imag, 0, **tol):
            C = C.real
    elif basis is not None:  # if None, we don't check for Nambu
        raise ValueError("Invalid `basis` " + repr(basis))

    return C


assert_nambu_hamiltonian = partial(assert_nambu, offset=0, name="Hamiltonian")
assert_nambu_hamiltonian.__doc__ = """
Indicates if a Hamiltonian is not Nambu symmetric.

See :func:`assert_nambu` for details. (Arguments ``offset`` and ``name`` are fixed.)"""

assert_nambu_correlation = partial(assert_nambu, offset=1, name="correlation matrix")
assert_nambu_correlation.__doc__ = """
Indicates if a correlation matrix is not Nambu symmetric.

See :func:`assert_nambu` for details. (Arguments ``offset`` and ``name`` are fixed.)"""


def correlation_matrix(
    H: np.ndarray, basis: str | None = None, *, rtol: float = 0, atol: float = 1e-10
) -> np.ndarray:
    r"""Ground-state correlation matrix of a mean-field Nambu Hamiltonian.

    Parameters
    ----------
    H:
        Real-space Nambu Hamiltonian.
         
        If ``basis`` is given, it must be in the basis
        indicated by its first character.
    basis:
        Basis used by the input and the output.

        If specified, it is a string of the form ``"X->Y"``,
        where the characters ``X`` and ``Y`` must be M or C:
        
        * ``X`` controls whether the Hamiltonian is given
          in the Majorana or the complex-fermion basis.
        * ``Y`` controls whether the correlation matrix is returned
          in the Majorana or the complex-fermion basis.

        If not specified, the output is returned in the same basis as the input,
        without testing for Nambu symmetry.
    rtol:
        Relative tolerance for eigenvalue and matrix-element checks.
    atol:
        Absolute tolerance for eigenvalue and matrix-element checks.

    
    Returns
    -------
        The correlation matrix C in the format indicated by
        the last character of ``mode``.

    Notes
    -----
    In Majorana mode ``"M->*"``, the Hamiltonian is expected as the
    array of coefficients of :math:`\gamma_i \gamma_j`.

    In complex-fermion mode ``"C->*"``, the Hamiltonian is expected as 
    2×2 blocks of coefficients of

    .. math::
    
        \begin{pmatrix}
            c_i^\dagger c_j & c_i^\dagger c_j^\dagger \\
            c_i         c_j & c_i         c_j^\dagger
        \end{pmatrix}.

    For the format of the correlation matrices, see :ref:`nambu`.
    """
    # check argument `basis`
    basis_error = f"Invalid basis spec {basis!r}, should be of form '[MC]->[MC]'"
    assert basis in [None, "M->M", "M->C", "C->M", "C->C"], basis_error

    tol = dict(rtol=rtol, atol=atol)

    # check Hamiltonian
    H = assert_nambu_hamiltonian(H, None if basis is None else basis[0], **tol)
    n = len(H) // 2

    # diagonalise
    e, v = eigh(H)

    # check that spectrum is symmetric
    assert_allclose(e + e[::-1], 0, **tol)

    # check that there are no almost-zero eigenvalues
    if np.any(abs(e) < atol):
        raise RuntimeError(
            "Some energy eigenvalues are zero. You need to construct\n"
            "your own correlation matrix!\n"
            f"Middle 10 eigenvalues:\n{e[n - 5 : n + 5, None]}"
        )

    # Filter negative eigenvalues
    assert_array_less(e[:n], 0, "Lower half of eigenvalues is not all negative")
    v = v[:, :n]

    # basis conversions if needed
    if basis == "C->M":
        v = vector_C2M(v)
    elif basis == "M->C":
        v = vector_M2C(v)

    C = v @ HT(v)

    # check Nambu symmetry of result
    C = assert_nambu_correlation(C, None if basis is None else basis[3], **tol)
    return C


def parity(V: np.ndarray, *, tol: float = 1e-12) -> int:
    r"""Fermion parity of a Boguliubov vacuum.

    The calculation is based on the Bloch-Messiah decomposition.
    If the Boguliubov operators :math:`\gamma` that define the vacuum
    are related to the standard fermion operators :math:`c` by

    .. math::

        (\gamma^\dagger, \gamma) = (c^\dagger, c) \begin{pmatrix}
            U & V^* \\ V & U^*
        \end{pmatrix},

    the singular values of the matrix :math:`V` (and also :math:`V^*`) are
    (in decreasing order)
    :math:`1, \dots, 1, \sigma_1, \sigma_1, \dots, \sigma_n, \sigma_n, 0, \dots, 0`.
    The parity of the vacuum is that of the number of completely filled modes
    (i.e., singular values 1).

    Parameters
    ----------
        V:
            submatrix of Nambu mode unitary that maps :math:`c` to
            :math:`\gamma^\dagger` (or :math:`c^\dagger` to :math:`\gamma`).
        tol:
            tolerance for considering two numbers equal
            (only needed for edge cases of 1 and 2 modes)

    Returns
    -------
        parity as 0 (even) or 1 (odd)
    """
    if len(V) == 0:
        return 0
    elif len(V) == 1:
        V = V.item()
        if np.isclose(V, 0.0, rtol=0, atol=tol):
            return 0
        elif np.isclose(abs(V), 1.0, rtol=0, atol=tol):
            return 1
        else:
            raise RuntimeError("Invalid 1x1 V")
    else:
        s = svd(V, compute_uv=False)
        if len(V) > 2:
            # instead of isolating the precisely 1 singular values,
            # note that SVs between 0 and 1 come in pairs,
            # so it is enough to find the largest gap between them
            # the parity of SVs above that gap is the same as the 1s
            n = np.argmax(-np.diff(s))
            return (n + 1) % 2  # NB arrays are zero indexed
        else:
            # usual autodetection fails (there's only 1 difference)
            # there is either a 1 and a 0 SV (-> parity odd)
            # or two equal ones (-> parity even)
            if np.allclose(s, [1.0, 0.0], rtol=0, atol=tol):
                return 1
            elif np.isclose(s[0], s[1], rtol=0, atol=tol):
                return 0
            else:
                raise ValueError("Invalid 2x2 V")


#### SCHMIDT ORBITALS ####
#### ---------------- ####
@dataclass(frozen=True)
class SchmidtModes:
    """Boguliubov excitations that generate the Schmidt vectors
    of a Nambu mean-field state.
    """

    nL: int
    """Size of the left half of the system."""
    nR: int
    """Size of the right half of the system."""
    e: np.ndarray
    r"""array (:attr:`n_entangled`,) --
    Entangled eigenvalues :math:`0 < \lambda\le 1/2` of the
    diagonal blocks of the correlation matrix.
    """
    vL: np.ndarray | None
    r"""array (2\ :attr:`nL`, 2\ :attr:`nL`) --
    Eigenvectors of the left-left block of the correlation matrix
    in the complex-fermion basis, if computed.

    The eigenvectors are the columns of the matrix in the order of eigenvalues

    - :math:`0\le \lambda\le 1/2` in increasing order
    - :math:`1\ge \lambda\ge 1/2` in decreasing order,

    i.e. the entangled modes are ``vL[:, nL-n_entangled : nL]`` (eigenvalues: :attr:`e`)
    and ``vL[:, -n_entangled:]`` (eigenvalues: ``1-e``).
    
    The eigenvectors obey the Nambu symmetry:

        vL[::2, nL:] == vL[1::2, :nL].conj()
        vL[1::2, nL:] == vL[::2, :nL].conj()
    """
    vR: np.ndarray | None
    r"""array (2\ :attr:`nR`, 2\ :attr:`nR`) --
    Eigenvectors of the right-right block of the correlation matrix
    in the complex-fermion basis, if computed.

    The eigenvectors are the columns of the matrix in the order of eigenvalues

    - :math:`1/2\ge \lambda\ge 0` in decreasing order
    - :math:`1/2\le \lambda\le 1` in increasing order,

    i.e. the entangled modes are ``vR[:, :n_entangled]`` (eigenvalues: ``e[::-1]``)
    and ``vR[:, nR : nR+n_entangled]`` (eigenvalues: ``1-e[::-1]``).

    They obey the Nambu symmetry::

        vR[::2, nR:] == vR[1::2, :nR].conj()
        vR[1::2, nR:] == vR[::2, :nR].conj()

    If both :attr:`vL` and :attr:`vR` are known, their entangled components
    the left-right block of the Hamiltonian::
    
        S = sqrt(e * (1-e))
        S = concatenate((S, -S))
        vL_entangled.T.conj() @ C[:2*nL, 2*nL:] @ vR_entangled[:, ::-1] == diag(S)

    However, to better handle fermion anticommutation, the sign of :attr:`vR` is
    reversed if the left Boguliubov vacuum is parity odd.
    """
    pL: int | None
    """0 or 1 --
    Parity of the Boguliubov vacuum that annihilates the operators
    defined by eigenvectors ``vL[:, nL:]``, if computed."""
    pR: int | None
    """0 or 1 --
    Parity of the Boguliubov vacuum that annihilates the operators
    defined by eigenvectors ``vR[:, nR:]``, if computed."""

    def parity(self, which: str = "T") -> int | None:
        """Parity of the Boguliubov vacuum on the specified half of the system.

        Parameters
        ----------
        which:
            Either "L" for left or "R" for right side
            or "T" (default) for the overall parity of the state.

        Returns
        -------
            :attr:`pL`, :attr:`pR`, or the overall parity, depending on ``which``
            or :obj:`None` if the requested parity is unknown.
        """
        which_ = which[0].upper()
        if which_ == "L":
            return self.pL
        elif which_ == "R":
            return self.pR
        elif which_ == "T":
            if (self.pL is None) or (self.pR is None):
                return None
            else:
                return (self.pL + self.pR) % 2
        else:
            raise ValueError("`which` must start with L, R, or T, got " + repr(which))

    def __post_init__(self):
        if self.vL is not None:
            assert self.pL is not None, "`pL` must be specified with `vL`"
            assert 2 * self.nL == len(self.vL), "`nL` must match the size of `vL`"
        if self.vR is not None:
            assert self.pR is not None, "`pR` must be specified with `vR`"
            assert 2 * self.nR == len(self.vR), "`nR` must match the size of `vR`"
        err = "Must specify at least one of `vL`, `vR`"
        assert (self.vL is not None) or (self.vR is not None), err

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

    @property
    def vL_entangled(self) -> np.ndarray | None:
        """Entangled left Schmidt mode orbitals, if computed."""
        ix = np.arange(self.nL - self.n_entangled, self.nL)
        ix = np.concatenate((ix, ix + self.nL))
        return None if self.vL is None else self.vL[:, ix]

    @property
    def vR_entangled(self) -> np.ndarray | None:
        """Entangled right Schmidt mode orbitals, if computed."""
        ix = np.arange(self.n_entangled)
        ix = np.concatenate((ix, ix + self.nR))
        return None if self.vR is None else self.vR[:, ix]

    def mode_vectors(self, which: str, entangled: bool = False) -> np.ndarray:
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
            if not entangled:
                e = np.concatenate((np.zeros(self.nL - self.n_entangled), e))
        elif which_ == "R":
            if self.vR is None:
                return None
            e = self.e[::-1]
            if not entangled:
                e = np.concatenate((e, np.zeros(self.nR - self.n_entangled)))
        else:
            raise ValueError("`which` must start with L or R, got " + repr(which))
        return np.concatenate((e, 1 - e))

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
        SV *= -1 if self.pL == 1 else +1  # anticommutation sign
        return np.concatenate((SV, -SV))  # Nambu sign

    @classmethod
    def from_correlation_matrix(
        cls: Type["SchmidtModes"],
        C: np.ndarray,
        x: int,
        trunc_par: dict | StoppingCondition,
        *,
        basis: str,
        which: str = "LR",
        diag_tol: float = _DIAG_TOL,
        total_parity: int | None = None,
    ) -> "SchmidtModes":
        r"""Computes the :class:`~SchmidtModes` of a Nambu mean-field state
        with correlation matrix ``C`` for an entanglement cut between
        sites ``x-1`` and ``x`` (zero-indexed).

        We do this by diagonalising the left-left and right-right blocks of ``C``.
        The eigenvectors give the Schmidt modes, the eigenvalues their
        relative weight in the entangled mode. We only treat modes with
        eigenvalues away from 0 or 1 by at least `cutoff` as entangled.

        We enforce Nambu symmetry by redefining the eigenvectors for
        eigenvalues above 1/2 in terms of the other half.

        If both left and right modes are computed, the entangled ones are
        paired up by ensuring that the eigenvalues sum to 1, and that they
        SVD-diagonalise the top right block C_LR and their signs are fixed
        so as to handle anticommutation signs optimally.

        Parameters
        ----------
        C:
            The correlation matrix.
        x:
            Position of the entanglement cut.
        trunc_par:
            Which Schmidt modes should be kept as entangled.

            Must be either a :class:`~temfpy.schmidt_sets.StoppingCondition` object
            or a dictionary with matching keys.
        basis:
            whether the correlation matrix is supplied in
            Majorana (``"M"``, default) or complex-fermion (``"C"``) basis.
        which:
            Whether to return Left and/or Right Schmidt modes.
            Must be a combination of ``"L"``, ``"R"``.
        diag_tol:
            If ``which=="LR"``, largest allowed offdiagonal matrix element in
            diagonalised / SVD correlation submatrices before an error is raised.
        total_parity:
            Used to infer the parity of the leading Schmidt state on both sides
            if Schmidt modes are only computed on one side.

        Note
        ----
        - If :attr:`trunc_par.svd_min` is not provided, a default of 1e-6
          (i.e., a truncation threshold of 1e-12) is used.
        - If :attr:`trunc_par.degeneracy_tol` is not provided, the degeneracy tolerance
          defaults to 1e-12.
        """
        trunc_par = to_stopping_condition(trunc_par)
        cutoff = trunc_par.svd_min**2  # eigenvalues -> squared Schmidt values
        deg_tol = trunc_par.degeneracy_tol

        # Internally, we always use the Majorana basis for diagonalisation
        if basis == "C":
            C = matrix_C2M(C)
        elif basis != "M":
            raise ValueError(f"Argument `basis` must be 'M' or 'C', got {basis!r}")
        C = assert_nambu_correlation(C, "M", atol=cutoff)

        L = len(C) // 2
        assert 0 <= x <= L, f"Invalid entanglement cut {x}, must be between 0 and {L}"
        y = L - x  # size of right half

        which = which.upper()
        err = "`which` must specify at least one of (L)eft or (R)ight"
        assert ("L" in which) or ("R" in which), err

        def diag_nambu(c):
            """Args:
                c: diagonal submatrix of Majorana correlation matrix

            Returns: e, v, ke, kh
                e: eigenvalues of c in ascending order
                v: corresponding eigenvectors
                    for e[i] away from 1/2, v[:,i] = v[:,-i-1].conj()
                    for e[i] approximately 1/2, v[i] is purely real
                    (both are possible in Majorana basis)
                ke: half number of entangled modes
                    corresponding modes at e[n//2-ke : n//2+ke]
                kh: half number of approximately 1/2 eigenvalues
                    corresponding modes at e[n//2-kh : n//2+kh]
            """
            if c is None:  # just need dummies for tuple unpacking
                return (None,) * 4

            n = len(c) // 2

            if n == 0:
                e = np.zeros((0,), float)
                v = np.zeros((0, 0), c.dtype)
                return e, v, 0, 0

            e, v = eigh(c)

            # clip corr. matrix spectrum to [0,1]
            err = "Invalid correlation matrix eigenvalues (should be between 0 and 1)"
            assert_array_less(-deg_tol, e, err_msg=err)
            e[e < 0] = 0
            assert_array_less(e, 1 + deg_tol, err_msg=err)
            e[e > 1] = 1

            # check symmetry of spectrum
            err = "Eigenvalues break Nambu symmetry"
            assert_allclose(e, 1 - e[::-1], rtol=0, atol=deg_tol, err_msg=err)

            # isolate 1/2 eigenvalues
            x0, x1 = np.searchsorted(e, [0.5 - deg_tol, 0.5 + deg_tol])
            kh = x1 - n
            assert x0 == n - kh, "1/2 eigenvalues asymmetrical in spectrum"

            # make 1/2 modes real
            if kh != 0 and np.iscomplexobj(v):
                w = np.column_stack((v[:, x0:x1].real, v[:, x0:x1].imag))
                w, s, _ = svd(w)

                s_exp = [1] * (2 * kh) + [0] * (2 * kh)
                err = "1/2 eigenvectors cannot be made real"
                assert_allclose(s, s_exp, rtol=0, atol=diag_tol, err_msg=err)

                v[:, x0:x1] = w[:, : 2 * kh]

            # isolate entangled modes
            x0, x1 = np.searchsorted(e, [cutoff, 1 - cutoff])
            ke = x1 - n
            assert x0 == n - ke, "Entangled modes asymmetrical in spectrum"

            return e, v, ke, kh

        eL, vL, keL, khL = diag_nambu(C[: 2 * x, : 2 * x] if "L" in which else None)
        eR, vR, keR, khR = diag_nambu(C[2 * x :, 2 * x :] if "R" in which else None)

        if eL is None:
            if eR is None:
                raise RuntimeError()  # should error earlier
            else:
                k = keR
                kh = khR
                e = eR[y - k : y]
        else:
            if eR is None:
                k = keL
                kh = khL
                e = eL[x - k : x]
            else:
                # both "L" and "R" were done, need consistency checks
                assert keL == keR, "Unequal number of entangled modes"
                k = keL
                assert khL == khR, "Unequal number of 1/2 modes"
                kh = khL
                e = eL[x - k : x]

                err = "Eigenvalues of C_LL and C_RR do not match"
                assert_allclose(e, eR[y - k : y], rtol=0, atol=deg_tol, err_msg=err)

                # SVD 0 < λ < 1/2 modes
                CLR = C[: 2 * x, 2 * x :]
                vLE = vL[:, x - k : x - kh]
                vRE = vR[:, y + kh : y + k][:, ::-1]
                block_svd(CLR, vLE, vRE, eL[x - k : x - kh], deg_tol, diag_tol)

                # SVD 1/2 modes
                # keep both sides real, even though C_LR is pure imaginary
                # -> SVD the imaginary part, so (w_Li)T C_LR w_Rj = i S δ_ij
                ixL = slice(x - kh, x + kh)
                ixR = slice(y - kh, y + kh)
                s_block = vL[:, ixL].real.T @ CLR.imag @ vR[:, ixR].real
                U, _, Vh = svd(s_block)
                vL[:, ixL] = vL[:, ixL] @ U
                vR[:, ixR] = vR[:, ixR] @ Vh.T

        logger.info(f"{k} Schmidt modes found")

        # Fix Nambu symmetry, convert to complex-fermion basis, compute parity
        def nambu(v, kh, LR):
            x = len(v) // 2
            if LR == "L":
                # Turn 1/2 eigenvectors from real to conjugate (Nambu) pairs
                v[:, x - kh : x] = (v[:, x - kh : x] + 1j * v[:, x : x + kh]) / 2**0.5
                # Replace upper half with (Nambu) conjugates of the lower
                v[:, x:] = v[:, :x].conj()
            elif LR == "R":
                v[:, x : x + kh] = (v[:, x - kh : x] - 1j * v[:, x : x + kh]) / 2**0.5
                # Replace lower half with (Nambu) conjugates of the upper
                v[:, :x] = v[:, x:].conj()
            else:
                raise RuntimeError()

            v = vector_M2C(v)
            p = parity(v[1::2, :x])
            return v, p

        if "L" in which:
            vL, pL = nambu(vL, kh, "L")
            logger.info(f"Parity of left Boguliubov vacuum: {pL}")
            if "R" not in which and total_parity is not None:
                pR = (total_parity + pL) % 2
                logger.info(f"Inferred parity of right Boguliubov vacuum: {pR}")

        if "R" in which:
            vR, pR = nambu(vR, kh, "R")
            logger.info(f"Parity of right Boguliubov vacuum: {pR}")
            if "L" not in which and total_parity is not None:
                pL = (total_parity + pR) % 2
                logger.info(f"Inferred parity of left Boguliubov vacuum: {pL}")

        # if parity of left vacuum is odd, need to flip the sign of all
        # right modes to compensate for commuting through it
        if ("L" in which) and ("R" in which) and (pL == 1):
            vR *= -1

        modes = cls(e=e, vL=vL, vR=vR, pL=pL, pR=pR, nL=x, nR=y)
        check_schmidt_decomposition(modes, matrix_M2C(C), diag_tol)
        return modes

    @property
    def e_ratio(self) -> np.ndarray:
        r""":math:`\log((1-\lambda)/\lambda` for all eigenvalues in :attr:`e`."""
        return np.log((1 - self.e) / self.e)

    def embed_subsets(
        modes, sets: np.ndarray
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        r"""Given an array of :math:`\gamma^\dagger` excitations listed in order
        of the left singular vectors, generates the set of excitations on both sides.

        Parameters
        ----------
        sets: bool :class:`~np.ndarray` (n, :attr:`n_entangled`)
            Array of occupation numbers.

            Each row specifies one Schmidt state by listing whether each
            :math:`\gamma^\dagger` excitation is "occupied".

            The order of the excitations is consistent with :attr:`vL`.

        Returns
        -------
        left_sets: bool :class:`~np.ndarray` (n, :attr:`n_entangled`) | :obj:`None`
            Occupation of excitations in the order of :attr:`vL`.

            Returned if :attr:`vL` is not :obj:`None`.
        right_sets: bool :class:`~np.ndarray` (n, :attr:`n_entangled`) | :obj:`None`
            Occupation of excitations in the order of :attr:`vR`.

            Returned if :attr:`vR` is not :obj:`None`.
        """

        # `sets` is ordered correctly for left modes, must be reversed for right
        left_sets = sets if modes.vL is not None else None
        right_sets = sets[:, ::-1] if modes.vR is not None else None

        return left_sets, right_sets

    def schmidt_values(self, sets: np.ndarray) -> np.ndarray:
        r"""Schmidt values of the Schmidt vectors with given excitation numbers.

        Parameters
        ----------
        sets: bool :class:`~np.ndarray` (n, :attr:`n_entangled`)
            Array of occupation numbers.

            Each row specifies one Schmidt state by listing whether each
            :math:`\gamma^\dagger` excitation is "occupied".

            The order of the excitations is consistent with :attr:`vL`.

        Returns
        -------
        λ: :class:`np.ndarray` (n,)
            Schmidt values corresponding to the input Schmidt vectors.
        """
        return np.where(sets, self.e, 1 - self.e).prod(axis=1) ** 0.5


#### SCHMIDT VECTORS ####
#### --------------- ####


def _parity_n_argsort(x: np.ndarray) -> np.ndarray:
    """Sorts an array of integers first by parity, then value.

    Returns
    -------
    idx: The indices that stable sort the flattened array.
    dict_n: Dict mapping values to slices in the sorted array.
    dict_p: Dict mapping parities to slices in the sorted array."""
    x = x.ravel()
    idx = np.lexsort((np.arange(len(x)), x, x % 2))
    x = x[idx]
    return idx, _bunched_slices(x), _bunched_slices(x % 2)


def _bunched_slices(x: np.ndarray) -> dict[int, slice]:
    """Given a sorted array of integers, returns a dictionary that maps unique
    elements to the corresponding slice of the array."""
    (idx,) = np.nonzero(x[1:] != x[:-1])
    idx = np.concatenate(([0], idx + 1, [len(x)]))
    return {x[idx[i]]: slice(idx[i], idx[i + 1]) for i in range(len(idx) - 1)}


@dataclass(frozen=True)
class SchmidtVectors:
    """Schmidt vectors of a Nambu mean-field state."""

    modes: SchmidtModes
    """The Boguliubov excitations underlying the Schmidt vectors."""
    left_sets: np.ndarray | None
    r"""bool (:attr:`n_schmidt`, :attr:`n_entangled`) --
    Left Schmidt vectors.

    Each row contains the occupation of all entangled :math:`\gamma_L^\dagger`
    excitations, described by ``vL[:, :n_entangled]``, in one left Schmidt vector,
    if :attr:`modes.vL` is not :obj:`None`."""
    right_sets: np.ndarray | None
    r"""bool (:attr:`n_schmidt`, :attr:`n_entangled`) --
    Right Schmidt vectors.

    Each row contains the occupation of all entangled :math:`\gamma_R^\dagger`
    excitations, described by ``vR[:, nR-n_entangled:nR]``,
    in one right Schmidt vector, if :attr:`vR` is not :obj:`None`.

    If both  :attr:`left_sets` and :attr:`right_sets` are computed,
    they are related by ``left_sets = right_sets[:, ::-1]``."""
    schmidt_values: np.ndarray
    r"""array (:attr:`n_schmidt`,):
    Schmidt values :math:`\lambda_\alpha` corresponding to each Schmidt vector.
    
    Collated by number of Boguliubov excitations (cf. :attr:`idx_n`);
    sorted in decreasing order within each excitation-number sector."""
    idx_n: dict[int, slice]
    r"""Dictionary mapping the number of :math:`\gamma^\dagger` excitations
    to slice of sets/singular values corresponding to that excitation number.

    That is, each row in ``left_sets[idx_n[n]]`` contains n ``True`` entries.

    Slices follow each other in the order 0, 2,..., 1, 3,...
    """
    idx_parity: dict[int, slice]
    r"""Dictionary mapping the parity (0 or 1) of :math:`\gamma^\dagger` excitations
    to slice of sets/singular values corresponding to that excitation number.

    That is, each row in ``left_sets[idx_parity[n]]`` contains an even (odd)
    number of ``True`` entries if n=0 (1).
    """

    @property
    def n_schmidt(self) -> int:
        """Number of Schmidt vectors."""
        return self.schmidt_values.size

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

    def mode_vectors(self, which: str, entangled: bool = False) -> np.ndarray:
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

    @property
    def pL(self) -> int | None:
        """Parity of left Boguliubov vacuum :attr:`~SchmidtModes.pL`."""
        return self.modes.pL

    @property
    def pR(self) -> int | None:
        """Parity of right Boguliubov vacuum :attr:`~SchmidtModes.pR`."""
        return self.modes.pR

    def parity(self, which: str = "T") -> int | None:
        """Parity of the Boguliubov vacuum on the specified half of the system.

        Parameters
        ----------
        which:
            Either "L" for left or "R" for right side
            or "T" (default) for the overall parity of the state.

        Returns
        -------
            :attr:`~SchmidtModes.pL`, :attr:`~SchmidtModes.pR`,
            or the overall parity, depending on ``which``.
        """
        return self.modes.parity(which)

    def sets(self, which: str):
        """Returns the sets of Boguliubov excitations on the specified side.

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
            raise ValueError("`which` must start with L or R, got " + repr(which))

    @classmethod
    def from_schmidt_modes(
        cls: Type["SchmidtVectors"],
        modes: SchmidtModes,
        trunc_par: dict | StoppingCondition,
    ) -> "SchmidtVectors":
        r"""Generates the most significant :class:`SchmidtVectors` from an
        instance of :class:`SchmidtModes`.

        Parameters
        ----------
        modes:
            The Schmidt modes.
        trunc_par:
            Specifies which Schmidt states should be kept.

            Must be either a :class:`~temfpy.schmidt_sets.StoppingCondition` object
            or a dictionary with matching keys.

            The filtering function `is_sector` is applied to the number of
            :math:`\gamma^\dagger` excitations.
        """
        trunc_par = to_stopping_condition(trunc_par)

        # The ratio of excited Schmidt states' weight relative to vacua is the product
        # of the weights of each included γ† relative to the corresponding γ
        # Find the sets with the largest products of these
        _, sets = lowest_sums(modes.e_ratio / 2, trunc_par)
        if len(sets) == 0:
            err = "No Schmidt vectors left after filtering by `trunc_par.sectors`!"
            raise ValueError(err)

        # collate `sets` by number and parity of excitations
        exc = sets.sum(axis=1)
        idx, idx_n, idx_parity = _parity_n_argsort(exc)
        sets = sets[idx]

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
            idx_n=idx_n,
            idx_parity=idx_parity,
        )

    @classmethod
    def from_correlation_matrix(
        cls: Type["SchmidtVectors"],
        C: np.ndarray,
        x: int,
        trunc_par: dict | StoppingCondition,
        *,
        basis: str,
        which: str = "LR",
        diag_tol: float = _DIAG_TOL,
        total_parity: int | None = None,
    ) -> "SchmidtVectors":
        """Computes the most significant :class:`SchmidtVectors` of a Nambu
        mean-field state with correlation matrix C for an entanglement cut
        between sites ``x-1`` and ``x`` (zero-indexed).

        Calls :meth:`SchmidtModes.from_correlation_matrix` (see there for
        details of the parameters) and :meth:`from_schmidt_modes`.
        """
        which = which.upper()
        trunc_par = to_stopping_condition(trunc_par)

        modes = SchmidtModes.from_correlation_matrix(
            C,
            x,
            trunc_par,
            basis=basis,
            which=which,
            diag_tol=diag_tol,
            total_parity=total_parity,
        )
        vectors = cls.from_schmidt_modes(modes, trunc_par)
        return vectors


#### CONSTRUCTING MPS TENSORS ####
#### ------------------------ ####


#### Overlap of Boguliubov states ####


def _pfaffian_matrix(
    V1, V2, sets1, sets2, *, mode, tolerance=1e-8, min_SV=1e-6
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, int]:
    # NB this docstring doesn't go into sphinx, don't bother beautifying it
    """Computes the Pfaffian entries for all bogulon excitations that
    appear in at least one Schmidt state.

    First, the Schmidt modes of the "ket" basis, V2, are expressed
    in terms of the "bra" basis, V1, as Vr = inv(V1) @ V2.
    We also check that the resulting U is parity even and invertible.

    Then, if
        Vr = (U V*)
             (V U*)
    the vacuum of the "ket" basis is
        |vac_B> = exp(1/2 M_ij a†_i a†_j) |vac_A>
    where the a† are creation operators in the "bra" basis and
        M = V* @ inv(U*).

    Now we need overlaps of the type
        <vac_A| a_ik ... a_i1 b†_j1 ... b†_jl |vac_B>,
    which can be computed as the Pfaffian of a diagonal submatrix of
    the matrix N generated by this function.

    N has a row and a column for each active a and b† (i.e., those that
    are excited in at least one Schmidt vector). They are listed in
    reverse order of b†, then regular order of a.
    The entries of each block are:
        (b†,b†): N_BB = V^T (MV - U) = -V^T inv(U†) = inv(U*) V
        (a,a): N_AA = M
        (a,b†): N_AB = MV - U = -inv(U†),
        (b†,a): N_BA = -N_AB^T = inv(U*)
    Each of these formulas are restricted to the active modes.

    Since |vac_B> as defined above is not normalised, we also compute
    the overlap of the normalised vacua from the Onishi formula.

    Parameters
    ----------
        V1, V2:
            Schmidt-Nambu orbital vectors in the bra and ket bases
        sets1, sets2:
            excitations of Schmidt-Nambu orbitals in the bra and ket Schmidt vectors
        mode:
            if "left" ("right"), the active a,b† modes are taken from the
            end (start) of the relevant segment of V1,V2
        tolerance:
            numerical tolerance applied for various comparisons
        min_SV:
            smallest acceptable singular value of U before it is considered singular

    Returns
    -------
        norm:
            overlap of the normalised <vac_A|vac_B>
        N:
            the Pfaffian matrix described above.
            new_sets* index the rows/columns of this matrix.
        new_sets1:
            sets1 with any inactive modes removed and prepended
            with False for the ket modes.
        new_sets2:
            sets2 with any inactive modes removed, reversed,
            and appended with False for the bra modes.
        split:
            index of first a mode

    Notes
    -----
        The two vacua are expected to have an overlap. In particular,
        their parities must match.
        There is no attempt to ensure that a distinguished "physical leg"
        be retained. This has to be handled by the caller.
    """
    # sanity checks
    n, m = V1.shape
    assert n == m > 0
    assert n % 2 == 0
    L = n // 2
    assert V2.shape == (n, m)
    # mapping between bra and ket basis
    Vr = HT(V1) @ V2  # use that V1 is unitary

    # check that it has expected Nambu structure
    # NB operators and daggers are now fully blocked along both axes
    nambu_err = "Nambu symmetry violated"
    assert_allclose(
        Vr[:L, :L].conj(), Vr[L:, L:], rtol=0, atol=tolerance, err_msg=nambu_err
    )
    assert_allclose(
        Vr[:L, L:].conj(), Vr[L:, :L], rtol=0, atol=tolerance, err_msg=nambu_err
    )

    # check that vacua have finite overlap (i.e. U has all positive SVs)
    s = svd(Vr[:L, :L], compute_uv=False)
    logger.info(f"Boguliubov vacuum overlap: {s.prod():.3e}")
    logger.debug(f"Range of singular values: ({s.min():.3e}, {s.max():.3e})")
    assert_array_less(
        min_SV, s, err_msg="Boguliubov vacua do not overlap (U nearly singular)"
    )
    # compute normalising factor from Onishi formula
    norm = s.prod() ** 0.5

    def prune(sets, reverse) -> tuple[np.ndarray, np.ndarray]:
        """Eliminates all-false columns of `sets` and returns the
        list of indices to prune other matrices. If `reverse`, also
        flips the order of columns and indices."""
        idx = np.any(sets, axis=0)
        (idx,) = np.nonzero(idx)
        if reverse:
            idx = idx[::-1]
        return sets[:, idx], idx

    active1 = sets1.shape[1]  # that is, active modes before pruning
    active2 = sets2.shape[1]
    sets1, idx1 = prune(sets1, False)  # a modes
    sets2, idx2 = prune(sets2, True)  # b† modes

    # embed idx into the full arrays
    if mode == "left":  # active modes at end
        idx1 += L - active1
        idx2 += L - active2
    elif mode != "right":  # right -> active modes at front -> nothing to do
        raise ValueError('`mode` must be "left" or "right"')

    # will need the inverse of U* or U† several times
    Uxinv = inv(Vr[L:, L:])

    # AA block: M = V* inv(U*)
    AA = Vr[idx1, L:] @ Uxinv[:, idx1]
    # BA block: inv(U*)
    BA = Uxinv[np.ix_(idx2, idx1)]
    # BB block: inv(U*) V
    BB = Uxinv[idx2] @ Vr[L:, idx2]

    # check antisymmetry
    assert_allclose(AA, -AA.T, rtol=0, atol=tolerance, err_msg=nambu_err)
    AA = (AA - AA.T) / 2
    assert_allclose(BB, -BB.T, rtol=0, atol=tolerance, err_msg=nambu_err)
    BB = (BB - BB.T) / 2

    # assemble Pfaffian matrix
    N = np.block([[BB, BA], [-BA.T, AA]])

    # expand sets1, sets2
    new_sets1 = np.concatenate(
        (np.zeros((sets1.shape[0], sets2.shape[1]), dtype=bool), sets1), axis=1
    )
    new_sets2 = np.concatenate(
        (sets2, np.zeros((sets2.shape[0], sets1.shape[1]), dtype=bool)), axis=1
    )

    return norm, N, new_sets1, new_sets2, sets2.shape[1]


def _many_pfaffian(matrices: np.ndarray, **kwargs) -> np.ndarray:
    """Computes a bunch of Pfaffians by iterating through them with
    :func:`pfapack.ctypes.pfaffian`.

    If the input has shape ``(...,N,N)``, returns an array of shape ``(...)``.
    """
    err = f"Expected square matrices in the last two axes, got {matrices.shape}"
    assert (matrices.ndim >= 2) and (matrices.shape[-1] == matrices.shape[-2]), err
    shape = matrices.shape[:-2]
    matrices = matrices.reshape(np.prod(shape), *matrices.shape[-2:])
    with warnings.catch_warnings(category=np.exceptions.ComplexWarning):
        warnings.simplefilter("ignore")
        pf = np.asarray([cpf(A, **kwargs) for A in matrices])
    return pf.reshape(shape)


def _tensor_block(N, new_sets1, new_sets2) -> np.ndarray:
    """Computes a fixed-bogulon-number block of the MPS tensor.

    Parameters
    ----------
        N:
            the Pfaffian matrix generated by `_pfaffian_matrix`
        new_sets1, new_sets2:
            rows of `new_sets` returned by `_pfaffian_matrix` with
            fixed excitation numbers.

            `new_sets1` must include the physical leg too, with
            any necessary adjustments for parity

    Returns:
        the MPS tensor block as a matrix
    """
    # check that every new_set has the same number of particles
    n1 = new_sets1.sum(axis=1)
    assert np.all(n1 == n1[0]), "Bra sets of different excitation numbers supplied"
    n1 = n1[0]

    n2 = new_sets2.sum(axis=1)
    assert np.all(n2 == n2[0]), "Ket sets of different excitation numbers supplied"
    n2 = n2[0]

    # check parity conservation
    assert n1 % 2 == n2 % 2, "Bra and ket excitations do not preserve parity"

    # convert new_sets to lists of positions
    ns1 = len(new_sets1)
    _, new_sets1 = new_sets1.nonzero()
    new_sets1 = new_sets1.reshape(ns1, n1)

    ns2 = len(new_sets2)
    _, new_sets2 = new_sets2.nonzero()
    new_sets2 = new_sets2.reshape(ns2, n2)

    # broadcast and concatenate indices
    idx = np.concatenate(
        (
            np.broadcast_to(np.expand_dims(new_sets2, 0), (ns1, ns2, n2)),
            np.broadcast_to(np.expand_dims(new_sets1, 1), (ns1, ns2, n1)),
        ),
        axis=-1,
    )

    # extract matching rows and columns
    O = N[np.expand_dims(idx, 3), np.expand_dims(idx, 2)]
    O = _many_pfaffian(O)
    return O


#### Tenpy helper functions ####


def _make_legcharge(idx, parity, qconj=+1):
    """Builds a LegCharge object from the given parity slices,
    offset by a reference parity."""
    idx = {(i + parity) % 2: idx[i] for i in idx}
    return npc.LegCharge.from_qdict(chinfo, idx, qconj=qconj)


@dataclass(frozen=True)
class MPSTensorData:
    r"""Data for computing one MPS tensor of a Pfaffian state.

    - If :attr:`mode` is ``"left"``, contains an implicit description of the
      left canonical tensor

      .. math::

        A^{n_i}_{\alpha\beta} =
        (\langle n_i | \otimes_g \langle L^{(i-1)}_{\alpha}|)
        |L^{(i)}_{\beta} \rangle.

    - If :attr:`mode` is ``"right"``, contains an implicit description of the
      right canonical tensor

      .. math::

        B^{n_i}_{\beta\alpha} =
        (\langle R^{(i)}_\alpha| \otimes_g \langle n_i|)
        |R^{(i-1)}_\beta \rangle.
    """

    mode: str
    """Whether the overlap is between ``"left"`` or ``"right"`` Schmidt vectors."""
    norm: float
    """Overlap of the normalised Boguliubov vacua for the two entanglement cuts."""
    pfaffian_matrix: np.ndarray
    """Matrix of Pfaffian entries for all bogulon excitations that appear in
    at least one Schmidt state.
    
    It is a block matrix, with the first rows/columns corresponding to excitations
    on the ket side, followed by excitations on the bra side."""
    labels: list[str, str]
    """Labels of the bra and ket leg(pipe)s."""
    qtotal: int
    """Total fermion parity of the tensor to ensure matching parities in
    Schmidt vectors.
    
    - 0 if ``mode == "left"``.
    - 0 if ``mode == "right"`` and virtual legs are labelled
      with fermion parity to the right.
    - Otherwise, difference of total fermion parity between
      the ket and bra Schmidt vectors.
    """

    leg_bra: npc.LegCharge
    """Tensor leg corresponding to the bra Schmidt vectors.

    If the tensor contains a physical leg, it is united to the bra leg as 
    an `unsorted` :class:`tenpy.linalg.charges.LegPipe`."""
    new_sets_bra: np.ndarray
    """Bra Schmidt vectors in terms of the excitations in :attr:`pfaffian_matrix`.
    
    Given the block structure of :attr:`pfaffian_matrix`, each row starts with
    a number of ``False`` entries corresponding to the ket excitations.
    
    If the tensor contains a physical leg, it is double the length of
    :attr:`~SchmidtVectors.sets` and contains all Schmidt vectors with the on-site
    degree of freedom once empty, one filled.
     
    If needed for matching the bra and ket parities, the Boguliubov orbital
    with λ closest to 1/2 is particle-hole flipped. 

    After all this, the sets are resorted by parity and number of Boguliubov
    excitations."""
    idx_n_bra: np.ndarray
    """Analogue of :attr:`~SchmidtVectors.idx_n` for :attr:`new_sets_bra`."""
    leg_idx_bra: np.ndarray
    """Mapping between :attr:`leg_bra` and :attr:`new_sets_bra`.
    
    Namely, ``leg_idx_bra[i]`` gives the index in :attr:`leg_bra`
    corresponding to ``new_sets_bra[i]``.
    """

    leg_ket: npc.LegCharge
    """Tensor leg corresponding to the ket Schmidt vectors."""
    new_sets_ket: np.ndarray
    """Ket Schmidt vectors in terms of the excitations in :attr:`pfaffian_matrix`.
    
    Given the block structure of :attr:`pfaffian_matrix`, each row ends with
    a number of ``False`` entries corresponding to the bra excitations.
    """
    idx_n_ket: np.ndarray
    """:attr:`~SchmidtVectors.idx_n` of the ket Schmidt vector."""

    @classmethod
    def from_schmidt_vectors(
        cls: Type["MPSTensorData"],
        Schmidt_bra: SchmidtVectors,
        Schmidt_ket: SchmidtVectors,
        mode: str,
        *,
        nambu_tolerance: float = 1e-8,
        min_SV: float = 1e-6,
    ) -> "MPSTensorData":
        """Builds an :class:`MPSTensorData` object from the Schmidt vectors of
        the two surrounding entanglement cuts.

        Parameters
        ----------
        Schmidt_bra:
            Schmidt vectors on the "bra" side.

            That is, for the entanglement cut to the left if ``mode=="left"``
            or to the right if ``mode="right"``.

            Must contain a description of left (right) Schmidt vectors if
            ``mode=="left"`` (``"right"``).
        Schmidt_ket:
            Schmidt vectors on the "ket" side.

            That is, for the entanglement cut to the right if ``mode=="left"``
            or to the left if ``mode="right"``.

            Must contain a description of left (right) Schmidt vectors if
            ``mode=="left"`` (``"right"``).
        mode:
            Specifies which Schmidt vectors are used to build the MPS tensor
            and the canonical form the tensor is computed in.
        nambu_tolerance:
            numerical tolerance for checking Nambu symmetry
        min_SV:
            smallest acceptable singular value of U before it is considered singular
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

        # virtual legs of the MPS tensor
        p_bra = Schmidt_bra.pL
        p_ket = Schmidt_ket.pL
        if (p_bra is None) or (p_ket is None):  # only happens if mode == "right"
            warnings.warn(
                "\nParity to the left is unknown.\n"
                "Virtual legs will carry parity to the right!"
            )
            p_bra = Schmidt_bra.pR
            p_ket = Schmidt_ket.pR
            qtotal = 0
        elif mode == "right":  # we have left and right -> total parities
            qtotal = (Schmidt_bra.parity() + Schmidt_ket.parity()) % 2
        else:  # mode == 'left'
            qtotal = 0
        qconj = (+1, -1) if mode == "left" else (-1, +1)
        labels = ["vL", "vR"] if mode == "left" else ["vR", "vL"]
        leg_bra = _make_legcharge(Schmidt_bra.idx_parity, p_bra, qconj=qconj[0])
        leg_ket = _make_legcharge(Schmidt_ket.idx_parity, p_ket, qconj=qconj[1])

        # bra must be 1 longer than ket
        if len(v_bra) + 2 == len(v_ket):
            ns_bra = len(sets_bra)
            n = len(v_bra) // 2
            # add the physical orbital and combine sets_bra and sets_phys
            # sets_phys shall be more major to sets_bra
            leg_bra = npc.LegPipe(
                [fermion_leg, leg_bra], qconj=leg_bra.qconj, sort=False, bunch=False
            )
            labels[0] = f"(p.{labels[0]})"

            z_col = np.zeros((2 * n, 1))
            z_row = np.zeros((1, n))
            if mode == "left":
                # anticommutation signs:
                # if bra vacuum is parity odd, flip sign of physical leg
                u_p = -1 if Schmidt_bra.parity(mode) % 2 == 1 else 1
                # the physical leg ought to be the last on the bra side
                v_bra = np.block(
                    [
                        [v_bra[:, :n], z_col, v_bra[:, n:], z_col],
                        [z_row, u_p, z_row, 0.0],
                        [z_row, 0.0, z_row, u_p],
                    ]  # NB row space keeps c_i and c†_i together
                )
                sets_bra = np.block(
                    [
                        [sets_bra, np.zeros((ns_bra, 1), bool)],
                        [sets_bra, np.ones((ns_bra, 1), bool)],
                    ]
                )
            else:  # mode == "right":
                # the physical leg ought to be the first on the bra side
                v_bra = np.block(
                    [
                        [1, z_row, 0, z_row],
                        [0, z_row, 1, z_row],
                        [z_col, v_bra[:, :n], z_col, v_bra[:, n:]],
                    ]  # NB row space keeps c_i and c†_i together
                )
                sets_bra = np.block(
                    [
                        [np.zeros((ns_bra, 1), bool), sets_bra],
                        [np.ones((ns_bra, 1), bool), sets_bra],
                    ]
                )
        elif len(v_bra) == len(v_ket):
            # copy v_bra and sets_bra if parity fix needed
            if Schmidt_bra.parity(mode) % 2 != Schmidt_ket.parity(mode) % 2:
                v_bra = v_bra.copy().setflags(write=True)
                sets_bra = sets_bra.copy().setflags(write=True)
        else:
            raise ValueError(
                f"{mode.capitalize()} sides `Schmidt_bra` and `Schmidt_ket` must match\n"
                f"or `Schmidt_bra` must be one bond to the {mode} of `Schmidt_ket`,\n"
                f"got lengths {len(v_bra) // 2} and {len(v_ket) // 2}."
            )

        # fix reference parity
        if Schmidt_bra.parity(mode) % 2 != Schmidt_ket.parity(mode) % 2:
            # flip the "most entangled" bra mode, i.e. last for left, first for right
            n = len(v_bra) // 2
            if mode == "left":
                v_bra[:, [n - 1, -1]] = v_bra[:, [-1, n - 1]]
                sets_bra[:, -1] = np.logical_not(sets_bra[:, -1])
            else:  # mode == "right":
                # anticommutation signs:
                # need to flip sign of all other Boguliubov operators
                v_bra *= -1
                v_bra[:, [0, n]] = -v_bra[:, [n, 0]]
                sets_bra[:, 0] = np.logical_not(sets_bra[:, 0])

        norm, N, sets_bra, sets_ket, _ = _pfaffian_matrix(
            v_bra,
            v_ket,
            sets_bra,
            Schmidt_ket.sets(mode),
            mode=mode,
            tolerance=nambu_tolerance,
            min_SV=min_SV,
        )

        # sort sets_bra
        leg_idx_bra, idx_n_bra, _ = _parity_n_argsort(sets_bra.sum(axis=1))
        sets_bra = sets_bra[leg_idx_bra]

        return cls(
            mode=mode,
            norm=norm,
            pfaffian_matrix=N,
            labels=labels,
            qtotal=qtotal,
            leg_bra=leg_bra,
            leg_ket=leg_ket,
            new_sets_bra=sets_bra,
            new_sets_ket=sets_ket,
            idx_n_bra=idx_n_bra,
            idx_n_ket=Schmidt_ket.idx_n,
            leg_idx_bra=leg_idx_bra,
        )

    def to_npc_array(self) -> npc.Array:
        """Computes the MPS tensor as a TeNPy
        :class:`~tenpy.linalg.np_conserved.Array` object.

        If :attr:`mode` is "left", returns a left canonical tensor;
        if "right", a right canonical one."""

        # empty tensor
        B = npc.zeros(
            [self.leg_bra, self.leg_ket],
            labels=self.labels,
            qtotal=(self.qtotal,),
            dtype=self.pfaffian_matrix.dtype,
        )

        # fill the tensor in blocks of fixed bogulon numbers
        for n_bra, slice_bra in self.idx_n_bra.items():
            for n_ket, slice_ket in self.idx_n_ket.items():
                if (n_bra + n_ket) % 2 == 1:
                    continue

                # compute tensor block
                B[self.leg_idx_bra[slice_bra], slice_ket] = self.norm * _tensor_block(
                    self.pfaffian_matrix,
                    self.new_sets_bra[slice_bra],
                    self.new_sets_ket[slice_ket],
                )

        return B.split_legs()


#### High-level functions ####
#### -------------------- ####


def C_to_MPS(
    C: np.ndarray,
    trunc_par: dict | StoppingCondition,
    *,
    basis: str,
    diag_tol: float = _DIAG_TOL,
    ortho_center: int = None,
) -> networks.MPS:
    r"""
    MPS representation of a Nambu mean-field ground state from its correlation matrix.

    Parameters
    ----------
    C:
        Nambu correlation matrix in the basis indicated by ``basis``.
    trunc_par:
        Specifies which Schmidt states should be kept.

        Must be either a :class:`~temfpy.schmidt_sets.StoppingCondition` object
        or a dictionary with matching keys.

        Only specify the field ``sectors`` if you know what you are doing!!
    basis:
        "M" or "C", indicates whether the correlation matrix is given
        in the Majorana or the complex-fermion basis.
    ortho_center:
        Orthogonality centre of the mixed canonical MPS.
        Midpoint of the chain by default.
    diag_tol:
        Largest allowed offdiagonal matrix element in diagonalised / SVD
        correlation submatrices before an error is raised.

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

    L = len(C) // 2

    # lists for accumulating the tensors and singular values
    tensors = [None] * L
    λs = [None] * (L + 1)

    # Main bipartition, in the middle if not otherwise specified
    ortho_center = ortho_center or L // 2
    logger.info("Central bond %d", ortho_center)
    Schmidt_center = SchmidtVectors.from_correlation_matrix(
        C, ortho_center, trunc_par, basis=basis, diag_tol=diag_tol
    )
    λs[ortho_center] = normalize_SV(Schmidt_center.schmidt_values, logger)
    parity = Schmidt_center.parity()

    # Right half of the chain
    Schmidt = Schmidt_center
    for i in range(ortho_center, L):
        logger.info(f"Site {i}")
        # details of bond to the right
        Schmidt_new = SchmidtVectors.from_correlation_matrix(
            C,
            i + 1,
            trunc_par,
            which="R",
            basis=basis,
            diag_tol=diag_tol,
            total_parity=parity,
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
    Schmidt = Schmidt_center
    for i in reversed(range(ortho_center)):
        logger.info(f"Site {i}")
        # details of bond to the left
        Schmidt_new = SchmidtVectors.from_correlation_matrix(
            C,
            i,
            trunc_par,
            which="L",
            basis=basis,
            diag_tol=diag_tol,
            total_parity=parity,
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
    basis: str,
    diag_tol: float = _DIAG_TOL,
    unitary_tol: float = iMPS._UNITARY_TOL,
    schmidt_tol: float = iMPS._SCHMIDT_TOL,
) -> tuple[networks.MPS, iMPS.iMPSError]:
    r"""iMPS representation of a Nambu mean-field state from correlation matrices.

    The two correlation matrices are expected to represent the ground states
    of a gapped, translation invariant Hamiltonian on two system sizes that
    differ by one repeating unit cell.

    The method is analogous to :func:`.iMPS.MPS_to_iMPS`, with two differences:

    - No explicit MPS tensors are computed for the environment of the iMPS unit
      cell. Instead, the Schmidt vector overlaps needed for gauge fixing are
      computed using the Pfaffian state overlap formulas implemented in
      :class:`MPSTensorData`.
    - The rightmost tensor is computed directly using the right Schmidt vectors
      of the shorter chain. This means that no separate :class:`~.iMPS.iMPSError`\ s
      are returned for the right side.

    Parameters
    ----------
    C_short:
        Nambu correlation matrix in the basis indicated by ``basis``
        for the shorter chain.
    C_long:
        Nambu correlation matrix in the basis indicated by ``basis``
        for the longer chain.
    trunc_par:
        Specifies which Schmidt states should be kept.

        Must be either a :class:`~temfpy.schmidt_sets.StoppingCondition` object
        or a dictionary with matching keys.

        Only specify the field ``sectors`` if you know what you are doing!!
    sites_per_cell:
        Size of the iMPS unit cell.
    cut:
        First site of the repeating unit cell in ``C_long``.
    basis:
        "M" or "C", indicates whether the correlation matrix is given
        in the Majorana or the complex-fermion basis.
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

    L_short = len(C_short) // 2
    err = f"Got correlation matrix of invalid shape {C_short.shape}."
    assert C_short.shape == (2 * L_short, 2 * L_short), err

    L_long = len(C_long) // 2
    err = f"Got correlation matrix of invalid shape {C_long.shape}."
    assert C_long.shape == (2 * L_long, 2 * L_long), err

    assert L_short + sites_per_cell == L_long, (
        "The given two MPS must differ by one unit cell, got "
        f"{L_long} - {L_short} != {sites_per_cell}"
    )

    # lists for accumulating the tensors and singular values
    tensors = []
    λs = []

    # Reference bipartition in the two chains
    Schmidt_short = SchmidtVectors.from_correlation_matrix(
        C_short, cut, trunc_par=trunc_par, diag_tol=diag_tol, basis=basis
    )
    λs.append(normalize_SV(Schmidt_short.schmidt_values, logger))
    Schmidt_long = SchmidtVectors.from_correlation_matrix(
        C_long, cut, trunc_par=trunc_par, diag_tol=diag_tol, basis=basis
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
                C_long,
                cut + i + 1,
                trunc_par,
                which="R",
                diag_tol=diag_tol,
                basis=basis,
                total_parity=Schmidt_long.parity(),
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
