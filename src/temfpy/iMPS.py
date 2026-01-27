# Copyright (C) TeMFPy Developers, MIT license
r"""Tools for converting finite to infinite MPS."""

import logging
import warnings
from typing import NamedTuple

import numpy as np

import tenpy.linalg.np_conserved as npc
from tenpy import networks as nw

logger = logging.getLogger(__name__)

_NUMERICAL_TOL = 1e-14
_UNITARY_TOL = 1e-6
_SCHMIDT_TOL = 1e-6


def overlap_schmidt(bra: nw.mps.MPS, ket: nw.mps.MPS, mode: str) -> npc.Array:
    r"""Overlap or optimal basis rotation between two sets of Schmidt vectors.

    Parameters
    ----------
    bra:
        Bra Schmidt vectors.
    ket:
        Ket Schmidt vectors.
    mode:
        If the overlap is between "left" or "right" Schmidt vectors.

    Returns
    -------
    overlap: :class:`~tenpy.linalg.np_conserved.Array`
        The overlaps :math:`(C_0)_{\alpha\beta} = \langle L_\alpha'|L_\beta \rangle`
        or :math:`(D_0)_{\alpha\beta} = \langle R_\beta'|R_\alpha \rangle`,
        depending on ``mode``.

        Legs are labelled ``"vL"`` (incoming) and ``"vR"`` (outgoing).
    """
    assert bra.L == ket.L, "The two MPS have different lengths."
    mode = mode.lower()

    # ---- Compute overlap ----
    if mode == "left":
        TM = nw.TransferMatrix(bra, ket, transpose=True, form="A")
    elif mode == "right":
        TM = nw.TransferMatrix(bra, ket, transpose=False, form="B", charge_sector=None)
    else:
        raise ValueError("`mode` must be either 'left' or 'right', got " + repr(mode))

    # Identity matrix with legs matching the input legs of the transfer matrix
    Id = npc.Array.from_ndarray([[1.0]], TM.pipe.legs, labels=TM.label_split)

    # nw.TransferMatrix makes sure that the first legtwo_leg is always ingoing and
    # the second leg is always outgoing.
    # Therefore, it is safe to just set the labels here instead of using
    # ireplace_labels.
    overlap = TM.matvec(Id).iset_leg_labels(["vL", "vR"])

    return overlap


def basis_rotation(
    overlap: npc.Array,
    Schmidt_bra: np.ndarray,
    Schmidt_ket: np.ndarray,
    mode: str,
    form: str = "B",
    unitary_tol: float = _UNITARY_TOL,
    schmidt_tol: float = _SCHMIDT_TOL,
) -> tuple[npc.Array, float, float]:
    r"""Overlap or optimal basis rotation between two sets of Schmidt vectors.

    Parameters
    ----------
    overlap:
        The overlaps :math:`(C_0)_{\alpha\beta} = \langle L_\alpha'|L_\beta \rangle`
        or :math:`(D_0)_{\alpha\beta} = \langle R_\beta'|R_\alpha \rangle`,
        depending on ``mode``.

        Legs should be labelled ``"vL"`` (incoming) and ``"vR"`` (outgoing).
    Schmidt_bra:
        Schmidt values corresponding to the bra Schmidt vectors :math:`\langle L'|`
        or :math:`\langle R'|`.
    Schmidt_ket:
        Schmidt values corresponding to the ket Schmidt vectors :math:`|L \rangle`
        or :math:`|R \rangle`.
    mode:
        If the overlap is between "left" or "right" Schmidt vectors.
    form:
        Whether the basis rotation is to be used for a left ("A")
        or a right ("B", default) canonical MPS tensor.
    unitary_tol:
        Highest allowed deviation from unitarity (weighted with Schmidt values)
        in the overlaps before a warning is raised.
    schmidt_tol:
        Highest allowed mixing between unequal Schmidt value sectors
        before a warning is raised.

    Returns
    -------
    rotation_matrix: :class:`~tenpy.linalg.np_conserved.Array`
        If :obj:`True`, the optimal unitary basis rotation matrix :math:`C` that
        minimises the iMPS conversion error.

        Legs are labelled ``"vL"`` (incoming) and ``"vR"`` (outgoing).
    unitary_error: float
        Deviation of the overlap matrix from unitarity,
        measured as the square root of the trace of
        :math:`S_{\rm ket} (C_0^\dagger C_0 - \mathbb{I}) S_{\rm ket}`.
    schmidt_error: float
        Degree of mixing of Schmidt vectors with unequal Schmidt values,
        measured as the norm of either :math:`S_{\rm bra} C - C_0 S_{\rm ket}`
        or :math:`(C - C_0) S_{\rm ket}`, depending on ``mode`` and ``form``.
    """
    mode = mode.lower()
    err = f"`mode` must be either 'left' or 'right', got {mode!r}"
    assert mode in ["left", "right"], err

    form = form.upper()
    assert form in ["A", "B"], f"`form` must be either 'A' or 'B', got {form!r}"

    # make what follows independent of mode
    v_bra, v_ket = ("vL", "vR") if mode == "left" else ("vR", "vL")

    # ---- Test unitarity ----
    # C @ S_ket
    C_Sk = overlap.scale_axis(Schmidt_ket, v_ket)
    # unitary_error^2 = tr(S_ket^2 - S_ket @ C^dagger @ C @ S_ket)
    unitary_error_square = np.sum(Schmidt_ket**2) - npc.inner(C_Sk, C_Sk, do_conj=True)

    if unitary_error_square < 0:
        assert abs(unitary_error_square) < _NUMERICAL_TOL, (
            f"{mode.capitalize()} devitation from unitary: The square of the "
            f"unitary error {unitary_error_square} is negative and beyond "
            f"the numerical tolerance {_NUMERICAL_TOL:.1e}"
        )
        logging.info(
            f"{mode.capitalize()} devitation from unitary: The square of the "
            f"unitary error {unitary_error_square:.4e} is negative but within "
            f"the numerical tolerance {_NUMERICAL_TOL:.1e}, setting it to zero."
        )
        unitary_error = 0.0
    else:
        unitary_error = np.sqrt(unitary_error_square)
        logging.info(f"{mode.capitalize()} deviation from unitary: {unitary_error:.4e}")

    if unitary_error > unitary_tol:
        warnings.warn(
            f"\n{mode.capitalize()} overlap matrix deviates from unitarity by "
            f"{unitary_error}.\n"
            "Increasing the bond dimension may be useful."
        )

    # ---- Convert to unitary rotation matrix ----
    if (mode, form) in [("left", "A"), ("right", "B")]:
        # Schmidt values are inserted into the mixed canonical form
        # at this entanglement cut. => Orthogonal Procustes for S_bra @ C @ S_ket
        U, _, V = npc.svd(C_Sk.scale_axis(Schmidt_bra, v_bra))
    else:
        # Schmidt values are inserted into the mixed canonical form
        # far from this entanglement cut. => Orthogonal Procustes for C @ S_ket^2
        U, _, V = npc.svd(C_Sk.scale_axis(Schmidt_ket, v_ket))
    overlap = npc.tensordot(U, V, 1)

    # ---- Test Schmidt value deviations ----
    # As above, the difference is due to whether Schmidt values appear in
    # the mixed canonical form at this cut or far away
    if (mode, form) in [("left", "A"), ("right", "B")]:
        Sb_C = overlap.scale_axis(Schmidt_bra, v_bra)
    else:
        # not just C_Sk bc `overlap` has changed
        Sb_C = overlap.scale_axis(Schmidt_ket, v_ket)

    schmidt_error = npc.norm(Sb_C - C_Sk)
    logging.info(f"{mode.capitalize()} Schmidt value mixing:   {schmidt_error:.4e}")
    if schmidt_error > schmidt_tol:
        warnings.warn(
            f"\nMixing between unequal Schmidt value sectors on the {mode} side is\n"
            f"{schmidt_error}. Increasing the number of sites may help."
        )

    return overlap, unitary_error, schmidt_error


class iMPSError(NamedTuple):
    """Container of the approximation errors accrued by :func:`MPS_to_iMPS`.

    If printed, all non-zero approximation errors are displayed.
    """
    left_unitary: float
    """Deviation of left environment from unitarity."""
    left_schmidt: float
    """Mixing between unequal Schmidt values by the left environment."""
    right_unitary: float
    """Deviation of right environment from unitarity."""
    right_schmidt: float
    """Mixing between unequal Schmidt values by the right environment."""

    @property
    def left_total(self) -> float:
        """Total approximation error of the left environment."""
        return (self.left_schmidt**2 + self.left_unitary**2) ** 0.5

    @property
    def right_total(self) -> float:
        """Total approximation error of the right environment."""
        return (self.right_schmidt**2 + self.right_unitary**2) ** 0.5

    @property
    def total_error(self) -> float:
        """Total approximation error."""
        return np.linalg.norm(self)

    def __repr__(self) -> str:
        fields = [f"    {f}={x:.8e}" for f, x in zip(self._fields, self) if x != 0]
        if len(fields) == 0:
            return "iMPSError()"
        else:
            return "iMPSError(\n" + (",\n".join(fields)) + "\n)"


def MPS_to_iMPS(
    mps_short: nw.MPS,
    mps_long: nw.MPS,
    sites_per_cell: int,
    cut: int,
    unitary_tol: float = _UNITARY_TOL,
    schmidt_tol: float = _SCHMIDT_TOL,
) -> tuple[nw.MPS, iMPSError]:
    """Constructs an iMPS by comparing two finite MPS.

    The two MPS are expected to represent the ground states of a gapped,
    translation invariant Hamiltonian on two system sizes that differ by
    one repeating unit cell.

    For sufficiently large systems, therefore, they are of the form

    .. code::

        ...(A...B)(A...B)...
        ...(A...B)(A...B)(A...B)...

    up to gauge transformations. The repeating unit cell ``(A...B)`` is
    extracted from the longer chain, and its gauge is fixed by comparing its
    left and right environments to the Schmidt vectors of the shorter chain.

    Parameters
    ----------
    mps_short:
        MPS of the shorter chain.
    mps_long:
        MPS of the longer chain.
    sites_per_cell:
        Size of the iMPS unit cell.
    cut:
        First site of the repeating unit cell in ``mps_long``.
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
    validation_metric: :class:`iMPSError`
        Errors introduced during the conversion.
    """
    # preliminary checks
    L_short, L_long = mps_short.L, mps_long.L
    if L_short + sites_per_cell != L_long:
        raise ValueError(
            "The given two MPS must differ by one unit cell, got "
            f"{L_long} - {L_short} != {sites_per_cell}"
        )
    if mps_short.chinfo != mps_long.chinfo:
        raise ValueError("Incompatible ChargeInfo in the two MPS")
    assert all(x is not None for x in mps_short.form), "mps_short is not canonical"
    assert all(x is not None for x in mps_long.form), "mps_long is not canonical"

    # TODO: In TenPy unit_cell_width for a segment is
    # TODO: not set correctly. If this is fixed by TenPy, remove workaround
    # ------------------------
    mps_short.unit_cell_width = mps_short.L
    mps_long.unit_cell_width = mps_long.L
    # ------------------------

    # Schmidt values in the short chain at the reference cut
    S0 = mps_short.get_SL(cut)

    # Left gauge fixing matrix C
    bra = mps_short.extract_segment(0, cut - 1)
    ket = mps_long.extract_segment(0, cut - 1)
    # TODO: In TenPy unit_cell_width for a segment is
    # TODO: not set correctly. If this is fixed by TenPy, remove workaround
    # ------------------------
    bra.unit_cell_width = bra.L
    ket.unit_cell_width = ket.L
    # ------------------------
    S_ket = mps_long.get_SL(cut)
    C = overlap_schmidt(bra, ket, mode="left")
    C, left_unitary, left_schmidt = basis_rotation(
        C, S0, S_ket, mode="left", unitary_tol=unitary_tol, schmidt_tol=schmidt_tol
    )

    # Right gauge fixing matrix D
    bra = mps_short.extract_segment(cut, L_short - 1)
    ket = mps_long.extract_segment(cut + sites_per_cell, L_long - 1)
    # TODO: In TenPy unit_cell_width for a segment is
    # TODO: not set correctly. If this is fixed by TenPy, remove workaround
    # ------------------------
    bra.unit_cell_width = bra.L
    ket.unit_cell_width = ket.L
    # ------------------------

    S_ket = mps_long.get_SL(cut + sites_per_cell)
    D = overlap_schmidt(bra, ket, mode="right")
    D, right_unitary, right_schmidt = basis_rotation(
        D, S0, S_ket, mode="right", unitary_tol=unitary_tol, schmidt_tol=schmidt_tol
    )

    # Extract middle section of MPS in right canonical form
    sites = mps_long.sites[cut : cut + sites_per_cell]
    tensors = [mps_long.get_B(cut + i, form="B") for i in range(sites_per_cell)]
    schmidt_values = mps_long._S[cut + 1 : cut + sites_per_cell]

    # Apply gauge unitaries to first and last tensor
    tensors[0] = npc.tensordot(C, tensors[0], axes=["vR", "vL"])
    tensors[-1] = npc.tensordot(tensors[-1], D, axes=["vR", "vL"])

    # Set Schmidt values on both ends to that of the reference MPS
    schmidt_values = [S0] + schmidt_values + [S0]

    iMPS = nw.MPS(sites, tensors, schmidt_values, bc="infinite", form="B")
    error = iMPSError(left_unitary, left_schmidt, right_unitary, right_schmidt)
    return iMPS, error
