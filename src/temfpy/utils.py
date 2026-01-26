# Copyright (C) TeMFPy Developers, MIT license
"""Various utilities used by the rest of the library."""
import logging

import numpy as np


def HT(M: np.ndarray) -> np.ndarray:
    """Hermitian conjugate of the input array."""
    return M.T.conj()


def n_slice(x: slice) -> int:
    """Number of elements returned by a slice, assuming a very long array."""
    step = x.step or 1
    return (x.stop - x.start) // step


def block_svd(
    CLR: np.ndarray,
    vL: np.ndarray,
    vR: np.ndarray,
    e: np.ndarray,
    degeneracy_tol: float = 1e-12,
    overwrite: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Completes a block singular-value decomposition.

    Assuming the matrix :math:`v_L^\dagger C_{LR} v_R` is block diagonal,
    performs the SVD of each block to obtain singular vectors of ``CLR``.

    Blocks are delineated by approximately-equal values of ``e``.

    Parameters
    ----------
    CLR: :class:`~np.ndarray` (N, M)
        The matrix whose SVD is sought.
    vL: :class:`~np.ndarray` (N, K)
        The left almost-singular vectors of ``CLR``.
    vR: :class:`~np.ndarray` (M, K)
        The right almost-singular vectors of ``CLR``.
    e: :class:`~np.ndarray` (K,)
        Eigenvalues used to delineate blocks.
    degeneracy_tol:
        Threshold for considering consecutive entries of ``e`` equal.

        Rows/columns ``i`` and ``i+1`` are assumed to be in different blocks
        if ``abs(e[i] - e[i+1]) > degeneracy_tol``.
    overwrite:
        Overwrite ``vL`` and ``vR`` with the singular vectors (default: :obj:`True`).

    Returns
    -------
        The singular vectors of ``CLR``.
    """
    # Conformity checks
    err = "Mismatched number of eigenvalues and eigenvectors"
    assert vL.shape[1] == vR.shape[1] == e.size, err
    assert vL.shape[0] == CLR.shape[0], "Mismatched row dimension"
    assert vR.shape[0] == CLR.shape[1], "Mismatched column dimension"

    if e.size == 0:  # nothing to do
        return vL, vR

    if not overwrite:  # need deep copies to protect input data
        vL = vL.copy()
        vR = vR.copy()

    # Group blocks by multiplicity
    # Determine intervals of degenerate eigenvalues
    (split_ix,) = np.nonzero(np.abs(np.diff(e)) > degeneracy_tol)
    split_ix = np.concatenate(([0], split_ix + 1, [len(e)]))

    # Calculate the size of each interval (= dimension of each subspace)
    multiplicities = np.diff(split_ix)

    # only keep the starting indices of each interval
    split_ix = split_ix[:-1]

    # SVD all degeneracy blocks, grouped by multiplicity
    # --------------------------------------------------
    for mult in np.unique(multiplicities):
        # 2d array of indices, each row are the indices of one subspace
        deg_ix = split_ix[multiplicities == mult, None] + np.arange(mult)

        # block of s = vL.T.conj() @ C_LR @ vR_E_I for each subspace
        s_deg_block = np.einsum(
            "kdi,km,mdj->dij", vL[:, deg_ix].conj(), CLR, vR[:, deg_ix]
        )
        U, _, Vh = np.linalg.svd(s_deg_block)

        # apply unitaries to vL, vR
        vL[:, deg_ix] = np.einsum("idk,dkj->idj", vL[:, deg_ix], U)
        vR[:, deg_ix] = np.einsum("idk,djk->idj", vR[:, deg_ix], Vh.conj())

    return vL, vR


def normalize_SV(λ: np.ndarray, logger: logging.Logger) -> np.ndarray:
    """Normalises the input array and prints the norm in the logs."""
    norm = np.linalg.norm(λ)
    logger.info(f"Norm of Schmidt values: {norm}")
    return λ / norm
