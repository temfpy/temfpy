# Copyright (C) TeMFPy Developers, MIT license
r"""Numerical tests with controllable strictness.

The key difference to :mod:`numpy.testing` is the global switch :data:`TEST_ACTION`
that determines whether the tests raise :class:`AssertionError`\ s or issue
:class:`ComparisonWarning`\ s. For speed-critical applications, they may
also be turned off altogether.
"""
import warnings
import numpy as np

from .utils import HT

_DIAG_TOL = 1e-8

TEST_ACTION: str = "warn"
"""Determines how the testing functions in this module (and elsewhere in TeMFpy) behave.

Allowed values:

* ``"raise"``: raise an :class:`AssertionError`
* ``"warn"``: issue a :class:`ComparisonWarning` (default)
* ``"pass"``: don't test at all

The behaviour of the tests reacts dynamically to changing :data:`TEST_ACTION`:

.. code:: python

    from temfpy import testing

    testing.assert_allclose(1, 2)  # prints a warning
    testing.TEST_ACTION = "raise"
    testing.assert_allclose(1, 2)  # raises an exception
    testing.TEST_ACTION = "pass"
    testing.assert_allclose(1, 2)  # nothing happens
"""


class ComparisonWarning(Warning):
    """Generic warning class for failed equality testing or comparison."""


def assert_allclose(
    actual: np.ndarray,
    desired: np.ndarray,
    rtol: float = 1e-7,
    atol: float = 0.0,
    equal_nan: bool = True,
    err_msg: str = "",
    verbose: bool = False,
    *,
    strict: bool = False,
):
    """Indicates if the two objects are not equal up to desired tolerance.

    See :func:`np.testing.assert_allclose` for details of the calling sequence.

    Raises
    ------
    AssertionError | ComparisonWarning
        If the shapes of the two arrays don't match or any two entries
        deviate by more than the given tolerance.
    """
    if TEST_ACTION == "raise":
        np.testing.assert_allclose(
            actual, desired, rtol, atol, equal_nan, err_msg, verbose, strict=strict
        )
    elif TEST_ACTION == "warn":
        try:
            np.testing.assert_allclose(
                actual, desired, rtol, atol, equal_nan, "", verbose, strict=strict
            )
        except AssertionError as err:
            warnings.warn("\n" + err_msg + str(err), category=ComparisonWarning)
    elif TEST_ACTION != "pass":
        raise ValueError(
            f"Invalid value {TEST_ACTION!r} of `temfpy.testing.TEST_ACTION`,\n"
            "must be one of 'raise', 'warn', 'pass'."
        )


def assert_array_less(
    x: np.ndarray,
    y: np.ndarray,
    err_msg: str = "",
    verbose: bool = False,
    *,
    strict: bool = False,
):
    """Indicates if the first object is not elementwise smaller than the second.

    See :func:`np.testing.assert_array_less` for details of the calling sequence.

    Raises
    ------
    AssertionError | ComparisonWarning
        If the shapes of the two arrays don't match or any entry of ``x`` is
        not less than the corresponding entry of ``y``.
    """
    if TEST_ACTION == "raise":
        np.testing.assert_array_less(x, y, err_msg, verbose, strict=strict)
    elif TEST_ACTION == "warn":
        try:
            np.testing.assert_array_less(x, y, "", verbose, strict=strict)
        except AssertionError as err:
            warnings.warn("\n" + err_msg + str(err), category=ComparisonWarning)
    elif TEST_ACTION != "pass":
        raise ValueError(
            f"Invalid value {TEST_ACTION!r} of `temfpy.testing.TEST_ACTION`!\n"
            "Must be one of 'raise', 'warn', 'pass'."
        )


def check_schmidt_decomposition(modes, C: np.ndarray, diag_tol: float = _DIAG_TOL):
    """Checks if the given Schmidt modes and correlation matrix are consistent.

    Parameters
    ----------
    modes: :class:`slater.SchmidtModes` | :class:`pfaffian.SchmidtModes`
        Schmidt modes to check
    C:
        (Nambu) correlation matrix
    diag_tol:
        Maximum absolute deviation from diagonalisation / SVD before error is raised.

    Raises
    ------
    AssertionError | ComparisonWarning
        If any entry of the correlation matrix deviates from its
        reconstruction from ``modes`` by more than ``diag_tol``.
    """
    if TEST_ACTION == "pass":  # no need to do any computation
        return

    tol = dict(rtol=0, atol=diag_tol)
    # C_LL
    if modes.vL is not None:
        N = len(modes.vL)

        err = "vL is not unitary"
        assert_allclose(modes.vL @ HT(modes.vL), np.eye(N), **tol, err_msg=err)

        CLL = (modes.eigenvalues("L") * modes.vL) @ HT(modes.vL)
        assert_allclose(CLL, C[:N, :N], **tol, err_msg="vL does not diagonalise C_LL")
    # C_RR
    if modes.vR is not None:
        M = len(modes.vR)
        n = len(C) - M

        err = "vR is not unitary"
        assert_allclose(modes.vR @ HT(modes.vR), np.eye(M), **tol, err_msg=err)

        CRR = (modes.eigenvalues("R") * modes.vR) @ HT(modes.vR)
        assert_allclose(CRR, C[n:, n:], **tol, err_msg="vR does not diagonalise C_RR")
    # C_LR
    if (modes.vL is not None) and (modes.vR is not None):
        assert n == N, f"Inconsistent sizes ({N} + {M} != {len(C)})"
        SV = modes.singular_values
        CLR = (SV * modes.vL_entangled) @ HT(modes.vR_entangled[:, ::-1])
        assert_allclose(CLR, C[:N, N:], **tol, err_msg="vL and vR do not SVD C_LR")
