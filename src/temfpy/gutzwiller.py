# Copyright (C) TeMFPy Developers, MIT license
r"Tools for Gutzwiller projecting MPS to a smaller on-site Hilbert space."

import logging
import warnings

import numpy as np

from tenpy import networks
import tenpy.linalg as npc

logger = logging.getLogger(__name__)


def abrikosov_ph(
    mps: networks.MPS,
    *,
    inplace: bool = False,
    return_canonical: bool = True,
    cutoff: float = 1e-12,
) -> None | networks.MPS:
    r"""Gutzwiller projection from Abrikosov fermions to a spin-1/2 Hilbert space.

    The input MPS is assumed to describe  Abrikosov fermions, with the down
    spins particle-hole transformed; i.e.:

    .. math::

        c_{i,\uparrow} := f_{i,\uparrow},
        \qquad \qquad
        c_{i,\downarrow} := f_{i,\downarrow}^\dagger.

    Therefore, it must contain an even number of spinless fermion sites:
    sites :math:`2i` and :math:`2i+1` represent modes :math:`c_{i\uparrow}` and
    :math:`c_{i\downarrow}`, respectively.
    These pairs are projected to a spin-1/2 Hilbert space using the following rules:

    - Zero occupation → spin-down state
    - Single occupation → unphysical states, dropped
    - Double occupation → spin-up state

    Therefore, depending on the conserved charge of the input MPS, only the following charge blocks
    of the virtual legs are kept:

    - ``'N'`` (particle number) → even ``'N'`` blocks → ``'S_z'`` conserved
    - ``'parity'`` → even ``'parity'`` blocks → no conserved charge

    Parameters
    ----------
    mps:
        MPS representing the wave function to be projected.
        Must be of even length and every site must be an instance
        of :class:`~tenpy.networks.site.FermionSite`.
    inplace:
        Whether to transform the original MPS in place.
    return_canonical:
        Whether to transform the output MPS to right canonical form.
    cutoff:
        Cutoff for Schmidt values to keep in the canonical form.

    Returns
    -------
        The Gutzwiller projected ``mps``, if ``inplace`` is :obj:`False`.

    Note
    ----
        Currently, no symmetry quantum numbers other than fermion
        number or parity can be handled.
    """

    assert (
        mps.L % 2 == 0
    ), "Odd-length MPS cannot represent an Abrikosov fermion Hilbert space"
    # TODO: allow grouped sites which include a FermionicSite
    assert isinstance(
        mps.sites[0], networks.FermionSite
    ), f"All sites must be fermionic, found: {mps.sites[0]}"

    def gen_leg_mask(leg: npc.charges.LegCharge) -> np.ndarray:
        """Generates a mask selecting the physical charge blocks for a given
        fermionic leg.

        The physical charge blocks depend on the conserved charge of the
        given ``leg``:

        - ``'N'`` → even particle charge blocks
        - ``'parity'`` → even parity charge block


        Parameters
        ----------
        leg:
            The fermionic leg for which the mask is generated.

        Returns
        -------
            A boolean mask selecting the physical charge blocks
            that can be used by :class:`~tenpy.networks.Array.iproject`.

        """
        mask = (leg.to_qflat() % 2 == 0).ravel()

        return mask

    if not inplace:
        mps = mps.copy()
        logger.debug(f"Deep copied MPS before Gutzwiller projection.")

    conserved_fermion = mps.sites[0].conserve
    if conserved_fermion == "N":
        conserved_spin = "Sz"
    elif conserved_fermion == "parity":
        conserved_spin = None
    else:
        raise ValueError(
            f"FermionSite must conserve either 'N' or 'parity', found {conserved_fermion}"
        )

    # TeNPy bindings
    spin_site = networks.SpinHalfSite(conserved_spin)
    spin_leg = spin_site.leg
    chinfo_s = spin_leg.chinfo

    # We start by grouping neighboring sites
    # This will result in LegPipe objects for all physical legs
    mps.group_sites(2)

    # The mask for the physical leg is independent of the site
    mask_p = gen_leg_mask(mps._B[0].get_leg("p"))

    for idx, B in enumerate(mps._B):
        # Remove LegPipe structure
        B.legs[B.get_leg_index("p")] = B.get_leg("p").to_LegCharge()

        mask_vL = gen_leg_mask(B.get_leg("vL"))
        mask_vR = gen_leg_mask(B.get_leg("vR"))

        # Change the occupation number leg charges to spin charges
        # --------------------------------------------------------

        B.iproject([mask_vL, mask_p, mask_vR], ["vL", "p", "vR"])

        # Change the occupation number leg charges to spin charges,
        # if conserved.
        # ------------------------------------------------------------
        if conserved_spin is "Sz":
            B.chinfo = chinfo_s

            leg_vL, leg_p, leg_vR = [B.get_leg(label) for label in ["vL", "p", "vR"]]

            leg_p.chinfo = chinfo_s
            leg_p.charges = spin_leg.charges

            leg_vL.chinfo = chinfo_s
            leg_vL.charges -= idx

            leg_vR.chinfo = chinfo_s
            leg_vR.charges -= idx + 1

        else:  # None
            B = B.drop_charge(charge="parity_N", chinfo=chinfo_s)

    mps.chinfo = chinfo_s
    mps.grouped = 1
    mps.sites = [spin_site] * mps.L

    # Transform into right canoncial form
    mps.form = [None] * mps.L
    mps._S = [None] * (mps.L + 1)

    logger.info(
        "Completed projection to spin-1/2 space. Conserved charge is now %s",
        conserved_spin,
    )

    if return_canonical:
        mps.canonical_form(cutoff=cutoff)
        logger.info("Transformed MPS to right canonical form")
    else:
        warnings.warn(
            "The MPS is not in canonical form after Gutzwiller projection.\n"
            "Consider setting 'return_canonical=True'",
        )

    if not inplace:
        return mps
