gutzwiller
==========

.. automodule:: temfpy.gutzwiller

.. rubric:: Gutzwiller projections for Abrikosov fermions

.. autosummary::
    :signatures: none

    abrikosov
    abrikosov_ph

Gutzwiller projections for Abrikosov fermions
---------------------------------------------
The input MPS is assumed to describe Abrikosov fermions :math:`f_{i,\sigma}` and
each site within the :class:`~tenpy.networks.mps.MPS` has to contain
an instance of :class:`~tenpy.networks.site.FermionSite` (spinless fermionic site).

.. note::
    Currently, no symmetry quantum numbers other than fermion number or parity 
    can be handled, i.e. :class:`~tenpy.networks.site.GroupedSite` is not supported.

Helper functions
^^^^^^^^^^^^^^^^

.. autofunction:: temfpy.gutzwiller.parity_mask

.. autofunction:: temfpy.gutzwiller.number_mask

Gutzwiller projections
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: temfpy.gutzwiller.abrikosov

.. autofunction:: temfpy.gutzwiller.abrikosov_ph

