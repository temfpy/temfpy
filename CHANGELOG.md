# Changelog

## TeMFpy 0.4 (in development)

## TeMFpy 0.3.1 (28 July 2026)

### Bug fixes

* The `offset` parameter is now applied to the gauge-fixing matrix in {func}`temfpy.slater.C_to_iMPS`. [#28](https://github.com/temfpy/temfpy/pull/28)

## TeMFpy 0.3 (21 July 2026)

### API breaking changes

* By default, the iMPS creation routines {func}`temfpy.slater.C_to_iMPS`, {func}`temfpy.slater.H_to_iMPS`, and {func}`temfpy.iMPS.MPS_to_iMPS` now subtract an estimate of the total U(1) charge to the left of the extracted unit cell from the quantum numbers on the virtual legs of the iMPS. The amount of this offset can be controlled with the new `offset` parameter; the old behaviour can be restored using `offset=0`. [#20](https://github.com/temfpy/temfpy/pull/20)

### New features

* Gutzwiller projections {func}`temfpy.gutzwiller.abrikosov` and {func}`temfpy.gutzwiller.abrikosov_ph` allow targeting different virtual charge/parity sectors, which may give access to different topological sectors. [#18](https://github.com/temfpy/temfpy/pull/18)
* All functions that create an {class}`~tenpy.networks.mps.MPS` support an optional `unit_cell_width` argument for specifying the physical length of the system being simulated. Sensible defaults are also provided. [#21](https://github.com/temfpy/temfpy/pull/21)

### Bug fixes

* Gutzwiller projections {func}`temfpy.gutzwiller.abrikosov` and {func}`temfpy.gutzwiller.abrikosov_ph` now handle most infinite fermion MPS correctly. [#18](https://github.com/temfpy/temfpy/pull/18)
* A check in {meth}`temfpy.pfaffian.SchmidtModes.from_correlation_matrix` assumed that at most half of all Nambu eigenvalues are 1/2, which is not always the case. These checks now pass. [#24](https://github.com/temfpy/temfpy/pull/24)
* A bug in the construction of Nambu eigenvectors with eigenvalue 1/2 is now fixed. [#25](https://github.com/temfpy/temfpy/pull/25)
* In certain edge cases involving many 1/2 Nambu-Schmidt eigenvalues, the preference of eigensolvers to return the identity matrix as eigen/singular vectors of the identity matrix resulted in zero overlaps between Boguliubov vacua when constructing {class}`temfpy.pfaffian.MPSTensorData` objects. This issue is mitigated by rotating the real Nambu eigenvectors with eigenvalue 1/2 with a quasi-random orthogonal matrix. This results in a unitary rotation of degenerate Boguliubov-Schmidt states and hence a gauge rotation of the MPS tensors. However, this should have no observable consequences. [#26](https://github.com/temfpy/temfpy/pull/26)

## TeMFpy 0.2.1 (28 January 2026)

### Bug fixes

* Fixed ill-defined unitary errors during iMPS conversion. [#14](https://github.com/temfpy/temfpy/pull/14)
* Improved the documentation and fixed typos. [#13](https://github.com/temfpy/temfpy/pull/13)

## TeMFpy 0.2 (23 January 2026)

First stable version without severe bugs.
