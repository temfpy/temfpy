# Changelog

## TeMFpy 0.3 (In development)

### API breaking changes

* By default, the iMPS creation routines {func}`temfpy.slater.C_to_iMPS`, {func}`temfpy.slater.H_to_iMPS`, and {func}`temfpy.iMPS.MPS_to_iMPS` now subtract an estimate of the total U(1) charge to the left of the extracted unit cell from the quantum numbers on the virtual legs of the iMPS. The amount of this offset can be controlled with the new `offset` parameter; the old behaviour can be restored using `offset=0`. [#20](https://github.com/temfpy/temfpy/pull/20)

### New features

* Gutzwiller projections {func}`temfpy.gutzwiller.abrikosov` and {func}`temfpy.gutzwiller.abrikosov_ph` allow targeting different virtual charge/parity sectors, which may give access to different topological sectors. [#18](https://github.com/temfpy/temfpy/pull/18)
* All functions that create an {class}`~tenpy.networks.mps.MPS` support an optional `unit_cell_width` argument for specifying the physical length of the system being simulated. Sensible defaults are also provided. [#21](https://github.com/temfpy/temfpy/pull/21)

### Bug fixes

* Gutzwiller projections {func}`temfpy.gutzwiller.abrikosov` and {func}`temfpy.gutzwiller.abrikosov_ph` now handle most infinite fermion MPS correctly. [#18](https://github.com/temfpy/temfpy/pull/18)

## TeMFpy 0.2.1 (28 January 2026)

### Bug fixes

* Fixed ill-defined unitary errors during iMPS conversion. [#14](https://github.com/temfpy/temfpy/pull/14)
* Improved the documentation and fixed typos. [#13](https://github.com/temfpy/temfpy/pull/13)

## TeMFpy 0.2 (23 January 2026)

First stable version without severe bugs.
