# TeMFpy: Tensor-Network Mean-Field Python

[TeMFpy](https://github.com/temfpy/temfpy) (**Te**nsor Networks for **M**ean-**F**ield States in **Py**thon, suggested pronuncation: ˈtɛmɸ.paɪ) is a [TeNPy](https://tenpy.readthedocs.io/en/latest/)-based Python library for converting fermionic mean-field states to finite or infinite matrix product state form. TeMFpy includes new, efficient, and easy-to-understand algorithms for both Slater determinants and Pfaffian states. Together with Gutzwiller projection, these also allow the user to build variational wave functions for various strongly correlated electron systems, such as quantum spin liquids. Being built on top of TeNPy, it integrates seamlessly with existing MPS-based algorithms.

## Installation
TeMFpy runs on all operating systems and requires Python 3.10 or later. We support installation through pip:

```bash
pip install --upgrade pip
pip install --upgrade temfpy
```

We will set up installation via Conda in the near future.

If you use Windows and want to use the Pfaffian-state functionality, [make sure you install the `pfapack` dependency correctly!](https://pfapack.readthedocs.io/en/latest/\#usage)

## Documentation

The documentation is available [on this website](https://temfpy.github.io/temfpy). In addition, every object in TeMFpy comes with detailed docstrings that are available through the `help()` function in Python.

## Contribute

The most up-to-date version of TeMFpy is available on [GitHub](https://github.com/temfpy/temfpy). If you find any bugs in the library, please let us know by [opening an issue](https://github.com/temfpy/temfpy/issues), including a minimal example to reproduce the issue.

We also welcome contributions, whether bug fixes or new features, please see our [contributor guidelines](https://temfpy.github.io/temfpy/getting_started/contribute.html) for details.

## Citing TeMFpy

If you find the **TeMFpy** project useful for your research, please consider citing the following software paper in order to help us continue devoting time and resources to the **TeMFpy** development:

```bibtex
@Article{temfpy,
	title={{TeMFpy: a Python library for converting fermionic mean-field states into tensor networks}},
	author={Simon Hans Hille and Attila Szabó},
	journal={TODO},
	pages={TODO},
	year={2025},
	publisher={TODO},
	doi={TODO},
	url={TODO},
}
```

Please consider citing [the underlying **TeNPy** library](https://scipost.org/10.21468/SciPostPhysCodeb.41) as well:

```bibtex
@Article{tenpy,
    title={{Tensor network Python (TeNPy) version 1}},
    author={Johannes Hauschild and Jakob Unfried and Sajant Anand and Bartholomew Andrews and Marcus Bintz and Umberto Borla and Stefan Divic and Markus Drescher and Jan Geiger and Martin Hefel and Kévin Hémery and Wilhelm Kadow and Jack Kemp and Nico Kirchner and Vincent S. Liu and Gunnar Möller and Daniel Parker and Michael Rader and Anton Romen and Samuel Scalet and Leon Schoonderwoerd and Maximilian Schulz and Tomohiro Soejima and Philipp Thoma and Yantao Wu and Philip Zechmann and Ludwig Zweng and Roger S. K. Mong and Michael P. Zaletel and Frank Pollmann},
    journal={SciPost Phys. Codebases},
    pages={41},
    year={2024},
    publisher={SciPost},
    doi={10.21468/SciPostPhysCodeb.41},
    url={https://scipost.org/10.21468/SciPostPhysCodeb.41},
}
```

## Licence

[MIT Licence](https://github.com/temfpy/temfpy/blob/master/LICENSE)
